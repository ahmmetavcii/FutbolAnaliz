"""Single-flow analysis driver behind the simplified Streamlit panel.

One call to :func:`run_analysis` performs the entire user-facing flow —
probe, prepare, real scheduler run (with the existing pipeline adapter),
jersey inference, post-processing, exports — while a background thread
writes a heartbeat every 5 seconds so the panel can show live, *real*
progress and detect stalls. Progress is derived from artifacts the pipeline
actually wrote (stage manifests, chunk manifests, driver phase); nothing is
estimated from wall-clock time alone.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .manifest import atomic_write_json

# ---------------------------------------------------------------------------
# Progress model: user-facing stages with percent ranges (spec-defined).
# ---------------------------------------------------------------------------

PANEL_STAGES: tuple[tuple[str, str, float, float], ...] = (
    ("preparing", "Video hazırlanıyor", 0.0, 5.0),
    ("validating", "Video doğrulanıyor", 5.0, 10.0),
    ("detection", "Oyuncular tespit ediliyor", 10.0, 32.0),
    ("tracking", "Oyuncular takip ediliyor", 32.0, 50.0),
    ("calibration", "Saha kalibrasyonu", 50.0, 58.0),
    ("identity", "Oyuncu kimlikleri ve roller", 58.0, 66.0),
    ("jersey", "Forma numarası", 66.0, 72.0),
    ("metrics", "Hız ve mesafe", 72.0, 78.0),
    ("events", "Maç olayları inceleniyor", 78.0, 88.0),
    ("render", "Sonuç videosu hazırlanıyor", 88.0, 95.0),
    ("reports", "Excel ve raporlar hazırlanıyor", 95.0, 100.0),
)

STAGE_LABELS = {key: label for key, label, _, _ in PANEL_STAGES}

#: MVP-2 pipeline stage name -> panel stage key. A pipeline stage marks its
#: panel stage complete when its stage_manifest.json exists on disk.
PIPELINE_TO_PANEL_STAGE: tuple[tuple[str, str], ...] = (
    ("ingest", "validating"),
    ("shot_classification", "validating"),
    ("detection", "detection"),
    ("tracking", "tracking"),
    ("track_quality", "tracking"),
    ("reid", "identity"),
    ("camera_motion", "calibration"),
    ("calibration", "calibration"),
    ("team_identity", "identity"),
    ("ball_state", "metrics"),
    ("possession", "metrics"),
    ("metrics", "metrics"),
    ("analytics_render", "render"),
    ("event_detection", "events"),
)

HEARTBEAT_INTERVAL_SECONDS = 5.0
HEARTBEAT_NAME = "heartbeat.json"
STATE_NAME = "analysis_state.json"
ACTIVE_POINTER_NAME = ".active_analysis.json"


def stage_percent(stage_key: str, fraction_within: float = 0.0) -> float:
    """Map a panel stage plus intra-stage fraction to an overall percent."""
    for key, _, start, end in PANEL_STAGES:
        if key == stage_key:
            fraction = min(max(fraction_within, 0.0), 1.0)
            return round(start + (end - start) * fraction, 1)
    return 0.0


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Process status helpers (used by both driver and panel).
# ---------------------------------------------------------------------------


def pid_status(pid: int | None) -> str:
    """Classify a PID as ``running``, ``zombie``, or ``dead`` (never lies)."""
    if not pid or pid <= 0:
        return "dead"
    try:
        import psutil

        process = psutil.Process(int(pid))
        if process.status() == psutil.STATUS_ZOMBIE:
            return "zombie"
        return "running"
    except Exception:
        return "dead"


def reap_children() -> int:
    """Reap any exited direct children so zombies never linger under the panel."""
    reaped = 0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except (ChildProcessError, OSError):
            break
        if pid == 0:
            break
        reaped += 1
    return reaped


def classify_heartbeat_age(age_seconds: float | None) -> str:
    """Spec status buckets for the time since the last heartbeat."""
    if age_seconds is None:
        return "Bilgi yok"
    if age_seconds <= 30:
        return "Çalışıyor"
    if age_seconds <= 90:
        return "Yavaş çalışıyor"
    if age_seconds <= 180:
        return "Yanıt bekleniyor"
    return "İşlem takılmış olabilir"


def should_offer_resume(state: dict[str, Any], heartbeat_age: float | None) -> bool:
    """Resume is offered only when work genuinely stopped, never while healthy."""
    status = str(state.get("status", "")).upper()
    if status == "FAILED":
        return True
    if status == "RUNNING":
        pid = state.get("pid")
        if pid_status(pid) != "running":
            return True
        if heartbeat_age is not None and heartbeat_age > 180:
            return True
    return False


def can_start_new_run(results_root: Path) -> tuple[bool, dict[str, Any]]:
    """Duplicate-process guard: a live analysis blocks starting another one."""
    pointer = load_active_pointer(results_root)
    if not pointer:
        return True, {}
    if pid_status(pointer.get("pid")) == "running":
        state = read_state(Path(pointer.get("match_dir", "")))
        if str(state.get("status", "RUNNING")).upper() in {"RUNNING", "STARTING"}:
            return False, pointer
    return True, pointer


# ---------------------------------------------------------------------------
# State and pointer persistence.
# ---------------------------------------------------------------------------


def write_state(match_dir: Path, **updates: Any) -> dict[str, Any]:
    state = read_state(match_dir)
    state.update(updates)
    state["updated_at"] = _utc_now()
    atomic_write_json(Path(match_dir) / STATE_NAME, state)
    return state


def read_state(match_dir: Path) -> dict[str, Any]:
    path = Path(match_dir) / STATE_NAME
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_active_pointer(results_root: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(Path(results_root) / ACTIVE_POINTER_NAME, payload)


def load_active_pointer(results_root: Path) -> dict[str, Any]:
    path = Path(results_root) / ACTIVE_POINTER_NAME
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def read_heartbeat(match_dir: Path) -> dict[str, Any]:
    path = Path(match_dir) / "run" / HEARTBEAT_NAME
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def heartbeat_age_seconds(heartbeat: dict[str, Any]) -> float | None:
    stamp = heartbeat.get("timestamp")
    if not stamp:
        return None
    try:
        then = dt.datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return max(0.0, (dt.datetime.now(dt.timezone.utc) - then).total_seconds())


# ---------------------------------------------------------------------------
# Real progress computation from on-disk artifacts.
# ---------------------------------------------------------------------------


def _latest_pipeline_run_dir(match_dir: Path) -> Path | None:
    candidates = sorted(Path(match_dir).glob("chunk_artifacts/*/*/pipeline_runs/run_*"))
    return candidates[-1] if candidates else None


def _tail_line(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return text.rsplit("\n", 1)[-1][-300:] if text else ""


def compute_progress(match_dir: Path) -> dict[str, Any]:
    """Derive percent / stage / frames from real driver + pipeline artifacts."""
    match_dir = Path(match_dir)
    state = read_state(match_dir)
    phase = state.get("phase", "preparing")
    total_frames = state.get("total_frames")
    processed_frames = None
    stage_key = phase if phase in STAGE_LABELS else "preparing"

    if phase == "pipeline":
        # Completed pipeline stages tell us the frontier of real work.
        stage_key = "detection"
        pipeline_run = _latest_pipeline_run_dir(match_dir)
        if pipeline_run is not None:
            done_panel_stages: list[str] = []
            current = None
            for pipeline_stage, panel_stage in PIPELINE_TO_PANEL_STAGE:
                manifest = pipeline_run / "stages" / pipeline_stage / "stage_manifest.json"
                if manifest.is_file():
                    done_panel_stages.append(panel_stage)
                elif current is None:
                    current = panel_stage
            stage_key = current or "render"
            metrics_path = pipeline_run / "stages" / "detection" / "metrics.json"
            if metrics_path.is_file():
                try:
                    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                    processed_frames = int(payload.get("frames") or 0)
                    total_frames = total_frames or processed_frames
                except (json.JSONDecodeError, OSError, ValueError):
                    pass

    percent = stage_percent(stage_key)
    status = str(state.get("status", "RUNNING")).upper()
    if status == "COMPLETED":
        percent, stage_key = 100.0, "reports"

    log_path = Path(state.get("log_path") or (match_dir / "driver.log"))
    last_log = _tail_line(log_path) if log_path.is_file() else ""

    return {
        "percent": percent,
        "stage_key": stage_key,
        "stage_label": STAGE_LABELS.get(stage_key, stage_key),
        "processed_frames": processed_frames,
        "total_frames": total_frames,
        "status": status,
        "last_log_line": last_log,
    }


# ---------------------------------------------------------------------------
# Heartbeat writer thread.
# ---------------------------------------------------------------------------


class HeartbeatWriter(threading.Thread):
    def __init__(self, match_dir: Path, camera_id: str = "camera_1") -> None:
        super().__init__(daemon=True, name="panel-heartbeat")
        self.match_dir = Path(match_dir)
        self.camera_id = camera_id
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def write_once(self) -> dict[str, Any]:
        progress = compute_progress(self.match_dir)
        payload = {
            "timestamp": _utc_now(),
            "pid": os.getpid(),
            "current_stage": progress["stage_key"],
            "current_camera": self.camera_id,
            "current_chunk": 0,
            "processed_frames": progress["processed_frames"],
            "total_frames": progress["total_frames"],
            "last_log_line": progress["last_log_line"],
            "status": progress["status"],
        }
        run_dir = self.match_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(run_dir / HEARTBEAT_NAME, payload)
        return payload

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.write_once()
            except Exception:  # noqa: BLE001 - heartbeat must never kill the run
                pass
            self._stop.wait(HEARTBEAT_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Jersey inference on real tracklets (optional; skipped without a checkpoint).
# ---------------------------------------------------------------------------


def _jersey_inference(pipeline_run_dir: Path, match_dir: Path) -> Any:
    import pandas as pd

    checkpoint = Path(__file__).resolve().parents[3] / "artifacts" / "jersey" / "best.pt"
    if not checkpoint.is_file():
        return None
    import cv2

    tracks = pd.read_parquet(pipeline_run_dir / "tracks.parquet")
    tracks = tracks[tracks["object_type"] == "person"]
    video = pipeline_run_dir / "input" / "test_clip.mp4"
    if not video.is_file():
        candidates = list((pipeline_run_dir / "input").glob("*.mp4"))
        if not candidates:
            return None
        video = candidates[0]

    out_root = match_dir / "jersey_tracklets"
    lengths = tracks.groupby("track_id").size().sort_values(ascending=False)
    capture = cv2.VideoCapture(str(video))
    folders: list[Path] = []
    for track_id in list(lengths.head(10).index):
        rows = tracks[tracks["track_id"] == track_id].sort_values("frame_id")
        step = max(1, len(rows) // 8)
        selected = rows.iloc[::step].head(8)
        folder = out_root / f"track_{int(track_id):04d}"
        folder.mkdir(parents=True, exist_ok=True)
        written = 0
        for row in selected.itertuples(index=False):
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(row.frame_id))
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            x1, y1 = max(0, int(row.bbox_x1)), max(0, int(row.bbox_y1))
            x2, y2 = min(width, int(row.bbox_x2)), min(height, int(row.bbox_y2))
            if x2 - x1 < 8 or y2 - y1 < 16:
                continue
            cv2.imwrite(str(folder / f"f{int(row.frame_id):05d}.jpg"), frame[y1:y2, x1:x2])
            written += 1
        if written:
            folders.append(folder)
    capture.release()
    if not folders:
        return None

    from football_analytics.jersey.infer import record_from_directory, run_inference

    records = [record_from_directory(folder) for folder in folders]
    predictions = run_inference(
        checkpoint, records, output_path=match_dir / "jersey_predictions.json"
    )
    return pd.DataFrame(
        [
            {
                "track_id": int(prediction.tracklet_id.split("_")[1]),
                "jersey_number": (
                    None if prediction.jersey_number < 0 else int(prediction.jersey_number)
                ),
                "confidence": float(prediction.confidence),
                "status": (
                    "predicted"
                    if prediction.jersey_number >= 0 and prediction.confidence >= 0.6
                    else "unresolved"
                ),
            }
            for prediction in predictions
        ]
    )


# ---------------------------------------------------------------------------
# The single-button flow.
# ---------------------------------------------------------------------------


def run_analysis(
    video_path: Path,
    match_dir: Path,
    *,
    camera_id: str = "camera_1",
    chunk_seconds: float | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute the full user flow; heartbeat and state files track progress."""
    from football_analytics.full_match import (
        prepare_full_match,
        resume_full_match,
        run_full_match,
    )
    from football_analytics.full_match.postprocess import postprocess_pipeline_run
    from football_analytics.full_match.video_probe import probe_camera

    video_path = Path(video_path)
    match_dir = Path(match_dir)
    match_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[3]

    os.environ["FA_FULL_MATCH_CHUNK_ARTIFACTS"] = str(match_dir / "chunk_artifacts")
    os.environ["FA_FULL_MATCH_RUN_ID"] = match_dir.name
    os.environ.setdefault("FA_PYTHON", sys.executable)

    started = time.monotonic()
    write_state(
        match_dir,
        run_id=match_dir.name,
        status="RUNNING",
        phase="preparing",
        pid=os.getpid(),
        video=str(video_path),
        started_at=_utc_now(),
        error=None,
    )
    heartbeat = HeartbeatWriter(match_dir, camera_id)
    heartbeat.start()
    heartbeat.write_once()

    try:
        # 1-2. probe + validate
        write_state(match_dir, phase="validating")
        probe = probe_camera(video_path)
        if not probe.decodable:
            raise RuntimeError("video kareleri çözülemedi (ilk/orta/son kare kontrolü)")
        total_frames = probe.nb_frames or int(round(probe.duration_seconds * probe.avg_frame_rate))
        write_state(
            match_dir,
            total_frames=total_frames,
            duration_seconds=probe.duration_seconds,
            resolution=f"{probe.width}x{probe.height}",
        )

        # 3-4. manifest + prepare (idempotent on resume)
        prepared_dir = match_dir / "prepared"
        if not (prepared_dir / "match_manifest.json").is_file():
            prepare_full_match(
                inputs=[video_path],
                camera_ids=[camera_id],
                output_dir=prepared_dir,
                match_id=match_dir.name,
                chunk_seconds=chunk_seconds
                or max(30.0, min(300.0, probe.duration_seconds)),
                force=True,
            )

        # 5-7. real scheduler run (chunks + consolidation) via the adapter
        write_state(match_dir, phase="pipeline")
        run_dir = match_dir / "run"
        adapter_config = project_root / "configs" / "full_match" / "existing_pipeline_adapter.yaml"
        if resume and (run_dir / "run_state.json").is_file():
            report = resume_full_match(run_dir=run_dir, repair_manifests=True)
        else:
            report = run_full_match(
                prepared_dir=prepared_dir,
                run_dir=run_dir,
                chunk_pipeline_config=adapter_config,
            )
        if report.get("status") != "PASS":
            raise RuntimeError(f"analiz aşaması başarısız: {report.get('status')}")

        pipeline_run = _latest_pipeline_run_dir(match_dir)
        if pipeline_run is None:
            raise RuntimeError("pipeline çıktı klasörü bulunamadı")

        # 8. jersey inference on real tracklets (honest: may stay unresolved)
        write_state(match_dir, phase="jersey")
        try:
            jersey_df = _jersey_inference(pipeline_run, match_dir)
        except Exception as exc:  # noqa: BLE001 - jersey is optional, never fatal
            jersey_df = None
            write_state(match_dir, jersey_error=f"{type(exc).__name__}: {exc}"[:200])

        # 9-10. post-process + exports + validations
        write_state(match_dir, phase="reports")
        output_dir = match_dir / "output"
        postprocess_pipeline_run(
            pipeline_run, output_dir, camera_id=camera_id, jersey_predictions=jersey_df
        )

        elapsed = time.monotonic() - started
        state = write_state(
            match_dir,
            status="COMPLETED",
            phase="reports",
            finished_at=_utc_now(),
            elapsed_seconds=round(elapsed, 1),
            output_dir=str(output_dir),
        )
        heartbeat.write_once()
        return state
    except Exception as exc:  # noqa: BLE001 - converted into a friendly state
        progress = compute_progress(match_dir)
        state = write_state(
            match_dir,
            status="FAILED",
            failed_stage=progress["stage_label"],
            error=f"{type(exc).__name__}: {exc}"[:400],
            traceback=traceback.format_exc()[-4000:],
            finished_at=_utc_now(),
        )
        heartbeat.write_once()
        return state
    finally:
        heartbeat.stop()


def archive_previous_run(match_dir: Path) -> Path | None:
    """Move prior analysis artifacts aside (never delete) before a restart."""
    match_dir = Path(match_dir)
    targets = [
        item
        for item in ("prepared", "run", "output", "chunk_artifacts", STATE_NAME)
        if (match_dir / item).exists()
    ]
    if not targets:
        return None
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = match_dir / f"archive_{stamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for item in targets:
        shutil.move(str(match_dir / item), str(archive_dir / item))
    return archive_dir
