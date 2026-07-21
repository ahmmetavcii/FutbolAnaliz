#!/usr/bin/env python3
"""Streamlit operator panel for full-match processing.

This module deliberately imports only the Python standard library.  Streamlit
is loaded by :func:`main`, and pipeline/model code is loaded only by the
subprocesses started by an operator.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = Path("/mnt/c/football_data/uploads")
RESULTS_ROOT = Path("/mnt/c/football_data/results")
RUN_PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "run_pipeline.py"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pipeline" / "mvp1_tracking.yaml"
ALLOWED_CAMERA_COUNTS = frozenset({1, 2, 4})
SECTION_NAMES = (
    "New Match",
    "Video Validation",
    "Synchronization",
    "Calibration",
    "Analysis Settings",
    "Process Management",
    "Global Identities",
    "Roles",
    "Match Events",
    "Results",
)

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def safe_path_under(root: str | os.PathLike[str], candidate: str | os.PathLike[str]) -> Path:
    """Return a resolved candidate confined to *root*, or raise ``ValueError``.

    Relative candidates are interpreted below ``root``.  Existing symlinks are
    resolved, so they cannot be used to escape the allowed tree.
    """

    root_path = Path(root).expanduser().resolve()
    raw = Path(candidate).expanduser()
    resolved = (raw if raw.is_absolute() else root_path / raw).resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"path must be under {root_path}: {candidate}") from exc
    return resolved


def safe_upload_path(candidate: str | os.PathLike[str]) -> Path:
    """Resolve a path below the configured upload root."""

    return safe_path_under(UPLOAD_ROOT, candidate)


def safe_result_path(candidate: str | os.PathLike[str]) -> Path:
    """Resolve a path below the configured results root."""

    return safe_path_under(RESULTS_ROOT, candidate)


safe_results_path = safe_result_path


def _safe_identifier(value: str, field: str) -> str:
    text = str(value).strip()
    if not _SAFE_COMPONENT.fullmatch(text) or text in {".", ".."}:
        raise ValueError(
            f"{field} must contain only letters, numbers, '.', '_' or '-' "
            "and may not start with punctuation"
        )
    return text


def _normalise_cameras(
    cameras: Mapping[str, str | os.PathLike[str]]
    | Sequence[str | os.PathLike[str]],
) -> list[dict[str, str]]:
    if isinstance(cameras, (str, bytes, os.PathLike)):
        camera_items = [("camera_1", cameras)]
    elif isinstance(cameras, Mapping):
        camera_items = list(cameras.items())
    else:
        camera_items = [(f"camera_{index}", path) for index, path in enumerate(cameras, 1)]
    if len(camera_items) not in ALLOWED_CAMERA_COUNTS:
        raise ValueError("a full-match manifest requires exactly 1, 2, or 4 cameras")

    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for camera_id, video_path in camera_items:
        identifier = _safe_identifier(str(camera_id), "camera_id")
        if identifier in seen:
            raise ValueError(f"duplicate camera_id: {identifier}")
        seen.add(identifier)
        result.append(
            {
                "camera_id": identifier,
                "video_path": str(safe_upload_path(video_path)),
            }
        )
    return result


def build_manifest_payload(
    match_id: str,
    cameras: Mapping[str, str | os.PathLike[str]]
    | Sequence[str | os.PathLike[str]],
    *,
    synchronization: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic full-match manifest for 1, 2, or 4 cameras."""

    normalised = _normalise_cameras(cameras)
    offsets = dict(synchronization or {})
    unknown = set(offsets) - {camera["camera_id"] for camera in normalised}
    if unknown:
        raise ValueError(f"synchronization references unknown cameras: {sorted(unknown)}")
    for camera in normalised:
        camera["offset_seconds"] = float(offsets.get(camera["camera_id"], 0.0))
    return {
        "schema_version": "1.0.0",
        "match_id": _safe_identifier(match_id, "match_id"),
        "camera_count": len(normalised),
        "cameras": normalised,
        "metadata": dict(metadata or {}),
    }


def construct_manifest_payload(
    match_id: str,
    cameras: Mapping[str, str | os.PathLike[str]]
    | Sequence[str | os.PathLike[str]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility alias for :func:`build_manifest_payload`."""

    return build_manifest_payload(match_id, cameras, **kwargs)


def build_run_command(
    input_video: str | os.PathLike[str],
    run_directory: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str] = DEFAULT_CONFIG,
    python_executable: str | os.PathLike[str] = sys.executable,
    script_path: str | os.PathLike[str] = RUN_PIPELINE_SCRIPT,
    resume_run_dir: str | os.PathLike[str] | None = None,
    rerun_from: str | None = None,
) -> list[str]:
    """Construct, but never execute, a pipeline subprocess command."""

    command = [
        str(python_executable),
        str(Path(script_path)),
        "--config",
        str(Path(config_path)),
        "--input",
        str(safe_upload_path(input_video)),
        "--runs-root",
        str(safe_result_path(run_directory)),
    ]
    if resume_run_dir is not None:
        command.extend(["--resume-run-dir", str(safe_result_path(resume_run_dir))])
    if rerun_from:
        command.extend(["--rerun-from", str(rerun_from)])
    return command


construct_run_command = build_run_command


PREPARE_FULL_MATCH_SCRIPT = PROJECT_ROOT / "scripts" / "prepare_full_match.py"
RUN_FULL_MATCH_SCRIPT = PROJECT_ROOT / "scripts" / "run_full_match.py"
RESUME_FULL_MATCH_SCRIPT = PROJECT_ROOT / "scripts" / "resume_full_match.py"
DEFAULT_FULL_MATCH_CONFIG = PROJECT_ROOT / "configs" / "full_match" / "single_camera.yaml"
DEFAULT_ADAPTER_CONFIG = (
    PROJECT_ROOT / "configs" / "full_match" / "existing_pipeline_adapter.yaml"
)


def build_prepare_command(
    match_id: str,
    camera_videos: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str] = DEFAULT_FULL_MATCH_CONFIG,
    python_executable: str | os.PathLike[str] = sys.executable,
    force: bool = False,
) -> list[str]:
    """Construct the real scheduler's prepare command (never executes it)."""
    command = [str(python_executable), str(PREPARE_FULL_MATCH_SCRIPT)]
    for video in camera_videos:
        command.extend(["--input", str(safe_upload_path(video))])
    command.extend(
        [
            "--config", str(Path(config_path)),
            "--output-dir", str(safe_result_path(output_dir)),
            "--match-id", _safe_identifier(match_id, "match_id"),
        ]
    )
    if force:
        command.append("--force")
    return command


def build_full_match_run_command(
    prepared_dir: str | os.PathLike[str],
    run_dir: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str] = DEFAULT_FULL_MATCH_CONFIG,
    chunk_pipeline_config: str | os.PathLike[str] | None = DEFAULT_ADAPTER_CONFIG,
    python_executable: str | os.PathLike[str] = sys.executable,
) -> list[str]:
    """Construct the real scheduler's run command (never executes it)."""
    command = [
        str(python_executable),
        str(RUN_FULL_MATCH_SCRIPT),
        "--prepared-dir", str(safe_result_path(prepared_dir)),
        "--config", str(Path(config_path)),
        "--run-dir", str(safe_result_path(run_dir)),
    ]
    if chunk_pipeline_config is not None:
        command.extend(["--chunk-pipeline-config", str(Path(chunk_pipeline_config))])
    return command


def build_resume_command(
    run_dir: str | os.PathLike[str],
    *,
    rerun_from_stage: str | None = None,
    python_executable: str | os.PathLike[str] = sys.executable,
) -> list[str]:
    """Construct the real scheduler's resume command (never executes it)."""
    command = [
        str(python_executable),
        str(RESUME_FULL_MATCH_SCRIPT),
        "--run-dir", str(safe_result_path(run_dir)),
        "--repair-manifests",
    ]
    if rerun_from_stage:
        command.extend(["--rerun-from-stage", str(rerun_from_stage)])
    return command


def read_run_progress(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Read real scheduler state (chunk manifest + run state) for the UI.

    Returns chunk status counts, per-stage statuses, and completion percent
    from the JSON manifests the scheduler persists atomically.
    """
    base = safe_result_path(run_dir)
    result: dict[str, Any] = {"run_dir": str(base), "exists": base.is_dir()}
    if not base.is_dir():
        return result

    chunk_path = base / "chunk_manifest.json"
    if chunk_path.is_file():
        payload = json.loads(chunk_path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        counts: dict[str, int] = {}
        for record in records:
            counts[record.get("status", "UNKNOWN")] = (
                counts.get(record.get("status", "UNKNOWN"), 0) + 1
            )
        done = counts.get("PASS", 0) + counts.get("SKIPPED", 0)
        result["chunks"] = {
            "total": len(records),
            "counts": counts,
            "percent": round(100.0 * done / len(records), 2) if records else None,
            "records": [
                {
                    "camera_id": record.get("camera_id"),
                    "chunk_index": record.get("chunk_index"),
                    "status": record.get("status"),
                    "attempts": record.get("attempts"),
                    "wall_seconds": record.get("wall_seconds"),
                }
                for record in records
            ],
        }
    state_path = base / "run_state.json"
    if state_path.is_file():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        result["match_id"] = payload.get("match_id")
        result["stages"] = {
            stage.get("name"): stage.get("status") for stage in payload.get("stages", [])
        }
    return result


def build_identity_correction_payload(
    match_id: str,
    *,
    merges: Sequence[Mapping[str, Any] | Sequence[Any]] = (),
    splits: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the persisted manual global-identity correction payload."""

    merge_records: list[dict[str, str]] = []
    for item in merges:
        if isinstance(item, Mapping):
            source = item.get("source_identity", item.get("source"))
            target = item.get("target_identity", item.get("target"))
        else:
            if len(item) != 2:
                raise ValueError("each identity merge must contain source and target")
            source, target = item
        if source is None or target is None or str(source) == str(target):
            raise ValueError("identity merge requires distinct source and target identities")
        merge_records.append(
            {"source_identity": str(source), "target_identity": str(target)}
        )

    split_records: list[dict[str, Any]] = []
    for item in splits:
        identity = item.get("identity", item.get("source_identity"))
        track_ids = item.get("track_ids")
        new_identity = item.get("new_identity", item.get("target_identity"))
        if identity is None or new_identity is None or not track_ids:
            raise ValueError("identity split requires identity, non-empty track_ids and new_identity")
        split_records.append(
            {
                "identity": str(identity),
                "track_ids": [str(track_id) for track_id in track_ids],
                "new_identity": str(new_identity),
            }
        )
    return {
        "schema_version": "1.0.0",
        "match_id": _safe_identifier(match_id, "match_id"),
        "identity_corrections": {"merges": merge_records, "splits": split_records},
    }


def build_role_event_correction_payload(
    match_id: str,
    *,
    roles: Sequence[Mapping[str, Any]] = (),
    events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build manual role and match-event corrections without doing I/O."""

    role_records: list[dict[str, Any]] = []
    for role in roles:
        identity = role.get("identity", role.get("global_identity"))
        value = role.get("role")
        if identity is None or value is None:
            raise ValueError("role correction requires identity and role")
        record = dict(role)
        record["identity"] = str(identity)
        record["role"] = str(value)
        record.pop("global_identity", None)
        role_records.append(record)

    event_records: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("event_type", event.get("type"))
        timestamp = event.get("timestamp_seconds", event.get("timestamp"))
        if event_type is None or timestamp is None or float(timestamp) < 0:
            raise ValueError("event correction requires event_type and non-negative timestamp")
        record = dict(event)
        record["event_type"] = str(event_type)
        record["timestamp_seconds"] = float(timestamp)
        record.pop("type", None)
        record.pop("timestamp", None)
        event_records.append(record)
    return {
        "schema_version": "1.0.0",
        "match_id": _safe_identifier(match_id, "match_id"),
        "role_corrections": role_records,
        "event_corrections": event_records,
    }


def build_correction_payload(
    match_id: str,
    *,
    merges: Sequence[Mapping[str, Any] | Sequence[Any]] = (),
    splits: Sequence[Mapping[str, Any]] = (),
    roles: Sequence[Mapping[str, Any]] = (),
    events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build one combined correction document for an operator edit."""

    payload = build_identity_correction_payload(match_id, merges=merges, splits=splits)
    payload.update(
        {
            key: value
            for key, value in build_role_event_correction_payload(
                match_id, roles=roles, events=events
            ).items()
            if key not in {"schema_version", "match_id"}
        }
    )
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def persist_correction_payload(
    match_id: str,
    filename: str,
    payload: Mapping[str, Any],
) -> Path:
    """Atomically persist a correction below the match results directory."""

    match = _safe_identifier(match_id, "match_id")
    name = _safe_identifier(filename, "filename")
    if not name.endswith(".json"):
        raise ValueError("correction filename must end in .json")
    if payload.get("match_id") != match:
        raise ValueError("payload match_id does not match destination match")
    return _atomic_write_json(safe_result_path(Path(match) / "corrections" / name), payload)


def persist_identity_corrections(
    match_id: str,
    *,
    merges: Sequence[Mapping[str, Any] | Sequence[Any]] = (),
    splits: Sequence[Mapping[str, Any]] = (),
) -> Path:
    payload = build_identity_correction_payload(match_id, merges=merges, splits=splits)
    return persist_correction_payload(match_id, "identity_corrections.json", payload)


save_identity_corrections = persist_identity_corrections


def persist_role_event_corrections(
    match_id: str,
    *,
    roles: Sequence[Mapping[str, Any]] = (),
    events: Sequence[Mapping[str, Any]] = (),
) -> Path:
    payload = build_role_event_correction_payload(match_id, roles=roles, events=events)
    return persist_correction_payload(match_id, "role_event_corrections.json", payload)


save_role_event_corrections = persist_role_event_corrections


def _append_event_correction(
    output_dir: Path,
    event_id: str,
    kind: str,
    *,
    value: Any = None,
) -> Path:
    """Append one panel event correction for later recompute."""
    path = Path(output_dir) / "event_corrections.json"
    payload: dict[str, Any] = {"corrections": []}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {"corrections": []}
    corrections = list(payload.get("corrections") or [])
    corrections.append({"event_id": event_id, "kind": kind, "value": value})
    payload["corrections"] = corrections
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def discover_results(
    match_id: str | None = None,
    *,
    extensions: Sequence[str] | None = None,
) -> list[Path]:
    """Discover downloadable files below results, sorted by relative path."""

    base = safe_result_path(_safe_identifier(match_id, "match_id")) if match_id else RESULTS_ROOT
    if not base.exists():
        return []
    allowed = {suffix.lower() for suffix in extensions} if extensions else None
    files = (
        path
        for path in base.rglob("*")
        if path.is_file()
        and (allowed is None or path.suffix.lower() in allowed)
        and safe_path_under(RESULTS_ROOT, path) == path.resolve()
    )
    return sorted(files, key=lambda path: str(path.relative_to(base)))


def read_result_download(path: str | os.PathLike[str]) -> bytes:
    """Read a result after enforcing result-root confinement."""

    result = safe_result_path(path)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result.read_bytes()


get_result_download = read_result_download


def _load_streamlit() -> Any:
    try:
        return importlib.import_module("streamlit")
    except ModuleNotFoundError as exc:
        if exc.name != "streamlit":
            raise
        raise RuntimeError(
            "The full-match panel requires Streamlit. Install it with "
            "`python -m pip install streamlit`, then launch "
            "`streamlit run apps/full_match_panel.py`."
        ) from exc


def _json_editor(st: Any, label: str, default: str = "[]") -> list[dict[str, Any]]:
    raw = st.text_area(label, value=default)
    try:
        value = json.loads(raw)
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError("expected a JSON array of objects")
        return value
    except (json.JSONDecodeError, ValueError) as exc:
        st.warning(f"{label}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Simplified single-button panel (default UI).
# ---------------------------------------------------------------------------

SIMPLE_MODE_DEFAULT = True
TECHNICAL_DETAILS_HIDDEN_DEFAULT = True
PANEL_ANALYSIS_SCRIPT = PROJECT_ROOT / "scripts" / "run_panel_analysis.py"
AUTO_REFRESH_SECONDS = 2.5


def _driver() -> Any:
    """Import the analysis driver lazily so panel import stays lightweight."""
    src = str(PROJECT_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module("football_analytics.full_match.panel_driver")


def derive_match_name(filename: str) -> str:
    """Auto-generate a filesystem-safe match name from an uploaded filename."""
    stem = Path(str(filename)).stem or "match"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-._") or "match"
    return cleaned[:60]


def format_clock(seconds: float | None) -> str:
    """Render seconds as MM:SS (or HH:MM:SS above one hour)."""
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def estimate_remaining_seconds(percent: float, elapsed_seconds: float) -> float | None:
    """ETA derived from real progress percent, never from wall-clock alone."""
    if percent <= 0 or elapsed_seconds <= 0:
        return None
    if percent >= 100:
        return 0.0
    return elapsed_seconds * (100.0 - percent) / percent


def probe_video_info(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read duration/resolution/size for display via ffprobe (real values)."""
    video = Path(path)
    info: dict[str, Any] = {
        "name": video.name,
        "size_bytes": video.stat().st_size if video.is_file() else None,
    }
    try:
        completed = subprocess.run(  # noqa: S603
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height:format=duration",
                "-of", "json", str(video),
            ],
            capture_output=True, text=True, timeout=30,
        )
        payload = json.loads(completed.stdout or "{}")
        streams = payload.get("streams") or [{}]
        info["width"] = streams[0].get("width")
        info["height"] = streams[0].get("height")
        duration = (payload.get("format") or {}).get("duration")
        info["duration_seconds"] = float(duration) if duration else None
    except (OSError, ValueError, subprocess.SubprocessError):
        info.setdefault("duration_seconds", None)
    return info


def build_panel_analysis_command(
    video_path: str | os.PathLike[str],
    match_dir: str | os.PathLike[str],
    *,
    resume: bool = False,
    chunk_seconds: float | None = None,
    python_executable: str | os.PathLike[str] = sys.executable,
) -> list[str]:
    """Construct the single-flow driver command (never executes it)."""
    command = [
        str(python_executable),
        str(PANEL_ANALYSIS_SCRIPT),
        "--video", str(safe_upload_path(video_path)),
        "--match-dir", str(safe_result_path(match_dir)),
    ]
    if chunk_seconds:
        command.extend(["--chunk-seconds", str(float(chunk_seconds))])
    if resume:
        command.append("--resume")
    return command


def start_panel_analysis(
    video_path: str | os.PathLike[str],
    match_id: str,
    *,
    resume: bool = False,
    chunk_seconds: float | None = None,
) -> dict[str, Any]:
    """Spawn the detached analysis driver once; refuse duplicates honestly."""
    driver = _driver()
    driver.reap_children()
    allowed, active = driver.can_start_new_run(RESULTS_ROOT)
    if not allowed and not resume:
        raise RuntimeError(
            f"Devam eden bir analiz var ({active.get('match_id')}); "
            "bitmeden yeni analiz başlatılamaz."
        )
    match = _safe_identifier(match_id, "match_id")
    match_dir = safe_result_path(match)
    match_dir.mkdir(parents=True, exist_ok=True)
    command = build_panel_analysis_command(
        video_path, match_dir, resume=resume, chunk_seconds=chunk_seconds
    )
    log_path = match_dir / "driver.log"
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pointer = {
        "match_id": match,
        "match_dir": str(match_dir),
        "pid": process.pid,
        "video": str(video_path),
        "resume": resume,
    }
    driver.save_active_pointer(RESULTS_ROOT, pointer)
    return pointer


def load_current_analysis() -> dict[str, Any]:
    """Recover the current analysis after refresh or page reopen."""
    driver = _driver()
    driver.reap_children()
    pointer = driver.load_active_pointer(RESULTS_ROOT)
    if not pointer:
        return {}
    match_dir = Path(pointer.get("match_dir", ""))
    state = driver.read_state(match_dir)
    heartbeat = driver.read_heartbeat(match_dir)
    age = driver.heartbeat_age_seconds(heartbeat)
    process_state = driver.pid_status(pointer.get("pid"))
    status = str(state.get("status", "")).upper()
    # A dead/zombie process must never be reported as RUNNING.
    if status == "RUNNING" and process_state != "running":
        status = "FAILED"
        state = dict(state)
        state["status"] = "FAILED"
        state.setdefault("error", "Analiz süreci beklenmedik şekilde sonlandı.")
    return {
        "pointer": pointer,
        "match_dir": match_dir,
        "state": state,
        "status": status,
        "heartbeat": heartbeat,
        "heartbeat_age": age,
        "process_state": process_state,
        "health_label": driver.classify_heartbeat_age(age),
        "offer_resume": driver.should_offer_resume(state, age),
    }


def build_failure_view(state: Mapping[str, Any], last_log_line: str = "") -> dict[str, str]:
    """Friendly failure summary; raw traceback is excluded on purpose."""
    stages = state.get("stages") or {}
    last_pass = ""
    if isinstance(stages, Mapping):
        passed = [name for name, status in stages.items() if status == "PASS"]
        last_pass = passed[-1] if passed else ""
    return {
        "title": "Analiz tamamlanamadı",
        "failed_stage": str(state.get("failed_stage") or state.get("phase") or "bilinmiyor"),
        "last_successful_stage": last_pass or "bilinmiyor",
        "error": str(state.get("error") or "Bilinmeyen hata"),
        "can_resume": "Evet" if (state.get("run_id")) else "Hayır",
        "last_log_line": last_log_line or "",
    }


_NOT_AVAILABLE = "Mevcut değil"


def _display_value(value: Any, suffix: str = "") -> str:
    """Show real values only; missing data reads 'Mevcut değil', never 0."""
    if value is None or value == "" or (isinstance(value, float) and value != value):
        return _NOT_AVAILABLE
    if isinstance(value, float):
        value = round(value, 2)
    return f"{value}{suffix}"


def load_results_summary(output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Read completion summary cards from real quality/run reports."""
    base = Path(output_dir)
    summary: dict[str, Any] = {}
    quality_path = base / "quality_report.json"
    if quality_path.is_file():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        summary["detections"] = quality.get("detections")
        summary["tracked_players"] = quality.get("global_players") or quality.get("tracks")
        summary["calibration_valid_ratio"] = quality.get("calibration_valid_ratio")
        summary["jersey_resolved"] = quality.get("jersey_resolved")
    run_report_path = base / "run_report.json"
    if run_report_path.is_file():
        report = json.loads(run_report_path.read_text(encoding="utf-8"))
        summary["duration_seconds"] = report.get("video_duration_seconds")
        summary["processing_seconds"] = report.get("total_seconds")
    return summary


def load_player_table(output_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Player rows for the results table with honest 'Mevcut değil' fallbacks."""
    import csv

    path = Path(output_dir) / "player_summary.csv"
    if not path.is_file():
        return []
    columns = {
        "global_id": ("global_id", "identity", "track_id"),
        "Takım": ("team_id", "team"),
        "Forma": ("jersey_number",),
        "Rol": ("role",),
        "Görünür süre (sn)": ("visible_seconds", "visible_time_seconds"),
        "Toplam mesafe (m)": ("total_distance_m", "distance_m"),
        "Ortalama hız (km/s)": ("avg_speed_kmh",),
        "Maksimum hız (km/s)": ("max_speed_kmh", "top_speed_kmh"),
        "Gol": ("goals",),
        "Asist": ("assists",),
        "Kalite": ("quality", "quality_score"),
    }
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for label, keys in columns.items():
                value = next(
                    (record[key] for key in keys if record.get(key) not in (None, "")),
                    None,
                )
                row[label] = _display_value(value)
            rows.append(row)
    return rows


def _render_upload_view(st: Any, state: Any) -> None:
    st.markdown("#### 1. Maç videosunu yükle")
    upload = st.file_uploader(
        "Videoyu buraya sürükle veya seç (MP4, MOV, MKV)",
        type=["mp4", "mov", "mkv"],
        key="simple_upload",
    )
    if upload is None:
        st.info("Analiz için bir maç videosu seçin.")
        return

    match_name_default = derive_match_name(upload.name)
    match_name = st.text_input("Maç adı", value=state.get("match_name", match_name_default))
    state["match_name"] = derive_match_name(match_name or match_name_default)

    # Save the upload exactly once per (name, size); refreshes must not rewrite.
    upload_key = f"{upload.name}:{getattr(upload, 'size', len(upload.getbuffer()))}"
    if state.get("saved_upload_key") != upload_key:
        destination = safe_upload_path(Path(state["match_name"]) / f"camera_1_{upload.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(upload.getbuffer())
        state["saved_upload_key"] = upload_key
        state["saved_upload_path"] = str(destination)

    info = probe_video_info(state["saved_upload_path"])
    columns = st.columns(4)
    columns[0].metric("Dosya", info.get("name") or _NOT_AVAILABLE)
    columns[1].metric("Süre", format_clock(info.get("duration_seconds")))
    resolution = (
        f"{info['width']}x{info['height']}" if info.get("width") else _NOT_AVAILABLE
    )
    columns[2].metric("Çözünürlük", resolution)
    size_mb = info.get("size_bytes")
    columns[3].metric(
        "Boyut", f"{size_mb / 1e6:.1f} MB" if size_mb else _NOT_AVAILABLE
    )

    st.markdown("---")
    if st.button("ANALİZİ BAŞLAT", type="primary", use_container_width=True):
        try:
            pointer = start_panel_analysis(
                state["saved_upload_path"],
                state["match_name"],
                chunk_seconds=state.get("advanced_chunk_seconds"),
            )
            state["active_match_id"] = pointer["match_id"]
            st.rerun()
        except (RuntimeError, OSError, ValueError) as exc:
            st.error(str(exc))


def _render_technical_details(st: Any, current: Mapping[str, Any]) -> None:
    with st.expander("Teknik ayrıntıları göster", expanded=False):
        driver = _driver()
        match_dir = Path(current["match_dir"])
        state = current.get("state", {})
        st.write(f"PID: {current.get('pointer', {}).get('pid')} ({current.get('process_state')})")
        st.write(f"Çalışma klasörü: {match_dir}")
        log_path = match_dir / "driver.log"
        if log_path.is_file():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            st.code("\n".join(lines[-20:]) or "(log boş)")
        run_dir = match_dir / "run"
        if (run_dir / "chunk_manifest.json").is_file():
            try:
                progress = read_run_progress(run_dir)
                st.json(progress.get("chunks", {}).get("counts", {}))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        try:
            import psutil

            memory = psutil.virtual_memory()
            st.write(f"RAM: {memory.used / 1e9:.1f} / {memory.total / 1e9:.1f} GB")
        except Exception:  # noqa: BLE001
            pass
        try:
            gpu = subprocess.run(  # noqa: S603
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True, text=True, timeout=5,
            )
            if gpu.returncode == 0 and gpu.stdout.strip():
                st.write(f"VRAM: {gpu.stdout.strip()}")
        except (OSError, subprocess.SubprocessError):
            pass
        if state.get("traceback"):
            st.code(str(state["traceback"]))
        _ = driver


def _render_running_view(st: Any, current: Mapping[str, Any]) -> None:
    driver = _driver()
    match_dir = Path(current["match_dir"])
    state = current.get("state", {})
    progress = driver.compute_progress(match_dir)
    percent = float(progress.get("percent") or 0.0)
    heartbeat = current.get("heartbeat", {})
    age = current.get("heartbeat_age")
    health = current.get("health_label", "Bilgi yok")

    st.markdown(f"### Analiz ediliyor: %{percent:.0f}")
    st.progress(min(1.0, percent / 100.0))
    st.markdown(f"**Şu an:** {progress.get('stage_label')}")

    processed = heartbeat.get("processed_frames") or progress.get("processed_frames")
    total = heartbeat.get("total_frames") or progress.get("total_frames")
    if processed and total:
        st.markdown(f"**İşlenen kare:** {int(processed)} / {int(total)}")

    elapsed = None
    started = state.get("started_at")
    if started:
        try:
            import datetime as _dt

            begin = _dt.datetime.fromisoformat(started)
            elapsed = (_dt.datetime.now(_dt.timezone.utc) - begin).total_seconds()
        except ValueError:
            elapsed = None
    remaining = estimate_remaining_seconds(percent, elapsed or 0.0)
    st.markdown(f"**Geçen süre:** {format_clock(elapsed)}")
    st.markdown(f"**Tahmini kalan:** {format_clock(remaining)}")
    updated = f"{int(age)} sn önce güncellendi" if age is not None else "bilgi yok"
    st.markdown(f"**Durum:** {health} · {updated}")

    stalled = current.get("offer_resume", False)
    if stalled:
        st.warning(
            "İşlem yanıt vermiyor olabilir. "
            f"Aşama: {progress.get('stage_label')} · "
            f"Son log: {heartbeat.get('last_log_line') or progress.get('last_log_line') or '-'}"
        )
        columns = st.columns(3)
        if columns[0].button("Yeniden Kontrol Et"):
            st.rerun()
        if columns[1].button("Kaldığı Yerden Devam Et"):
            pointer = current.get("pointer", {})
            try:
                start_panel_analysis(pointer.get("video", ""), pointer.get("match_id", ""), resume=True)
                st.rerun()
            except (RuntimeError, OSError, ValueError) as exc:
                st.error(str(exc))
        if columns[2].button("Analizi Durdur"):
            pid = current.get("pointer", {}).get("pid")
            if pid and driver.pid_status(pid) == "running":
                try:
                    os.kill(int(pid), 15)
                except OSError:
                    pass
            driver.write_state(match_dir, status="FAILED", error="Kullanıcı analizi durdurdu.")
            st.rerun()

    _render_technical_details(st, current)


def _render_completed_view(st: Any, current: Mapping[str, Any]) -> None:
    from football_analytics.panel.opta_labels import (
        DEFENSIVE_COLUMN_TR,
        DRIBBLE_COLUMN_TR,
        DUEL_COLUMN_TR,
        PASS_COLUMN_TR,
        PHYSICAL_COLUMN_TR,
        PLAYER_COLUMN_TR,
        TAB_NAMES,
        TEAM_COLUMN_TR,
        format_value,
        load_csv_or_empty,
        load_parquet_or_empty,
        quality_label,
        rename_frame,
        status_counts,
    )

    match_dir = Path(current["match_dir"])
    state = current.get("state", {})
    output_dir = Path(state.get("output_dir") or (match_dir / "output"))
    st.success("ANALİZ TAMAMLANDI")
    st.caption("Opta-benzeri otomatik video analizi (resmî Opta verisi değildir).")

    summary = load_results_summary(output_dir)
    publish_meta: dict[str, Any] = {}
    publish_path = output_dir / "opta_stats_publishable.json"
    if publish_path.is_file():
        publish_meta = json.loads(publish_path.read_text(encoding="utf-8"))
    stats_ok = bool(publish_meta.get("stats_publishable", False))
    overall_ok = bool(publish_meta.get("overall_publishable", False))
    if publish_path.is_file() and publish_meta.get("warning"):
        st.warning(str(publish_meta["warning"]))
    elif publish_path.is_file() and (
        publish_meta.get("gt_incomplete", {}).get("ball")
        or publish_meta.get("gt_incomplete", {}).get("identity")
    ):
        st.warning(
            "Model coverage yüksek ancak doğruluk ground truth ile doğrulanmadı."
        )
    if publish_path.is_file() and not stats_ok:
        st.warning(
            "Kimlik/kalite kapısı: Opta-benzeri oyuncu/takım istatistikleri gizlendi. "
            f"Neden: {publish_meta.get('reason') or 'validated player count güvenilir değil'}"
        )
    tabs = st.tabs(TAB_NAMES)

    with tabs[0]:
        cards = st.columns(6)
        cards[0].metric("Video süresi", format_clock(state.get("duration_seconds")))
        cards[1].metric("Tespit edilen", _display_value(summary.get("detections")))
        cards[2].metric("Takip edilen", _display_value(summary.get("tracked_players")))
        ratio = summary.get("calibration_valid_ratio")
        cards[3].metric(
            "Kalibrasyon",
            f"%{100 * float(ratio):.0f}" if ratio is not None else _NOT_AVAILABLE,
        )
        cards[4].metric("Forma no bulunan", _display_value(summary.get("jersey_resolved")))
        cards[5].metric("İşlem süresi", format_clock(state.get("elapsed_seconds")))
        annotated = output_dir / "annotated_match.mp4"
        if not annotated.is_file():
            annotated = output_dir / "analytics_annotated.mp4"
        tactical = output_dir / "tactical_map.mp4"
        if not tactical.is_file():
            tactical = output_dir / "tactical_preview.mp4"
        st.markdown("#### Analiz Videosu")
        if annotated.is_file():
            st.video(str(annotated))
        else:
            st.info(_NOT_AVAILABLE)
        st.markdown("#### Taktik Harita")
        if tactical.is_file():
            st.video(str(tactical))
        else:
            st.info(_NOT_AVAILABLE)

    with tabs[1]:
        if not stats_ok:
            st.info("İstatistikler yayınlanmadı (kimlik kalitesi kapısı).")
        else:
            players = load_csv_or_empty(output_dir / "player_opta_summary.csv")
            if players.empty:
                players = load_csv_or_empty(output_dir / "player_summary.csv")
            if not players.empty:
                st.dataframe(rename_frame(players, PLAYER_COLUMN_TR), use_container_width=True)
            else:
                legacy = load_player_table(output_dir)
                if legacy:
                    st.dataframe(legacy, use_container_width=True)
                else:
                    st.info(_NOT_AVAILABLE)

    with tabs[2]:
        if not stats_ok:
            st.info("İstatistikler yayınlanmadı (kimlik kalitesi kapısı).")
        else:
            teams = load_csv_or_empty(output_dir / "team_opta_summary.csv")
            if teams.empty:
                teams = load_csv_or_empty(output_dir / "team_summary.csv")
            if not teams.empty:
                st.dataframe(rename_frame(teams, TEAM_COLUMN_TR), use_container_width=True)
            else:
                st.info(_NOT_AVAILABLE)

    with tabs[3]:
        passes = load_parquet_or_empty(output_dir / "pass_events.parquet")
        counts = status_counts(passes)
        st.write(
            f"Onaylı: {counts['confirmed_count']} · Aday: {counts['candidate_count']} · "
            f"Çözümsüz: {counts['unresolved_count']}"
        )
        if not passes.empty:
            shown = passes[passes["status"].eq("confirmed")] if "status" in passes.columns else passes
            st.dataframe(rename_frame(shown, PASS_COLUMN_TR), use_container_width=True)
        else:
            st.info(_NOT_AVAILABLE)

    with tabs[4]:
        duels = load_parquet_or_empty(output_dir / "duel_events.parquet")
        st.write(status_counts(duels))
        if not duels.empty:
            st.dataframe(rename_frame(duels, DUEL_COLUMN_TR), use_container_width=True)
        else:
            st.info(_NOT_AVAILABLE)

    with tabs[5]:
        dribbles = load_parquet_or_empty(output_dir / "dribble_events.parquet")
        if not dribbles.empty:
            st.dataframe(rename_frame(dribbles, DRIBBLE_COLUMN_TR), use_container_width=True)
        else:
            st.info(_NOT_AVAILABLE)

    with tabs[6]:
        defensive = load_parquet_or_empty(output_dir / "defensive_actions.parquet")
        if not defensive.empty:
            st.dataframe(rename_frame(defensive, DEFENSIVE_COLUMN_TR), use_container_width=True)
        else:
            st.info(_NOT_AVAILABLE)

    with tabs[7]:
        metrics = load_parquet_or_empty(output_dir / "player_metrics.parquet")
        if not metrics.empty:
            cols = [c for c in PHYSICAL_COLUMN_TR if c in metrics.columns]
            st.dataframe(
                rename_frame(metrics[cols].head(500), PHYSICAL_COLUMN_TR),
                use_container_width=True,
            )
        else:
            st.info(_NOT_AVAILABLE)

    with tabs[8]:
        heat_dir = output_dir / "heatmaps"
        if heat_dir.is_dir():
            images = sorted(heat_dir.glob("*.png"))[:24]
            if images:
                for img in images:
                    st.image(str(img), caption=img.name)
            else:
                st.info(_NOT_AVAILABLE)
        else:
            st.info(_NOT_AVAILABLE)

    with tabs[9]:
        events_path = output_dir / "match_events.parquet"
        if not events_path.is_file():
            events_path = output_dir / "events.parquet"
        quality_path = output_dir / "quality_report.json"
        events_note = _NOT_AVAILABLE
        if quality_path.is_file():
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            events_note = (
                f"Onaylanmış: {int(quality.get('confirmed_events') or 0)} · "
                f"Aday: {int(quality.get('candidate_events') or 0)}"
                f" · {quality.get('events_reason', '')}"
            )
        st.markdown("#### MAÇ OLAYLARI")
        st.write(events_note)
        if events_path.is_file():
            try:
                import pandas as _pd

                events_frame = _pd.read_parquet(events_path)
            except Exception:  # noqa: BLE001
                events_frame = None
            if events_frame is not None and not events_frame.empty:
                st.dataframe(events_frame, use_container_width=True)
                clips_dir = output_dir / "event_clips"
                for row in events_frame.to_dict("records"):
                    event_id = str(row.get("event_id", ""))
                    status = str(row.get("status", "")).lower()
                    clip = clips_dir / f"{event_id}.mp4"
                    with st.expander(f"{event_id} · {row.get('event_type')} · {status}"):
                        if clip.is_file():
                            st.video(str(clip))
                        else:
                            st.caption("Klip yok")
                        if status == "candidate_review_required":
                            cols = st.columns(4)
                            if cols[0].button("Onayla", key=f"ev_ok_{event_id}"):
                                _append_event_correction(output_dir, event_id, "confirm")
                                st.rerun()
                            if cols[1].button("Reddet", key=f"ev_no_{event_id}"):
                                _append_event_correction(output_dir, event_id, "reject")
                                st.rerun()
                            scorer = cols[2].text_input("Golcü track id", key=f"ev_sc_{event_id}")
                            if cols[2].button("Golcüyü değiştir", key=f"ev_sc_btn_{event_id}") and scorer:
                                _append_event_correction(
                                    output_dir, event_id, "set_scorer", value=int(scorer)
                                )
                                st.rerun()
                            assist = cols[3].text_input("Asist track id", key=f"ev_as_{event_id}")
                            a1, a2 = st.columns(2)
                            if a1.button("Asisti değiştir", key=f"ev_as_btn_{event_id}") and assist:
                                _append_event_correction(
                                    output_dir, event_id, "set_assist", value=int(assist)
                                )
                                st.rerun()
                            if a2.button("Asisti kaldır", key=f"ev_as_clear_{event_id}"):
                                _append_event_correction(
                                    output_dir, event_id, "set_assist", value=None
                                )
                                st.rerun()
                if st.button("Olay düzeltmelerini uygula (recompute)"):
                    try:
                        import subprocess as _sp

                        _sp.run(
                            [
                                sys.executable,
                                str(PROJECT_ROOT / "scripts" / "recompute_match_events.py"),
                                "--run-dir",
                                str(output_dir),
                            ],
                            check=False,
                            cwd=str(PROJECT_ROOT),
                        )
                        st.success("Recompute tamamlandı")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))
            else:
                st.info("Bu klipte desteklenen olay yok (dürüst boş çıktı).")
        else:
            st.info(_NOT_AVAILABLE)

    with tabs[10]:
        pub_flags_path = output_dir / "publishability_flags.json"
        flags = {}
        if pub_flags_path.is_file():
            flags = json.loads(pub_flags_path.read_text(encoding="utf-8"))
        elif publish_meta:
            flags = publish_meta
        phys = {}
        phys_path = output_dir / "physical_metrics_quality.json"
        if phys_path.is_file():
            phys = json.loads(phys_path.read_text(encoding="utf-8"))

        def _yn(v: Any) -> str:
            return "Evet" if v else "Hayır"

        st.write(
            {
                "Top modeli değerlendirildi mi?": _yn(
                    not flags.get("gt_incomplete", {}).get("ball", True)
                    and flags.get("ball_detection_publishable")
                ),
                "Oyuncu kimliği değerlendirildi mi?": _yn(
                    not flags.get("gt_incomplete", {}).get("identity", True)
                    and flags.get("identity_publishable")
                ),
                "Touch doğrulandı mı?": _yn(
                    not flags.get("gt_incomplete", {}).get("touch", True)
                    and flags.get("touch_publishable")
                ),
                "Kalibrasyon coverage": (
                    f"%{100 * float(phys['calibration_coverage']):.1f}"
                    if phys.get("calibration_coverage") is not None
                    else _NOT_AVAILABLE
                ),
                "Fiziksel metrikler yayınlanabilir mi?": _yn(
                    flags.get("physical_metrics_publishable")
                    or publish_meta.get("physical_metrics_publishable")
                ),
                "Genel istatistikler yayınlanabilir mi?": _yn(
                    flags.get("overall_publishable")
                    or publish_meta.get("overall_publishable")
                    or overall_ok
                ),
            }
        )
        if flags.get("gt_incomplete", {}).get("ball") or publish_meta.get("gt_incomplete", {}).get(
            "ball"
        ):
            st.info(
                "Model coverage yüksek ancak doğruluk ground truth ile doğrulanmadı."
            )
        if publish_path.is_file():
            st.caption("Ham kimlik sayıları (hard-cap yok; coverage ≠ doğruluk)")
            st.write(
                {
                    "action_stats_publishable": _yn(publish_meta.get("action_stats_publishable")),
                    "identity_publishable": _yn(publish_meta.get("identity_publishable")),
                    "Bayraklar": publish_meta.get("identity_flags") or [],
                    "Doğrulanmış/takım": publish_meta.get("validated_by_team") or {},
                    "Ham sayı/takım": publish_meta.get("raw_count_by_team") or {},
                }
            )
        players = load_csv_or_empty(output_dir / "player_opta_summary.csv")
        if not players.empty and "quality_flags" in players.columns and stats_ok:
            rows = []
            for row in players.to_dict("records"):
                rows.append(
                    {
                        "Oyuncu": row.get("global_player_id"),
                        "Kalite": quality_label(
                            row.get("quality_flags"),
                            float(row["metric_quality"])
                            if row.get("metric_quality") is not None
                            and str(row.get("metric_quality")) != "nan"
                            else None,
                        ),
                        "Bayraklar": format_value(row.get("quality_flags")),
                        "Aktivasyon": format_value(row.get("activity_index")),
                    }
                )
            st.dataframe(rows, use_container_width=True)
        elif not stats_ok:
            st.info("Düşük kimlik kalitesi — Opta istatistikleri gizlendi.")
        else:
            st.info(_NOT_AVAILABLE)
        identity = output_dir / "global_player_identity.parquet"
        if not identity.is_file():
            identity = output_dir / "global_identity_report.parquet"
        if identity.is_file():
            idf = load_parquet_or_empty(identity)
            st.caption(
                f"Küresel kimlik satırı: {len(idf)} "
                "(ham track sayısı oyuncu sayısı değildir)."
            )
        cov = output_dir / "ball_coverage_report.json"
        if cov.is_file():
            st.markdown("#### Top kapsama (aday / trajectory — recall değil)")
            payload = json.loads(cov.read_text(encoding="utf-8"))
            st.write(
                {
                    "candidate_coverage": payload.get("raw_detection_coverage"),
                    "trajectory_coverage": payload.get("tracked_coverage"),
                    "wrong_object_switches": payload.get("wrong_object_switches"),
                }
            )
        touch_dbg = output_dir / "touch_debug"
        if touch_dbg.is_dir():
            imgs = sorted(touch_dbg.glob("*.jpg"))[:12]
            if imgs:
                st.markdown("#### Temas inceleme kareleri")
                for img in imgs:
                    st.image(str(img), caption=img.name)

    st.markdown("#### İndirmeler")
    downloads = [
        ("Excel raporu", output_dir / "full_match_report.xlsx"),
        ("Oyuncu özeti (Opta-benzeri)", output_dir / "player_opta_summary.csv"),
        ("Takım özeti (Opta-benzeri)", output_dir / "team_opta_summary.csv"),
        ("Oyuncu istatistikleri (CSV)", output_dir / "player_summary.csv"),
        ("Takım istatistikleri (CSV)", output_dir / "team_summary.csv"),
        ("Kalite raporu (JSON)", output_dir / "quality_report.json"),
        ("Çalışma raporu (JSON)", output_dir / "run_report.json"),
    ]
    columns = st.columns(min(4, len(downloads)))
    for idx, (label, path) in enumerate(downloads):
        column = columns[idx % len(columns)]
        if path.is_file():
            column.download_button(label, data=path.read_bytes(), file_name=path.name)
        else:
            column.caption(f"{label}: {_NOT_AVAILABLE}")
    st.caption(f"Sonuç klasörü: {output_dir}")

    if st.button("Yeni analiz başlat"):
        _driver().save_active_pointer(RESULTS_ROOT, {})
        for key in ("saved_upload_key", "saved_upload_path", "active_match_id"):
            st.session_state.pop(key, None)
        st.rerun()


def _render_failed_view(st: Any, current: Mapping[str, Any]) -> None:
    driver = _driver()
    match_dir = Path(current["match_dir"])
    state = current.get("state", {})
    heartbeat = current.get("heartbeat", {})
    view = build_failure_view(state, heartbeat.get("last_log_line", ""))

    st.error(view["title"])
    st.markdown(f"**Durduğu aşama:** {view['failed_stage']}")
    st.markdown(f"**Son başarılı aşama:** {view['last_successful_stage']}")
    st.markdown(f"**Hata:** {view['error']}")
    st.markdown(f"**Kaldığı yerden devam:** {view['can_resume']}")
    if view["last_log_line"]:
        st.markdown(f"**Son log satırı:** `{view['last_log_line']}`")

    columns = st.columns(2)
    pointer = current.get("pointer", {})
    if columns[0].button("Kaldığı Yerden Devam Et"):
        try:
            start_panel_analysis(pointer.get("video", ""), pointer.get("match_id", ""), resume=True)
            st.rerun()
        except (RuntimeError, OSError, ValueError) as exc:
            st.error(str(exc))
    if columns[1].button("Baştan Yeniden Başlat"):
        try:
            driver.archive_previous_run(match_dir)
            start_panel_analysis(pointer.get("video", ""), pointer.get("match_id", ""))
            st.rerun()
        except (RuntimeError, OSError, ValueError) as exc:
            st.error(str(exc))
    _render_technical_details(st, current)


def _render_advanced_settings(st: Any, state: Any) -> None:
    with st.expander("Gelişmiş Ayarlar", expanded=False):
        st.caption("Normal kullanım için bu ayarlara dokunmanız gerekmez.")
        state["advanced_chunk_seconds"] = st.number_input(
            "Chunk süresi (saniye)", min_value=30.0, max_value=300.0, value=120.0
        )
        st.text_input("Kamera sayısı", value="1", disabled=True)
        st.text_input("Kamera rolü", value="broadcast", disabled=True)
        st.text_input("Config yolu", value=str(DEFAULT_FULL_MATCH_CONFIG), disabled=True)
        st.text_input("Adapter yolu", value=str(DEFAULT_ADAPTER_CONFIG), disabled=True)
        st.text_input("Rerun from stage", value="", key="advanced_rerun_from")
        st.caption(
            "Kalibrasyon, senkronizasyon, debug ve artifact yönetimi panel "
            "tarafından otomatik yürütülür."
        )


def _render_panel(st: Any) -> None:
    st.set_page_config(page_title="Football Match Analysis", layout="centered")
    st.title("Football Match Analysis")
    state = st.session_state
    state.setdefault("simple_mode", SIMPLE_MODE_DEFAULT)

    current = load_current_analysis()
    status = current.get("status", "")

    if status == "RUNNING":
        _render_running_view(st, current)
        import time as _time

        _time.sleep(AUTO_REFRESH_SECONDS)
        st.rerun()
    elif status == "COMPLETED":
        _render_completed_view(st, current)
    elif status == "FAILED":
        _render_failed_view(st, current)
    else:
        _render_upload_view(st, state)

    _render_advanced_settings(st, state)


def _render_legacy_panel(st: Any) -> None:
    st.set_page_config(page_title="Full Match Panel", layout="wide")
    st.title("Football Analytics — Full Match Panel")
    state = st.session_state

    with st.expander("New Match", expanded=True):
        match_id = st.text_input("Match ID", value=state.get("match_id", "match-001"))
        camera_count = st.selectbox("Camera count", (1, 2, 4))
        uploads = [
            st.file_uploader(f"Camera {index}", type=["mp4", "mov", "mkv"])
            for index in range(1, camera_count + 1)
        ]
        if st.button("Save match manifest"):
            try:
                match = _safe_identifier(match_id, "match_id")
                camera_paths = []
                for index, upload in enumerate(uploads, 1):
                    if upload is None:
                        raise ValueError("select a video for every camera")
                    destination = safe_upload_path(Path(match) / f"camera_{index}_{upload.name}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(upload.getbuffer())
                    camera_paths.append(destination)
                manifest = build_manifest_payload(match, camera_paths)
                manifest_path = safe_upload_path(Path(match) / "manifest.json")
                _atomic_write_json(manifest_path, manifest)
                state.update(match_id=match, manifest=manifest, manifest_path=str(manifest_path))
                st.success(f"Saved {manifest_path}")
            except (OSError, ValueError) as exc:
                st.error(str(exc))

    with st.expander("Video Validation"):
        st.write("Review selected camera files, sizes, and container formats before processing.")
        for camera in state.get("manifest", {}).get("cameras", []):
            path = Path(camera["video_path"])
            st.write(f"{camera['camera_id']}: {path.name} ({path.stat().st_size:,} bytes)")

    with st.expander("Synchronization"):
        offsets: dict[str, float] = {}
        for camera in state.get("manifest", {}).get("cameras", []):
            offsets[camera["camera_id"]] = st.number_input(
                f"{camera['camera_id']} offset (seconds)", value=0.0, step=0.04
            )
        state["synchronization"] = offsets

    with st.expander("Calibration"):
        st.write("Calibration runs through the dedicated CLI and is never loaded in this panel.")
        state["calibration_path"] = st.text_input("Calibration JSON (optional)")

    with st.expander("Analysis Settings"):
        state["config_path"] = st.text_input(
            "Full-match config", value=str(DEFAULT_FULL_MATCH_CONFIG)
        )
        state["adapter_config"] = st.text_input(
            "Chunk pipeline adapter (real model stages)",
            value=str(DEFAULT_ADAPTER_CONFIG),
        )
        state["rerun_from"] = st.text_input("Rerun from stage (optional)")

    with st.expander("Process Management"):
        if st.button("Prepare and start full-match analysis"):
            try:
                manifest = state["manifest"]
                match = manifest["match_id"]
                prepared_dir = Path(match) / "prepared"
                run_dir = Path(match) / "run"
                prepare_command = build_prepare_command(
                    match,
                    [camera["video_path"] for camera in manifest["cameras"]],
                    prepared_dir,
                    config_path=state.get("config_path", DEFAULT_FULL_MATCH_CONFIG),
                    force=True,
                )
                run_command = build_full_match_run_command(
                    prepared_dir,
                    run_dir,
                    config_path=state.get("config_path", DEFAULT_FULL_MATCH_CONFIG),
                    chunk_pipeline_config=state.get("adapter_config") or None,
                )
                subprocess.run(prepare_command, cwd=PROJECT_ROOT, check=True)  # noqa: S603
                process = subprocess.Popen(run_command, cwd=PROJECT_ROOT)  # noqa: S603
                state["run_dir"] = str(run_dir)
                state["processes"] = [process]
                st.success(
                    f"Prepared {safe_result_path(prepared_dir)} and started run "
                    f"{safe_result_path(run_dir)} (PID {process.pid})."
                )
            except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
                st.error(str(exc))
        if st.button("Resume run") and state.get("run_dir"):
            try:
                command = build_resume_command(
                    state["run_dir"], rerun_from_stage=state.get("rerun_from") or None
                )
                process = subprocess.Popen(command, cwd=PROJECT_ROOT)  # noqa: S603
                state.setdefault("processes", []).append(process)
                st.success(f"Resume started (PID {process.pid}).")
            except (OSError, ValueError) as exc:
                st.error(str(exc))
        for process in state.get("processes", []):
            st.write(f"PID {process.pid}: {'running' if process.poll() is None else process.returncode}")
        if state.get("run_dir"):
            progress = read_run_progress(state["run_dir"])
            st.write(f"Run directory: {progress['run_dir']}")
            if progress.get("match_id"):
                st.write(f"Match / run id: {progress['match_id']}")
            if progress.get("stages"):
                st.json(progress["stages"])
            chunks = progress.get("chunks")
            if chunks:
                if chunks.get("percent") is not None:
                    st.progress(int(chunks["percent"]))
                st.json(chunks["counts"])

    with st.expander("Global Identities"):
        merges = _json_editor(st, "Identity merges (JSON)", "[]")
        splits = _json_editor(st, "Identity splits (JSON)", "[]")
        if st.button("Save identity corrections"):
            try:
                st.success(str(persist_identity_corrections(match_id, merges=merges, splits=splits)))
            except (OSError, ValueError) as exc:
                st.error(str(exc))

    with st.expander("Roles"):
        roles = _json_editor(st, "Role corrections (JSON)", "[]")

    with st.expander("Match Events"):
        events = _json_editor(st, "Event corrections (JSON)", "[]")
        if st.button("Save role and event corrections"):
            try:
                st.success(str(persist_role_event_corrections(match_id, roles=roles, events=events)))
            except (OSError, ValueError) as exc:
                st.error(str(exc))

    with st.expander("Results"):
        try:
            for result in discover_results(match_id):
                st.download_button(
                    str(result.relative_to(safe_result_path(match_id))),
                    data=read_result_download(result),
                    file_name=result.name,
                )
        except ValueError as exc:
            st.info(str(exc))


def main() -> None:
    """Launch the Streamlit UI, with an actionable optional-dependency error."""

    _render_panel(_load_streamlit())


if __name__ == "__main__":
    main()
