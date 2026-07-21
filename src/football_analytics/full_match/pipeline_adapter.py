"""Adapter binding the proven single-feed MVP pipeline to full-match chunks.

The adapter does not reimplement detection/tracking/calibration logic. It
invokes :class:`football_analytics.orchestration.runner.PipelineRunner` (the
same code path exercised by ``scripts/run_pipeline.py``) on one chunk of one
camera and validates the artifacts the runner produced. Model outputs are
therefore real or absent - never synthesized here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from football_analytics.utils.io import read_yaml

from .manifest import atomic_write_json
from .schemas import ChunkRecord

DEFAULT_PIPELINE_CONFIG = "configs/pipeline/opta_analytics.yaml"

#: Environment variable that tells the module-level chunk processor where to
#: place per-chunk pipeline artifacts (the scheduler only hands the processor
#: a video path and a chunk record).
ARTIFACT_ROOT_ENV = "FA_FULL_MATCH_CHUNK_ARTIFACTS"
PIPELINE_CONFIG_ENV = "FA_FULL_MATCH_PIPELINE_CONFIG"

#: Artifacts the underlying pipeline must produce for a chunk to PASS, with
#: whether an empty table is acceptable (empty=no relevant objects found).
REQUIRED_ARTIFACTS: dict[str, bool] = {
    "detections.parquet": False,
    "tracks.parquet": False,
    "shot_segments.parquet": False,
    "calibration.parquet": True,
    "player_metrics.parquet": True,
    "track_identities.parquet": True,
    "game_state.parquet": True,
    "ball_state.parquet": True,
}


class PipelineAdapterError(RuntimeError):
    """The existing pipeline could not process a chunk."""


@dataclass(frozen=True)
class ChunkContext:
    """Everything a chunk processor needs to run one unit of work."""

    run_id: str
    camera_id: str
    period: int
    chunk_index: int
    source_path: Path
    frame_start: int
    frame_end: int
    start_seconds: float
    end_seconds: float
    config: dict[str, Any] = field(default_factory=dict)
    output_dir: Path = Path(".")

    @classmethod
    def from_record(
        cls,
        video_path: Path,
        record: ChunkRecord,
        *,
        run_id: str,
        output_dir: Path,
        config: dict[str, Any] | None = None,
        period: int = 1,
    ) -> "ChunkContext":
        return cls(
            run_id=run_id,
            camera_id=record.camera_id,
            period=period,
            chunk_index=record.chunk_index,
            source_path=Path(video_path),
            frame_start=record.frame_start,
            frame_end=record.frame_end,
            start_seconds=record.start_seconds,
            end_seconds=record.end_seconds,
            config=dict(config or {}),
            output_dir=Path(output_dir),
        )


class ExistingPipelineAdapter:
    """Run the real MVP pipeline for one chunk and validate its artifacts."""

    def __init__(
        self,
        pipeline_config_path: Path | str = DEFAULT_PIPELINE_CONFIG,
        project_root: Path | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.project_root = Path(
            project_root or Path(__file__).resolve().parents[3]
        )
        self.pipeline_config_path = (self.project_root / pipeline_config_path
                                     if not Path(pipeline_config_path).is_absolute()
                                     else Path(pipeline_config_path))
        self.python_executable = python_executable or os.environ.get(
            "FA_PYTHON", "python"
        )
        self._temp_clips: list[Path] = []

    # -- adapter interface --------------------------------------------------

    def validate_inputs(self, context: ChunkContext) -> None:
        if not context.source_path.is_file():
            raise PipelineAdapterError(f"source video missing: {context.source_path}")
        if not self.pipeline_config_path.is_file():
            raise PipelineAdapterError(
                f"pipeline config missing: {self.pipeline_config_path}"
            )
        if context.frame_end <= context.frame_start:
            raise PipelineAdapterError(
                f"empty frame range {context.frame_start}..{context.frame_end}"
            )

    def prepare(self, context: ChunkContext) -> dict[str, Any]:
        context.output_dir.mkdir(parents=True, exist_ok=True)
        config = read_yaml(self.pipeline_config_path)
        return {"pipeline_config": str(self.pipeline_config_path), "config": config}

    def _chunk_source(self, context: ChunkContext) -> Path:
        """Return a video covering exactly this chunk (stream-copied, no decode)."""
        duration = context.end_seconds - context.start_seconds
        source_duration = self._probe_duration(context.source_path)
        if context.start_seconds <= 0.01 and duration >= source_duration - 0.5:
            return context.source_path
        clip = context.output_dir / (
            f"{context.camera_id}_p{context.period}_chunk{context.chunk_index:05d}.mp4"
        )
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{context.start_seconds:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(context.source_path),
            "-c", "copy", str(clip),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0 or not clip.is_file():
            raise PipelineAdapterError(f"ffmpeg chunk extract failed: {completed.stderr}")
        self._temp_clips.append(clip)
        return clip

    @staticmethod
    def _probe_duration(path: Path) -> float:
        command = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1", str(path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        try:
            return float(completed.stdout.strip())
        except ValueError as exc:
            raise PipelineAdapterError(f"ffprobe failed for {path}") from exc

    def run_chunk(self, context: ChunkContext) -> dict[str, Any]:
        """Execute the real pipeline for one chunk; returns the chunk payload."""
        self.validate_inputs(context)
        self.prepare(context)
        clip = self._chunk_source(context)
        runs_root = context.output_dir / "pipeline_runs"
        runs_root.mkdir(parents=True, exist_ok=True)

        command = [
            self.python_executable,
            str(self.project_root / "scripts" / "run_pipeline.py"),
            "--config", str(self.pipeline_config_path),
            "--input", str(clip),
            "--runs-root", str(runs_root),
        ]
        env = dict(os.environ)
        env.setdefault("PYTHONPATH", str(self.project_root / "src"))
        # Stream pipeline output to disk so progress/stall monitors can tail it.
        log_path = context.output_dir / "pipeline_stdout.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command, stdout=log_handle, stderr=subprocess.STDOUT, text=True,
                env=env, cwd=str(self.project_root),
            )
        run_dirs = sorted(runs_root.glob("run_*"))
        if completed.returncode != 0 or not run_dirs:
            tail = ""
            if log_path.is_file():
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-800:]
            raise PipelineAdapterError(
                "pipeline failed for chunk "
                f"{context.camera_id}/{context.chunk_index}: rc={completed.returncode} "
                f"log_tail={tail}"
            )
        pipeline_run_dir = run_dirs[-1]
        validation = self.validate_outputs(pipeline_run_dir)
        report = json.loads((pipeline_run_dir / "run_report.json").read_text())

        payload = {
            "run_id": context.run_id,
            "camera_id": context.camera_id,
            "period": context.period,
            "chunk_index": context.chunk_index,
            "start_seconds": context.start_seconds,
            "end_seconds": context.end_seconds,
            "frames_decoded": int(
                (report.get("detection_metrics") or {}).get("frames") or 0
            ),
            "pipeline_run_dir": str(pipeline_run_dir),
            "pipeline_status": report.get("status"),
            "model": report.get("model", {}),
            "detections": int(
                (report.get("detection_metrics") or {}).get("detections") or 0
            ),
            "track_rows": int(
                ((report.get("tracking_metrics") or {}).get("bytetrack") or {}).get(
                    "track_rows"
                )
                or 0
            ),
            "unique_track_ids": int(
                ((report.get("tracking_metrics") or {}).get("bytetrack") or {}).get(
                    "unique_track_ids"
                )
                or 0
            ),
            "peak_ram_bytes": report.get("peak_ram_bytes"),
            "peak_vram_bytes": max(
                int((report.get("detection_metrics") or {}).get("peak_vram_bytes") or 0),
                int(
                    ((report.get("tracking_metrics") or {}).get("bytetrack") or {}).get(
                        "peak_vram_bytes"
                    )
                    or 0
                ),
            ),
            "artifact_validation": validation,
            "artifact_manifest": self.get_artifact_manifest(pipeline_run_dir),
            "model_stage_status": (
                "PASS" if report.get("status") == "PASS" else "FAILED"
            ),
            "model_outputs": "real",
        }
        return payload

    def validate_outputs(self, pipeline_run_dir: Path) -> dict[str, Any]:
        """Parse every required artifact; fail when a mandatory table is empty."""
        results: dict[str, Any] = {}
        for name, empty_ok in REQUIRED_ARTIFACTS.items():
            path = pipeline_run_dir / name
            entry: dict[str, Any] = {"exists": path.is_file(), "rows": None, "ok": False}
            if path.is_file():
                try:
                    frame = pd.read_parquet(path)
                    entry["rows"] = int(len(frame))
                    entry["ok"] = bool(len(frame) > 0 or empty_ok)
                except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                    entry["error"] = f"{type(exc).__name__}: {exc}"
            results[name] = entry
        bad = [name for name, entry in results.items() if not entry["ok"]]
        if bad:
            raise PipelineAdapterError(
                f"pipeline artifacts failed validation in {pipeline_run_dir}: {bad}"
            )
        return results

    def get_artifact_manifest(self, pipeline_run_dir: Path) -> dict[str, str]:
        manifest: dict[str, str] = {}
        for name in REQUIRED_ARTIFACTS:
            path = pipeline_run_dir / name
            if path.is_file():
                manifest[name] = str(path)
        for extra in ("run_report.json", "video_manifest.json",
                      "analytics_annotated.mp4", "annotated_video.mp4",
                      "tactical_preview.mp4"):
            path = pipeline_run_dir / extra
            if path.is_file():
                manifest[extra] = str(path)
        return manifest

    def cleanup(self) -> None:
        for clip in self._temp_clips:
            clip.unlink(missing_ok=True)
        self._temp_clips.clear()


def run_chunk_via_existing_pipeline(
    video_path: Path, record: ChunkRecord
) -> dict[str, Any]:
    """ChunkScheduler-compatible processor delegating to the real pipeline.

    The artifact root and pipeline config are taken from the environment
    because the scheduler's processor signature carries only the video path
    and the chunk record.
    """
    artifact_root = Path(
        os.environ.get(ARTIFACT_ROOT_ENV)
        or Path(tempfile.gettempdir()) / "full_match_chunk_artifacts"
    )
    adapter = ExistingPipelineAdapter(
        pipeline_config_path=os.environ.get(PIPELINE_CONFIG_ENV, DEFAULT_PIPELINE_CONFIG),
    )
    context = ChunkContext.from_record(
        Path(video_path),
        record,
        run_id=os.environ.get("FA_FULL_MATCH_RUN_ID", "full_match_run"),
        output_dir=artifact_root / record.camera_id / f"chunk_{record.chunk_index:05d}",
    )
    try:
        payload = adapter.run_chunk(context)
    finally:
        adapter.cleanup()
    atomic_write_json(context.output_dir / "chunk_payload.json", payload)
    return payload
