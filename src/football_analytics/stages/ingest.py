"""Ingest stage: ffprobe + OpenCV validation + video_manifest."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from football_analytics.stages.base import Stage
from football_analytics.utils.hashing import sha256_file
from football_analytics.utils.io import write_json
from football_analytics.video.ffprobe import probe_video, summarize_probe
from football_analytics.video.opencv_io import prepare_test_clip, verify_opencv_readable


class IngestStage(Stage):
    name = "ingest"

    def __init__(self, run_dir: Path, config: dict[str, Any], input_video: Path) -> None:
        super().__init__(run_dir, config)
        self.input_video = Path(input_video)
        self.working_video = self.run_dir / "input" / "test_clip.mp4"

    def validate_inputs(self) -> None:
        if not self.input_video.is_file():
            raise FileNotFoundError(f"Input video missing: {self.input_video}")
        if self.input_video.stat().st_size < 1024:
            raise RuntimeError(f"Input video too small / invalid: {self.input_video}")

    def prepare(self) -> None:
        self.working_video.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        max_seconds = float(self.config.get("runtime", {}).get("max_clip_seconds", 20))
        probe = probe_video(self.input_video)
        summary = summarize_probe(probe)
        clip_info = prepare_test_clip(
            self.input_video,
            self.working_video,
            max_seconds=max_seconds,
            duration_seconds=float(summary["duration_seconds"]),
        )
        # Re-probe working clip and validate OpenCV.
        working_probe = probe_video(self.working_video)
        working_summary = summarize_probe(working_probe)
        opencv = verify_opencv_readable(self.working_video)

        ffprobe_path = self.stage_dir / "ffprobe.json"
        write_json(ffprobe_path, working_probe)

        manifest = {
            "source_path": str(self.input_video),
            "working_path": str(self.working_video),
            "sha256": sha256_file(self.working_video),
            "clip_preparation": clip_info,
            "source_summary": summary,
            "working_summary": working_summary,
            "opencv": opencv,
        }
        manifest_path = self.run_dir / "video_manifest.json"
        write_json(manifest_path, manifest)
        # Keep a stage-local copy for resume clarity.
        shutil.copy2(manifest_path, self.stage_dir / "video_manifest.json")
        return {
            "video_manifest": manifest_path,
            "ffprobe": ffprobe_path,
            "working_video": self.working_video,
        }

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        for key in ("video_manifest", "ffprobe", "working_video"):
            path = Path(artifacts[key])
            if not path.exists():
                raise RuntimeError(f"Ingest missing artifact: {key}")
        opencv = verify_opencv_readable(Path(artifacts["working_video"]), max_frames=5)
        if not opencv["opened"]:
            raise RuntimeError("OpenCV validation failed after ingest")
