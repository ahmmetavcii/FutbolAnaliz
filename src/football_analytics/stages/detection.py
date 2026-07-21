"""Detection stage using Ultralytics adapter."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from football_analytics.adapters.ultralytics_adapter import UltralyticsAdapter
from football_analytics.contracts.schemas import validate_detections_frame
from football_analytics.stages.base import Stage
from football_analytics.utils.io import write_json, write_parquet
from football_analytics.video.ffprobe import probe_video, summarize_probe


class DetectionStage(Stage):
    name = "detection"

    def validate_inputs(self) -> None:
        video = self.run_dir / "input" / "test_clip.mp4"
        model = Path(self.config["model"]["path"])
        if not video.is_file():
            raise FileNotFoundError(f"Working video missing: {video}")
        if not model.is_file():
            raise FileNotFoundError(f"Model missing: {model}")

    def prepare(self) -> None:
        (self.stage_dir / "ultralytics").mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        video = self.run_dir / "input" / "test_clip.mp4"
        model_cfg = self.config["model"]
        fps = float(summarize_probe(probe_video(video))["avg_frame_rate"] or 25.0)
        if fps <= 0:
            fps = 25.0
        adapter = UltralyticsAdapter(
            model_path=Path(model_cfg["path"]),
            device=model_cfg.get("device", 0),
            imgsz=int(model_cfg.get("imgsz", 640)),
            batch=int(model_cfg.get("batch", 1)),
            half=bool(model_cfg.get("half", True)),
            conf=float(model_cfg.get("conf", 0.25)),
            classes=model_cfg.get("classes"),
        )
        frame, metrics = adapter.detect(
            video_path=video,
            project=self.stage_dir / "ultralytics",
            name="predict",
            save=bool(self.config.get("detection", {}).get("save_annotated", True)),
        )
        if not frame.empty:
            frame["timestamp_ms"] = frame["frame_id"].astype(float) * (1000.0 / fps)
        parquet_path = self.run_dir / "detections.parquet"
        write_parquet(parquet_path, frame)

        annotated_src = self._find_annotated_video(self.stage_dir / "ultralytics" / "predict")
        annotated_dst = self.run_dir / "detection_annotated.mp4"
        artifacts: dict[str, Any] = {
            "detections": parquet_path,
            "metrics": self.stage_dir / "metrics.json",
        }
        write_json(artifacts["metrics"], metrics)
        if annotated_src is None:
            raise RuntimeError("Ultralytics detection did not write an annotated video")
        self._materialize_mp4(annotated_src, annotated_dst)
        artifacts["detection_annotated"] = annotated_dst
        artifacts["ultralytics_annotated"] = annotated_src
        return artifacts

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        path = Path(artifacts["detections"])
        if not path.is_file():
            raise RuntimeError("detections.parquet missing")
        frame = pd.read_parquet(path)
        validate_detections_frame(list(frame.columns))
        if len(frame) < 1:
            raise RuntimeError("detections.parquet has zero rows — detection failed")
        annotated = artifacts.get("detection_annotated")
        if annotated is None or not Path(annotated).is_file():
            raise RuntimeError("detection annotated video missing")

    @staticmethod
    def _find_annotated_video(directory: Path) -> Path | None:
        if not directory.exists():
            return None
        for pattern in ("*.mp4", "*.avi", "*.mkv"):
            candidates = sorted(directory.rglob(pattern))
            if candidates:
                return candidates[0]
        return None

    @staticmethod
    def _materialize_mp4(source: Path, target: Path) -> None:
        import subprocess

        if source.suffix.lower() == ".mp4":
            shutil.copy2(source, target)
            return
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(target),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
