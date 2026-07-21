"""Tracking stage: ByteTrack and BoT-SORT via Ultralytics."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from football_analytics.adapters.ultralytics_adapter import UltralyticsAdapter
from football_analytics.contracts.schemas import validate_tracks_frame
from football_analytics.stages.base import Stage
from football_analytics.utils.io import write_json, write_parquet
from football_analytics.video.ffprobe import probe_video, summarize_probe


class TrackingStage(Stage):
    name = "tracking"

    def validate_inputs(self) -> None:
        video = self.run_dir / "input" / "test_clip.mp4"
        detections = self.run_dir / "detections.parquet"
        model = Path(self.config["model"]["path"])
        if not video.is_file():
            raise FileNotFoundError(f"Working video missing: {video}")
        if not detections.is_file():
            raise FileNotFoundError(f"detections.parquet missing: {detections}")
        if not model.is_file():
            raise FileNotFoundError(f"Model missing: {model}")

    def prepare(self) -> None:
        (self.stage_dir / "ultralytics").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "trackers").mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        video = self.run_dir / "input" / "test_clip.mp4"
        model_cfg = self.config["model"]
        tracking_cfg = self.config.get("tracking", {})
        primary = tracking_cfg.get("primary_tracker", "bytetrack")
        trackers = list(tracking_cfg.get("trackers", [primary]))
        if primary not in trackers:
            trackers.insert(0, primary)

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

        comparison: dict[str, Any] = {}
        artifacts: dict[str, Any] = {}
        primary_frame: pd.DataFrame | None = None

        for tracker in trackers:
            frame, metrics = adapter.track(
                video_path=video,
                tracker=tracker,
                project=self.stage_dir / "ultralytics",
                name=tracker,
                save=True,
                persist=bool(tracking_cfg.get("persist", True)),
            )
            if not frame.empty:
                frame["timestamp_ms"] = frame["frame_id"].astype(float) * (1000.0 / fps)

            tracker_dir = self.run_dir / "trackers" / tracker
            tracker_dir.mkdir(parents=True, exist_ok=True)
            tracker_parquet = tracker_dir / "tracks.parquet"
            write_parquet(tracker_parquet, frame)
            write_json(tracker_dir / "metrics.json", metrics)

            annotated_src = self._find_annotated_video(
                self.stage_dir / "ultralytics" / tracker
            )
            if annotated_src is None:
                raise RuntimeError(f"Tracker {tracker} did not write annotated video")
            annotated_dst = tracker_dir / "annotated_video.mp4"
            self._materialize_mp4(annotated_src, annotated_dst)
            artifacts[f"{tracker}_annotated"] = annotated_dst
            metrics["annotated_video"] = str(annotated_dst)

            if len(frame) < 1:
                raise RuntimeError(f"Tracker {tracker} produced zero track rows")
            if int(metrics.get("unique_track_ids", 0)) < 1:
                raise RuntimeError(f"Tracker {tracker} produced zero track IDs")

            comparison[tracker] = metrics
            artifacts[f"{tracker}_tracks"] = tracker_parquet
            if tracker == primary:
                primary_frame = frame

        assert primary_frame is not None
        canonical_tracks = self.run_dir / "tracks.parquet"
        write_parquet(canonical_tracks, primary_frame)
        artifacts["tracks"] = canonical_tracks

        primary_annotated = artifacts.get(f"{primary}_annotated")
        if primary_annotated is None:
            raise RuntimeError("Primary tracker annotated video missing")
        annotated_video = self.run_dir / "annotated_video.mp4"
        shutil.copy2(primary_annotated, annotated_video)
        artifacts["annotated_video"] = annotated_video

        comparison_path = self.stage_dir / "tracker_comparison.json"
        write_json(comparison_path, comparison)
        artifacts["tracker_comparison"] = comparison_path
        return artifacts

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        path = Path(artifacts["tracks"])
        if not path.is_file():
            raise RuntimeError("tracks.parquet missing")
        frame = pd.read_parquet(path)
        validate_tracks_frame(list(frame.columns))
        if len(frame) < 1:
            raise RuntimeError("tracks.parquet has zero rows")
        if frame["track_id"].nunique() < 1:
            raise RuntimeError("tracks.parquet has no track IDs")
        annotated = artifacts.get("annotated_video")
        if annotated is None or not Path(annotated).is_file():
            raise RuntimeError("annotated_video.mp4 missing")
        for tracker in self.config.get("tracking", {}).get("trackers", []):
            key = f"{tracker}_tracks"
            if key not in artifacts or not Path(artifacts[key]).is_file():
                raise RuntimeError(f"Missing tracker parquet for {tracker}")

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
