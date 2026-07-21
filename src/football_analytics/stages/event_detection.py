"""Pipeline stage: event_detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from football_analytics.events.orchestrator import run_event_detection
from football_analytics.stages.base import Stage
from football_analytics.utils.io import read_yaml


class EventDetectionStage(Stage):
    name = "event_detection"

    def validate_inputs(self) -> None:
        # Prefer existing MVP-2 artifacts; do not re-run detection/tracking.
        if not (self.run_dir / "tracks.parquet").is_file():
            raise FileNotFoundError("tracks.parquet required for event detection")
        if not (self.run_dir / "input" / "test_clip.mp4").is_file():
            raise FileNotFoundError("working video missing")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        event_cfg_path = Path("/home/ahmet/projects/football-analytics/configs/events/goal_detection.yaml")
        event_cfg = read_yaml(event_cfg_path) if event_cfg_path.is_file() else {}
        # Allow pipeline yaml override under key `events`.
        pipeline_events = self.config.get("events") or {}
        if pipeline_events:
            for key, value in pipeline_events.items():
                if isinstance(value, dict) and isinstance(event_cfg.get(key), dict):
                    merged = dict(event_cfg[key])
                    merged.update(value)
                    event_cfg[key] = merged
                else:
                    event_cfg[key] = value
        manifest = run_event_detection(
            self.run_dir,
            config=event_cfg,
            video_path=self.run_dir / "input" / "test_clip.mp4",
        )
        return {
            "match_events": self.run_dir / "match_events.parquet",
            "events": self.run_dir / "events.parquet",
            "event_evidence": self.run_dir / "event_evidence.json",
            "stage_manifest": self.run_dir / "stage_manifests" / "event_detection.json",
            "metrics": self.stage_dir / "metrics.json",
            "confirmed_event_count": manifest.get("confirmed_event_count", 0),
            "candidate_event_count": manifest.get("candidate_event_count", 0),
        }

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        for key in ("match_events", "events", "event_evidence"):
            path = Path(artifacts[key])
            if not path.is_file():
                raise FileNotFoundError(path)
        frame = pd.read_parquet(artifacts["match_events"])
        required = {"event_id", "event_type", "status", "confidence"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"match_events missing columns: {sorted(missing)}")
