"""Streaming shot/scene classification stage."""

from __future__ import annotations

from typing import Any

import pandas as pd

from football_analytics.analytics.shot_classifier import (
    ShotClassifier,
    ShotClassifierConfig,
)
from football_analytics.contracts.schemas import SHOT_SEGMENTS_SCHEMA, validate_mvp2_columns
from football_analytics.stages.base import Stage
from football_analytics.stages.mvp2_common import canonical_common
from football_analytics.utils.io import write_rows_with_schema
from football_analytics.video.streaming import StreamingVideoReader


class ShotClassificationStage(Stage):
    name = "shot_classification"

    def validate_inputs(self) -> None:
        if not (self.run_dir / "input" / "test_clip.mp4").is_file():
            raise FileNotFoundError("working video missing")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        cfg = self.config["shot_classifier"]
        classifier = ShotClassifier(
            ShotClassifierConfig(
                green_hue_min=int(self.config["geometry"]["green_hue_min"]),
                green_hue_max=int(self.config["geometry"]["green_hue_max"]),
                wide_green_ratio_min=float(cfg["wide_green_ratio_min"]),
                medium_green_ratio_min=float(cfg["medium_green_ratio_min"]),
                close_up_player_height_ratio=float(
                    cfg["close_up_player_height_ratio"]
                ),
                graphic_edge_density_min=float(cfg["graphic_edge_density_min"]),
                scene_cut_histogram_threshold=float(
                    cfg["scene_cut_histogram_threshold"]
                ),
                minimum_confidence=float(cfg["minimum_confidence"]),
            )
        )
        valid_types = set(cfg["valid_for_spatial"])
        rows: list[dict[str, Any]] = []
        for video_frame in StreamingVideoReader(
            self.run_dir / "input" / "test_clip.mp4",
            chunk_seconds=float(self.config["runtime"]["chunk_seconds"]),
        ):
            result = classifier.update(video_frame.image)
            valid = result.shot_type in valid_types
            rows.append(
                {
                    **canonical_common(
                        self.run_dir,
                        video_frame.frame_id,
                        video_frame.timestamp_ms,
                        "heuristic_streaming_shot_classifier",
                        result.confidence,
                        valid,
                    ),
                    "shot_type": result.shot_type,
                    "green_ratio": result.green_ratio,
                    "mean_player_height_ratio": None,
                    "player_count": None,
                    "histogram_difference": result.histogram_diff,
                    "optical_flow_magnitude": result.flow,
                    "scene_cut": result.scene_cut,
                    "invalid_reason": (
                        None if valid else f"shot_type_{result.shot_type}_not_spatial"
                    ),
                }
            )
        output = self.run_dir / "shot_segments.parquet"
        write_rows_with_schema(output, rows, SHOT_SEGMENTS_SCHEMA)
        return {"shot_segments": output}

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        frame = pd.read_parquet(artifacts["shot_segments"])
        validate_mvp2_columns("shot_segments", list(frame.columns))
        if frame.empty:
            raise RuntimeError("shot classifier produced zero rows")
