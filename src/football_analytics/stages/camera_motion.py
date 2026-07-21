"""Streaming robust camera-motion stage."""

from __future__ import annotations

from typing import Any

import pandas as pd

from football_analytics.analytics.camera_motion import (
    CameraMotionConfig,
    CameraMotionEstimator,
)
from football_analytics.contracts.schemas import CAMERA_MOTION_SCHEMA, validate_mvp2_columns
from football_analytics.stages.base import Stage
from football_analytics.stages.mvp2_common import canonical_common, read_required_parquet
from football_analytics.utils.io import write_rows_with_schema
from football_analytics.video.streaming import StreamingVideoReader


class CameraMotionStage(Stage):
    name = "camera_motion"

    def validate_inputs(self) -> None:
        read_required_parquet(self.run_dir / "tracks.parquet")
        read_required_parquet(self.run_dir / "shot_segments.parquet")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        tracks = read_required_parquet(self.run_dir / "tracks.parquet")
        grouped = {int(key): value for key, value in tracks.groupby("frame_id")}
        cfg = self.config["camera_motion"]
        estimator = CameraMotionEstimator(
            CameraMotionConfig(
                border_region_ratio=float(cfg["border_region_ratio"]),
                max_corners=int(cfg["max_corners"]),
                quality_level=float(cfg["quality_level"]),
                minimum_feature_distance=float(cfg["minimum_feature_distance"]),
                forward_backward_error=float(cfg["forward_backward_error"]),
                ransac_reprojection_threshold=float(
                    cfg["ransac_reprojection_threshold"]
                ),
                minimum_inlier_ratio=float(cfg["minimum_inlier_ratio"]),
                minimum_inliers=int(cfg["minimum_inliers"]),
                scene_cut_reset=bool(cfg["scene_cut_reset"]),
                scene_cut_histogram_threshold=float(
                    self.config["shot_classifier"]["scene_cut_histogram_threshold"]
                ),
            )
        )
        rows: list[dict[str, Any]] = []
        for video_frame in StreamingVideoReader(
            self.run_dir / "input" / "test_clip.mp4",
            chunk_seconds=float(self.config["runtime"]["chunk_seconds"]),
        ):
            frame_tracks = grouped.get(video_frame.frame_id)
            boxes = (
                frame_tracks[
                    ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
                ].to_numpy().tolist()
                if frame_tracks is not None
                else []
            )
            result = estimator.update(video_frame.image, exclude_bboxes=boxes)
            rows.append(
                {
                    **canonical_common(
                        self.run_dir,
                        video_frame.frame_id,
                        video_frame.timestamp_ms,
                        "lk_forward_backward_ransac_affine",
                        result.confidence,
                        result.valid,
                    ),
                    "dx_pixel": result.dx,
                    "dy_pixel": result.dy,
                    "rotation_deg": result.rotation,
                    "scale": result.scale,
                    "inlier_count": result.inlier_count,
                    "inlier_ratio": result.inlier_ratio,
                    "reset_reason": result.reset_reason,
                }
            )
        output = self.run_dir / "camera_motion.parquet"
        write_rows_with_schema(output, rows, CAMERA_MOTION_SCHEMA)
        return {"camera_motion": output}

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        frame = pd.read_parquet(artifacts["camera_motion"])
        validate_mvp2_columns("camera_motion", list(frame.columns))
        if frame.empty:
            raise RuntimeError("camera motion produced zero rows")
