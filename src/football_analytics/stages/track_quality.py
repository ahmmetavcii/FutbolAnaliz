"""Canonical track-quality stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_analytics.analytics.track_quality import compute_track_quality
from football_analytics.contracts.schemas import TRACK_QUALITY_SCHEMA, validate_mvp2_columns
from football_analytics.geometry.bbox import BBox
from football_analytics.stages.base import Stage
from football_analytics.stages.mvp2_common import (
    canonical_common,
    load_video_manifest,
    read_required_parquet,
    video_frame_count,
    video_fps,
)
from football_analytics.utils.io import write_rows_with_schema


class TrackQualityStage(Stage):
    name = "track_quality"

    def validate_inputs(self) -> None:
        read_required_parquet(self.run_dir / "tracks.parquet")
        read_required_parquet(self.run_dir / "shot_segments.parquet")
        load_video_manifest(self.run_dir)

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        tracks = read_required_parquet(self.run_dir / "tracks.parquet")
        shots = read_required_parquet(self.run_dir / "shot_segments.parquet")
        scene_cut_frames = set(
            shots.loc[shots["scene_cut"].fillna(False), "frame_id"].astype(int)
        )
        manifest = load_video_manifest(self.run_dir)
        width = int(manifest["working_summary"]["width"])
        height = int(manifest["working_summary"]["height"])
        total_frames = video_frame_count(self.run_dir)
        fps = video_fps(self.run_dir)
        cfg = self.config["track_quality"]
        geometry_cfg = self.config["geometry"]
        rows: list[dict[str, Any]] = []

        for track_id, group in tracks.groupby("track_id", sort=True):
            group = group.sort_values("frame_id")
            qualities = [
                BBox(
                    float(row.bbox_x1),
                    float(row.bbox_y1),
                    float(row.bbox_x2),
                    float(row.bbox_y2),
                ).player_crop_quality(
                    width,
                    height,
                    min_area=float(geometry_cfg["minimum_bbox_area"]),
                    min_visible_fraction=float(geometry_cfg["minimum_visible_ratio"]),
                )
                for row in group.itertuples()
            ]
            quality = compute_track_quality(
                group["frame_id"].astype(int).tolist(),
                total_frames=total_frames,
                tracking_confidences=group["tracking_confidence"].astype(float).tolist(),
                bbox_valid=[item.valid for item in qualities],
                visibility=[item.visible_fraction for item in qualities],
                foot_point_confidences=[item.foot_point_confidence for item in qualities],
            )
            centres = np.column_stack(
                [
                    (group["bbox_x1"].to_numpy() + group["bbox_x2"].to_numpy()) / 2.0,
                    group["bbox_y2"].to_numpy(),
                ]
            )
            heights = np.maximum(
                1.0, group["bbox_y2"].to_numpy() - group["bbox_y1"].to_numpy()
            )
            jitter = (
                float(np.median(np.linalg.norm(np.diff(centres, axis=0), axis=1) / heights[1:]))
                if len(group) > 1
                else 0.0
            )
            border_truncation = float(
                1.0 - np.mean([item.visible_fraction for item in qualities])
            )
            usable = (
                quality.observations >= int(cfg["minimum_track_frames"])
                and quality.coverage >= float(cfg["minimum_coverage"])
                and quality.mean_tracking_confidence
                >= float(cfg["minimum_detection_confidence"])
                and jitter <= float(cfg["maximum_bbox_jitter"])
                and border_truncation <= float(cfg["maximum_border_truncation"])
            )
            last_frame = int(group["frame_id"].max())
            confidence = float(
                np.clip(
                    quality.quality_score
                    * max(0.0, 1.0 - min(1.0, jitter))
                    * max(0.0, 1.0 - border_truncation),
                    0.0,
                    1.0,
                )
            )
            rows.append(
                {
                    **canonical_common(
                        self.run_dir,
                        last_frame,
                        last_frame * 1000.0 / fps,
                        "canonical_track_quality_v1",
                        confidence,
                        usable,
                    ),
                    "track_id": int(track_id),
                    "track_length": int(quality.frame_span),
                    "visible_frames": int(quality.observations),
                    "coverage": float(quality.coverage),
                    "detection_confidence_mean": float(
                        group["tracking_confidence"].mean()
                    ),
                    "detection_confidence_min": float(
                        group["tracking_confidence"].min()
                    ),
                    "bbox_jitter": jitter,
                    "fragmentation": int(quality.fragmentation_count),
                    "scene_cut_count": sum(
                        int(frame_id) in scene_cut_frames
                        for frame_id in group["frame_id"]
                    ),
                    "border_truncation": border_truncation,
                    "team_consistency": None,
                    "usable_for_metrics": bool(usable),
                    "invalid_reason": None if usable else "track_quality_gate_failed",
                }
            )
        output = self.run_dir / "track_quality.parquet"
        write_rows_with_schema(output, rows, TRACK_QUALITY_SCHEMA)
        return {"track_quality": output}

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        frame = pd.read_parquet(artifacts["track_quality"])
        validate_mvp2_columns("track_quality", list(frame.columns))
