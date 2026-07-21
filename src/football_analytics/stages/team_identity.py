"""Streaming temporal shirt-colour identity stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from football_analytics.analytics.role_identity import PlayerRole
from football_analytics.analytics.team_identity import TeamIdentityAssigner, sample_upper_torso
from football_analytics.contracts.schemas import (
    TRACK_IDENTITIES_SCHEMA,
    validate_mvp2_columns,
)
from football_analytics.geometry.bbox import BBox
from football_analytics.stages.base import Stage
from football_analytics.stages.mvp2_common import (
    canonical_common,
    read_required_parquet,
    video_fps,
)
from football_analytics.utils.io import write_rows_with_schema
from football_analytics.video.streaming import StreamingVideoReader


class TeamIdentityStage(Stage):
    name = "team_identity"

    def validate_inputs(self) -> None:
        read_required_parquet(self.run_dir / "tracks.parquet")
        read_required_parquet(self.run_dir / "track_quality.parquet")
        read_required_parquet(self.run_dir / "shot_segments.parquet")
        if not (self.run_dir / "input" / "test_clip.mp4").is_file():
            raise FileNotFoundError("working video missing")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        tracks = read_required_parquet(self.run_dir / "tracks.parquet")
        shots = read_required_parquet(self.run_dir / "shot_segments.parquet")
        cfg = self.config["team_identity"]
        assigner = TeamIdentityAssigner(
            min_samples=int(cfg["minimum_team_training_samples"]),
            min_tracks=max(2, min(4, int(cfg["minimum_team_training_samples"]) // 2)),
            history_size=int(cfg["maximum_samples_per_track"]),
            min_valid_pixel_fraction=float(self.config["geometry"]["minimum_crop_quality"]),
            unknown_confidence_threshold=float(cfg["minimum_team_confidence"]),
        )
        fps = video_fps(self.run_dir)
        grouped = {int(key): value for key, value in tracks.groupby("frame_id")}
        cut_frames = set(
            shots.loc[shots["scene_cut"].fillna(False), "frame_id"].astype(int).tolist()
        )
        rows: list[dict[str, Any]] = []
        reader = StreamingVideoReader(
            self.run_dir / "input" / "test_clip.mp4",
            chunk_seconds=float(self.config["runtime"]["chunk_seconds"]),
        )
        for video_frame in reader:
            frame_id = video_frame.frame_id
            image = video_frame.image
            if frame_id in cut_frames and bool(cfg.get("reset_on_scene_cut", True)):
                assigner.reset_scene()
            frame_tracks = grouped.get(frame_id)
            if frame_tracks is not None:
                observations: list[tuple[Any, float]] = []
                for item in frame_tracks.itertuples():
                    box = BBox(
                        float(item.bbox_x1),
                        float(item.bbox_y1),
                        float(item.bbox_x2),
                        float(item.bbox_y2),
                    )
                    feature, color_quality = sample_upper_torso(image, box)
                    if feature is not None:
                        assigner.add_sample(
                            int(item.track_id),
                            feature,
                            role=PlayerRole.UNKNOWN,
                            valid_pixel_fraction=color_quality,
                        )
                    observations.append((item, color_quality))
                assigner.fit()
                for item, color_quality in observations:
                    assignment = assigner.assign(
                        int(item.track_id), role=PlayerRole.UNKNOWN
                    )
                    team_id = (
                        f"team_{assignment.team_id}"
                        if assignment.team_id is not None
                        else None
                    )
                    valid = team_id is not None
                    rows.append(
                        {
                            **canonical_common(
                                self.run_dir,
                                frame_id,
                                frame_id * 1000.0 / fps,
                                "temporal_upper_torso_lab_kmeans",
                                assignment.confidence,
                                valid,
                            ),
                            "track_id": int(item.track_id),
                            "role": "unknown",
                            "role_confidence": 0.0,
                            "team_id": team_id,
                            "team_confidence": float(assignment.confidence),
                            "color_quality": float(color_quality),
                            "temporal_consistency": float(
                                assignment.temporal_consistency
                            ),
                        }
                    )
        output = self.run_dir / "track_identities.parquet"
        write_rows_with_schema(output, rows, TRACK_IDENTITIES_SCHEMA)
        return {"track_identities": output}

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        frame = pd.read_parquet(artifacts["track_identities"])
        validate_mvp2_columns("track_identities", list(frame.columns))
