"""Confidence-aware possession state-machine stage."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from football_analytics.analytics.possession import (
    BallSnapshot,
    PlayerSnapshot,
    PossessionConfig,
    PossessionTracker,
)
from football_analytics.contracts.schemas import (
    POSSESSION_TIMELINE_SCHEMA,
    validate_mvp2_columns,
)
from football_analytics.stages.base import Stage
from football_analytics.stages.mvp2_common import (
    canonical_common,
    read_required_parquet,
    video_frame_count,
    video_fps,
)
from football_analytics.utils.io import write_json, write_rows_with_schema


class PossessionStage(Stage):
    name = "possession"

    def validate_inputs(self) -> None:
        for name in ("tracks", "track_identities", "game_state", "ball_state"):
            read_required_parquet(self.run_dir / f"{name}.parquet")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        tracks = read_required_parquet(self.run_dir / "tracks.parquet")
        identities = read_required_parquet(self.run_dir / "track_identities.parquet")
        game = read_required_parquet(self.run_dir / "game_state.parquet")
        ball = read_required_parquet(self.run_dir / "ball_state.parquet").set_index(
            "frame_id"
        )
        identity_lookup = identities.set_index(["frame_id", "track_id"])
        game_lookup = game.set_index(["frame_id", "track_id"])
        grouped_tracks = {int(key): value for key, value in tracks.groupby("frame_id")}
        fps = video_fps(self.run_dir)
        cfg = self.config["possession"]
        tracker = PossessionTracker(
            PossessionConfig(
                control_radius_m=float(cfg["field_distance_m"]),
                control_radius_heights=float(cfg["normalized_pixel_distance"]),
                contest_ratio=float(cfg["contested_ratio"]),
                debounce_frames=max(
                    1, int(math.ceil(float(cfg["minimum_control_seconds"]) * fps))
                ),
                unknown_timeout_ms=float(cfg["unknown_timeout_seconds"]) * 1000.0,
                pass_speed_threshold=float(cfg["pass_minimum_speed_mps"]),
            )
        )
        rows: list[dict[str, Any]] = []
        frames = video_frame_count(self.run_dir)
        for frame_id in range(frames):
            players: list[PlayerSnapshot] = []
            frame_tracks = grouped_tracks.get(frame_id)
            if frame_tracks is not None:
                for item in frame_tracks.itertuples():
                    key = (frame_id, int(item.track_id))
                    identity = (
                        identity_lookup.loc[key] if key in identity_lookup.index else None
                    )
                    team_raw = identity["team_id"] if identity is not None else None
                    if team_raw not in ("team_0", "team_1"):
                        continue
                    field = game_lookup.loc[key] if key in game_lookup.index else None
                    players.append(
                        PlayerSnapshot(
                            track_id=int(item.track_id),
                            team_id=0 if team_raw == "team_0" else 1,
                            foot_x=float(item.foot_x_pixel),
                            foot_y=float(item.foot_y_pixel),
                            bbox_height=max(
                                1.0, float(item.bbox_y2) - float(item.bbox_y1)
                            ),
                            x_field=(
                                float(field["x_field"])
                                if field is not None
                                and bool(field["valid"])
                                and pd.notna(field["x_field"])
                                else None
                            ),
                            y_field=(
                                float(field["y_field"])
                                if field is not None
                                and bool(field["valid"])
                                and pd.notna(field["y_field"])
                                else None
                            ),
                        )
                    )
            ball_row = ball.loc[frame_id] if frame_id in ball.index else None
            snapshot = None
            ball_confidence = 0.0
            if ball_row is not None and bool(ball_row["valid"]):
                ball_confidence = float(ball_row["trajectory_confidence"] or 0.0)
                snapshot = BallSnapshot(
                    x_pixel=(
                        float(ball_row["ball_x_pixel"])
                        if pd.notna(ball_row["ball_x_pixel"])
                        else None
                    ),
                    y_pixel=(
                        float(ball_row["ball_y_pixel"])
                        if pd.notna(ball_row["ball_y_pixel"])
                        else None
                    ),
                    x_field=(
                        float(ball_row["ball_x_field"])
                        if pd.notna(ball_row["ball_x_field"])
                        else None
                    ),
                    y_field=(
                        float(ball_row["ball_y_field"])
                        if pd.notna(ball_row["ball_y_field"])
                        else None
                    ),
                    in_play=ball_row["visibility_state"] != "out_of_frame",
                )
            timestamp_ms = frame_id * 1000.0 / fps
            result = tracker.step(frame_id, timestamp_ms, snapshot, players)
            known = result.state.value != "unknown"
            confidence = ball_confidence * (0.8 if result.owner_track_id else 0.6)
            rows.append(
                {
                    **canonical_common(
                        self.run_dir,
                        frame_id,
                        timestamp_ms,
                        "field_or_bbox_height_normalized_state_machine",
                        confidence,
                        known,
                    ),
                    "owner_track_id": result.owner_track_id,
                    "owner_team_id": (
                        f"team_{result.owner_team_id}"
                        if result.owner_team_id is not None
                        else None
                    ),
                    "possession_state": result.state.value,
                    "transition_reason": result.transition_reason,
                }
            )
        timeline = self.run_dir / "possession_timeline.parquet"
        write_rows_with_schema(timeline, rows, POSSESSION_TIMELINE_SCHEMA)
        frame = pd.DataFrame(rows)
        counts = frame["possession_state"].value_counts().to_dict()
        total = max(1, len(frame))
        summary = {
            "run_id": self.run_dir.name,
            "total_frames": len(frame),
            "state_frames": {str(key): int(value) for key, value in counts.items()},
            "state_percent": {
                str(key): float(value / total) for key, value in counts.items()
            },
            "known_coverage": float(
                (~frame["possession_state"].isin(["unknown", "loose_ball"])).mean()
            ),
            "method": "field_or_bbox_height_normalized_state_machine",
        }
        summary_path = self.run_dir / "team_possession_summary.json"
        write_json(summary_path, summary)
        return {
            "possession_timeline": timeline,
            "team_possession_summary": summary_path,
        }

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        frame = pd.read_parquet(artifacts["possession_timeline"])
        validate_mvp2_columns("possession_timeline", list(frame.columns))
        if frame.empty:
            raise RuntimeError("possession stage produced zero rows")
