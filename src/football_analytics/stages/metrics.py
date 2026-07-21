"""Timestamp-driven player and team spatial metrics stage."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from football_analytics.analytics.player_metrics import (
    PlayerMetricsConfig,
    PlayerSample,
    compute_player_metrics,
)
from football_analytics.analytics.team_metrics import (
    TeamMetricsConfig,
    TeamPlayerSample,
    compute_team_metrics,
)
from football_analytics.contracts.schemas import (
    PLAYER_METRICS_SCHEMA,
    TEAM_METRICS_SCHEMA,
    validate_mvp2_columns,
)
from football_analytics.stages.base import Stage
from football_analytics.stages.mvp2_common import (
    canonical_common,
    read_required_parquet,
)
from football_analytics.utils.io import write_rows_with_schema


class MetricsStage(Stage):
    name = "metrics"

    def validate_inputs(self) -> None:
        for name in ("game_state", "track_quality", "track_identities"):
            read_required_parquet(self.run_dir / f"{name}.parquet")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        game = read_required_parquet(self.run_dir / "game_state.parquet")
        quality = read_required_parquet(self.run_dir / "track_quality.parquet").set_index(
            "track_id"
        )
        identities = read_required_parquet(
            self.run_dir / "track_identities.parquet"
        ).set_index(["frame_id", "track_id"])
        player_cfg = self.config["player_metrics"]
        samples: list[PlayerSample] = []
        for item in game.itertuples():
            quality_ok = (
                bool(quality.loc[int(item.track_id), "usable_for_metrics"])
                if int(item.track_id) in quality.index
                else False
            )
            samples.append(
                PlayerSample(
                    track_id=int(item.track_id),
                    timestamp_ms=float(item.timestamp_ms),
                    x_field=float(item.x_field) if pd.notna(item.x_field) else None,
                    y_field=float(item.y_field) if pd.notna(item.y_field) else None,
                    confidence=float(item.confidence),
                    calibration_valid=bool(item.valid),
                    shot_type=str(item.shot_type),
                    quality_ok=quality_ok,
                )
            )
        summaries = compute_player_metrics(
            samples,
            PlayerMetricsConfig(
                max_speed_kmh=float(player_cfg["maximum_plausible_speed_kmh"]),
                max_acceleration_ms2=float(
                    player_cfg["maximum_plausible_acceleration_mps2"]
                ),
                max_segment_gap_ms=float(player_cfg["maximum_gap_seconds"]) * 1000.0,
                min_confidence=float(
                    player_cfg["minimum_calibration_confidence"]
                ),
                accepted_shot_types=frozenset(player_cfg["valid_shot_types"]),
                smoothing_window_s=float(player_cfg["smoothing_window_seconds"]),
                sprint_speed_kmh=float(player_cfg["sprint_speed_kmh"]),
                sprint_min_duration_s=float(player_cfg["minimum_sprint_seconds"]),
                min_samples=3,
            ),
        )
        player_rows = self._player_rows(game, summaries)
        team_rows = self._team_rows(game, identities)
        player_path = self.run_dir / "player_metrics.parquet"
        team_path = self.run_dir / "team_metrics.parquet"
        write_rows_with_schema(player_path, player_rows, PLAYER_METRICS_SCHEMA)
        write_rows_with_schema(team_path, team_rows, TEAM_METRICS_SCHEMA)
        return {"player_metrics": player_path, "team_metrics": team_path}

    def _player_rows(self, game: pd.DataFrame, summaries: dict[int, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        cumulative: dict[int, float] = defaultdict(float)
        previous: dict[int, tuple[float, float, float]] = {}
        speed_lookup = {
            (track_id, round(timestamp, 3)): speed
            for track_id, summary in summaries.items()
            for timestamp, speed in summary.speed_series
        }
        sprint_ranges = {
            track_id: [(item.start_ms, item.end_ms) for item in summary.sprints]
            for track_id, summary in summaries.items()
        }
        for item in game.sort_values(["track_id", "timestamp_ms"]).itertuples():
            track_id = int(item.track_id)
            summary = summaries.get(track_id)
            speed = speed_lookup.get((track_id, round(float(item.timestamp_ms), 3)))
            valid = bool(summary and summary.valid and speed is not None and item.valid)
            instantaneous = None
            if valid and track_id in previous:
                prev_ts, prev_x, prev_y = previous[track_id]
                dt = (float(item.timestamp_ms) - prev_ts) / 1000.0
                if dt > 0:
                    step = float(
                        np.hypot(float(item.x_field) - prev_x, float(item.y_field) - prev_y)
                    )
                    instantaneous = step / dt * 3.6
                    if instantaneous <= float(
                        self.config["player_metrics"]["maximum_plausible_speed_kmh"]
                    ):
                        cumulative[track_id] += step
                    else:
                        valid = False
                        instantaneous = None
            if item.valid and pd.notna(item.x_field) and pd.notna(item.y_field):
                previous[track_id] = (
                    float(item.timestamp_ms),
                    float(item.x_field),
                    float(item.y_field),
                )
            sprint = (
                "sprint"
                if any(
                    start <= float(item.timestamp_ms) <= end
                    for start, end in sprint_ranges.get(track_id, [])
                )
                else "not_sprint"
            )
            confidence = float(summary.mean_confidence) if valid else 0.0
            rows.append(
                {
                    **canonical_common(
                        self.run_dir,
                        int(item.frame_id),
                        float(item.timestamp_ms),
                        "timestamp_calibrated_track_motion",
                        confidence,
                        valid,
                    ),
                    "track_id": track_id,
                    "x_field": float(item.x_field) if valid else None,
                    "y_field": float(item.y_field) if valid else None,
                    "instantaneous_speed_kmh": instantaneous if valid else None,
                    "smoothed_speed_kmh": float(speed) if valid else None,
                    "cumulative_distance_m": cumulative[track_id] if valid else None,
                    "sprint_state": sprint if valid else None,
                    "coverage": float(summary.coverage) if summary else 0.0,
                    "invalid_reason": (
                        None
                        if valid
                        else (
                            summary.invalid_reason
                            if summary
                            else (item.invalid_reason or "track_quality_gate_failed")
                        )
                    ),
                }
            )
        return rows

    def _team_rows(
        self, game: pd.DataFrame, identities: pd.DataFrame
    ) -> list[dict[str, Any]]:
        cfg = self.config["team_metrics"]
        rows: list[dict[str, Any]] = []
        for frame_id, group in game.groupby("frame_id", sort=True):
            samples: list[TeamPlayerSample] = []
            timestamps = group["timestamp_ms"]
            for item in group.itertuples():
                key = (int(frame_id), int(item.track_id))
                identity = identities.loc[key] if key in identities.index else None
                team = identity["team_id"] if identity is not None else None
                if team not in ("team_0", "team_1"):
                    continue
                samples.append(
                    TeamPlayerSample(
                        track_id=int(item.track_id),
                        team_id=0 if team == "team_0" else 1,
                        x_field=float(item.x_field) if pd.notna(item.x_field) else None,
                        y_field=float(item.y_field) if pd.notna(item.y_field) else None,
                        team_confidence=float(identity["team_confidence"]),
                        calibration_valid=bool(item.valid),
                    )
                )
            computed = compute_team_metrics(
                samples,
                TeamMetricsConfig(
                    min_team_confidence=float(cfg["minimum_team_confidence"]),
                    min_players=int(cfg["minimum_players"]),
                ),
            )
            for team_id in (0, 1):
                metric = computed.get(team_id)
                valid = bool(metric and metric.valid)
                players = [sample for sample in samples if sample.team_id == team_id]
                last_line = (
                    max(float(sample.x_field) for sample in players if sample.x_field is not None)
                    if valid
                    else None
                )
                occupancy = self._occupancy(players) if valid else None
                confidence = (
                    float(np.mean([sample.team_confidence for sample in players]))
                    if valid
                    else 0.0
                )
                rows.append(
                    {
                        **canonical_common(
                            self.run_dir,
                            int(frame_id),
                            float(timestamps.iloc[0]),
                            "confident_calibrated_team_shape",
                            confidence,
                            valid,
                        ),
                        "team_id": f"team_{team_id}",
                        "centroid_x": metric.centroid_x if valid else None,
                        "centroid_y": metric.centroid_y if valid else None,
                        "width_m": metric.width_m if valid else None,
                        "depth_m": metric.depth_m if valid else None,
                        "mean_interplayer_distance_m": (
                            metric.mean_interplayer_distance_m if valid else None
                        ),
                        "compactness_m": metric.compactness_m if valid else None,
                        "last_line_height_m": last_line,
                        "player_count": metric.player_count if metric else len(players),
                        "player_coverage": min(
                            1.0,
                            len(players) / max(1, int(cfg["minimum_players"])),
                        ),
                        "regional_occupancy_json": (
                            json.dumps(occupancy) if occupancy is not None else None
                        ),
                        "invalid_reason": (
                            None
                            if valid
                            else (
                                metric.invalid_reason
                                if metric
                                else "no_confident_team_players"
                            )
                        ),
                    }
                )
        return rows

    def _occupancy(self, players: list[TeamPlayerSample]) -> list[list[int]]:
        nx = int(self.config["team_metrics"]["occupancy_grid_x"])
        ny = int(self.config["team_metrics"]["occupancy_grid_y"])
        grid = np.zeros((ny, nx), dtype=int)
        length = float(self.config["calibration"]["pitch_length_m"])
        width = float(self.config["calibration"]["pitch_width_m"])
        for item in players:
            if item.x_field is None or item.y_field is None:
                continue
            x = min(nx - 1, max(0, int(item.x_field / length * nx)))
            y = min(ny - 1, max(0, int(item.y_field / width * ny)))
            grid[y, x] += 1
        return grid.tolist()

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        player = pd.read_parquet(artifacts["player_metrics"])
        team = pd.read_parquet(artifacts["team_metrics"])
        validate_mvp2_columns("player_metrics", list(player.columns))
        validate_mvp2_columns("team_metrics", list(team.columns))
