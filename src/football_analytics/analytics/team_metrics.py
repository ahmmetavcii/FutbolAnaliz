"""Team shape metrics from confidently-identified, calibrated player positions.

Only players with valid field coordinates *and* a confident team identity
contribute; everything else is excluded rather than guessed. When fewer than
``min_players`` remain for a team, the frame result is marked invalid with a
reason instead of emitting misleading numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TeamPlayerSample:
    """One player observation considered for team-shape aggregation."""

    track_id: int
    team_id: int
    x_field: float | None
    y_field: float | None
    team_confidence: float = 1.0
    calibration_valid: bool = True


@dataclass(frozen=True)
class TeamMetricsConfig:
    min_team_confidence: float = 0.6
    min_players: int = 4

    def __post_init__(self) -> None:
        if self.min_players < 2:
            raise ValueError("min_players must be >= 2")


@dataclass(frozen=True)
class TeamMetrics:
    team_id: int
    valid: bool
    invalid_reason: str | None
    player_count: int = 0
    centroid_x: float | None = None
    centroid_y: float | None = None
    width_m: float | None = None
    depth_m: float | None = None
    mean_interplayer_distance_m: float | None = None
    compactness_m: float | None = None


def _usable(sample: TeamPlayerSample, config: TeamMetricsConfig) -> bool:
    return (
        sample.x_field is not None
        and sample.y_field is not None
        and math.isfinite(float(sample.x_field))
        and math.isfinite(float(sample.y_field))
        and sample.calibration_valid
        and sample.team_confidence >= config.min_team_confidence
    )


def _team_metrics(team_id: int, points: list[tuple[float, float]], cfg: TeamMetricsConfig) -> TeamMetrics:
    if len(points) < cfg.min_players:
        return TeamMetrics(
            team_id=team_id,
            valid=False,
            invalid_reason="insufficient_confident_players",
            player_count=len(points),
        )
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    centroid_x = sum(xs) / len(xs)
    centroid_y = sum(ys) / len(ys)
    pair_distances = [
        math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
        for i in range(len(points))
        for j in range(i + 1, len(points))
    ]
    spread = [math.hypot(x - centroid_x, y - centroid_y) for x, y in points]
    return TeamMetrics(
        team_id=team_id,
        valid=True,
        invalid_reason=None,
        player_count=len(points),
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        width_m=max(ys) - min(ys),
        depth_m=max(xs) - min(xs),
        mean_interplayer_distance_m=sum(pair_distances) / len(pair_distances),
        compactness_m=sum(spread) / len(spread),
    )


def compute_team_metrics(
    players: Sequence[TeamPlayerSample],
    config: TeamMetricsConfig | None = None,
) -> dict[int, TeamMetrics]:
    """Compute per-team shape metrics for one frame."""
    cfg = config or TeamMetricsConfig()
    grouped: dict[int, list[tuple[float, float]]] = {}
    seen_teams: set[int] = set()
    for sample in players:
        seen_teams.add(sample.team_id)
        if _usable(sample, cfg):
            grouped.setdefault(sample.team_id, []).append(
                (float(sample.x_field), float(sample.y_field))  # type: ignore[arg-type]
            )
    return {
        team_id: _team_metrics(team_id, grouped.get(team_id, []), cfg)
        for team_id in sorted(seen_teams)
    }
