"""Speed and sprint summaries per track, gated on calibration validity.

Reuses :mod:`football_analytics.analytics.player_metrics` (smoothed
timestamp-driven speeds, plausibility gates, sprint episodes). Invalid
calibration or insufficient usable samples yields ``None`` speeds — physical
metrics are never fabricated from uncalibrated pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from football_analytics.analytics.distance import filter_replay
from football_analytics.analytics.player_metrics import (
    PlayerMetrics,
    PlayerMetricsConfig,
    PlayerSample,
    SprintEpisode,
    compute_player_metrics,
)


@dataclass(frozen=True)
class SpeedResult:
    track_id: int
    max_speed_kmh: float | None
    mean_speed_kmh: float | None
    sprint_count: int
    sprints: tuple[SprintEpisode, ...]
    invalid_reason: str | None


def compute_speeds(
    samples: Iterable[PlayerSample],
    config: PlayerMetricsConfig | None = None,
    *,
    replay_flags: Sequence[bool] | None = None,
) -> dict[int, SpeedResult]:
    usable = filter_replay(samples, replay_flags)
    metrics = compute_player_metrics(usable, config)
    return {track_id: _to_result(m) for track_id, m in metrics.items()}


def _to_result(metrics: PlayerMetrics) -> SpeedResult:
    if not metrics.valid:
        return SpeedResult(
            track_id=metrics.track_id,
            max_speed_kmh=None,
            mean_speed_kmh=None,
            sprint_count=0,
            sprints=(),
            invalid_reason=metrics.invalid_reason,
        )
    return SpeedResult(
        track_id=metrics.track_id,
        max_speed_kmh=metrics.max_speed_kmh,
        mean_speed_kmh=metrics.mean_speed_kmh,
        sprint_count=len(metrics.sprints),
        sprints=metrics.sprints,
        invalid_reason=None,
    )
