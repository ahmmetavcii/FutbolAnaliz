"""Distance covered per track, gated on calibration validity.

Reuses :mod:`football_analytics.analytics.player_metrics` for the actual
integration (timestamp-driven, plausibility-gated). The contract enforced
here: when calibration is invalid (or too few calibrated samples survive the
gates) the result is ``None`` — never zero, never a guess. Replay samples are
excluded before computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from football_analytics.analytics.player_metrics import (
    PlayerMetrics,
    PlayerMetricsConfig,
    PlayerSample,
    compute_player_metrics,
)


@dataclass(frozen=True)
class DistanceResult:
    track_id: int
    total_distance_m: float | None
    invalid_reason: str | None
    used_sample_count: int


def filter_replay(
    samples: Iterable[PlayerSample], replay_flags: Sequence[bool] | None = None
) -> list[PlayerSample]:
    """Drop samples captured during replay footage.

    ``PlayerSample`` has no replay field; callers either pre-filter or pass a
    parallel ``replay_flags`` sequence.
    """
    materialized = list(samples)
    if replay_flags is None:
        return materialized
    if len(replay_flags) != len(materialized):
        raise ValueError("replay_flags must align with samples")
    return [s for s, replay in zip(materialized, replay_flags) if not replay]


def compute_distances(
    samples: Iterable[PlayerSample],
    config: PlayerMetricsConfig | None = None,
    *,
    replay_flags: Sequence[bool] | None = None,
) -> dict[int, DistanceResult]:
    """Total distance per track; ``None`` when metrics could not be computed."""
    usable = filter_replay(samples, replay_flags)
    metrics = compute_player_metrics(usable, config)
    return {track_id: _to_result(m) for track_id, m in metrics.items()}


def _to_result(metrics: PlayerMetrics) -> DistanceResult:
    if not metrics.valid:
        return DistanceResult(
            track_id=metrics.track_id,
            total_distance_m=None,
            invalid_reason=metrics.invalid_reason,
            used_sample_count=metrics.used_sample_count,
        )
    return DistanceResult(
        track_id=metrics.track_id,
        total_distance_m=metrics.total_distance_m,
        invalid_reason=None,
        used_sample_count=metrics.used_sample_count,
    )
