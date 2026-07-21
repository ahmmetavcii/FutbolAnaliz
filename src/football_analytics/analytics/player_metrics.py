"""Per-player physical metrics from calibrated track samples.

Principles:

- Timestamp-driven: all rates are computed from ``timestamp_ms`` deltas, so
  24 / 25 / 30 fps (or variable frame rate) inputs produce equivalent metric
  values for the same physical motion. Frame counts are never used as a proxy
  for time.
- Gating: samples are usable only when calibration is valid, the shot type is
  accepted (default: only wide/tactical shots), and the track-quality flag is
  set. Gated-out samples contribute nothing, not even zeros.
- Physical plausibility: segment speeds above ``max_speed_kmh`` and implied
  accelerations above ``max_acceleration_ms2`` are rejected, and a segment
  spanning longer than ``max_segment_gap_ms`` is treated as a discontinuity
  (no distance credited across it).
- No identity stitching: each ``track_id`` is summarized independently. Two
  track ids that belong to the same human are still reported separately.

Outputs include cumulative distance, smoothed and top speed, sprint episodes,
mean sample confidence, and temporal coverage of the observed span.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

_KMH_PER_MS = 3.6
_MIN_DT_S = 1e-3


@dataclass(frozen=True)
class PlayerSample:
    """One calibrated observation of a player track."""

    track_id: int
    timestamp_ms: float
    x_field: float | None
    y_field: float | None
    confidence: float = 1.0
    calibration_valid: bool = True
    shot_type: str = "wide"
    quality_ok: bool = True


@dataclass(frozen=True)
class PlayerMetricsConfig:
    max_speed_kmh: float = 40.0
    max_acceleration_ms2: float = 8.0
    max_segment_gap_ms: float = 1500.0
    min_confidence: float = 0.2
    accepted_shot_types: frozenset[str] = frozenset({"wide", "tactical"})
    smoothing_window_s: float = 1.0
    sprint_speed_kmh: float = 25.0
    sprint_min_duration_s: float = 1.0
    min_samples: int = 3

    def __post_init__(self) -> None:
        if self.smoothing_window_s <= 0.0:
            raise ValueError("smoothing_window_s must be positive")
        if self.min_samples < 2:
            raise ValueError("min_samples must be >= 2")


@dataclass(frozen=True)
class SprintEpisode:
    start_ms: float
    end_ms: float
    peak_speed_kmh: float

    @property
    def duration_s(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0


@dataclass(frozen=True)
class PlayerMetrics:
    track_id: int
    valid: bool
    invalid_reason: str | None
    total_distance_m: float = 0.0
    max_speed_kmh: float = 0.0
    mean_speed_kmh: float = 0.0
    sprints: tuple[SprintEpisode, ...] = ()
    sample_count: int = 0
    used_sample_count: int = 0
    coverage: float = 0.0
    mean_confidence: float = 0.0
    speed_series: tuple[tuple[float, float], ...] = field(default=(), repr=False)


def _sample_usable(sample: PlayerSample, config: PlayerMetricsConfig) -> bool:
    return (
        sample.x_field is not None
        and sample.y_field is not None
        and sample.calibration_valid
        and sample.quality_ok
        and sample.shot_type in config.accepted_shot_types
        and sample.confidence >= config.min_confidence
    )


def _smooth_speeds(
    speeds: Sequence[tuple[float, float]], window_s: float
) -> list[tuple[float, float]]:
    """Centered time-window moving average over (timestamp_ms, speed_kmh)."""
    half_ms = window_s * 500.0
    smoothed: list[tuple[float, float]] = []
    for i, (ts, _) in enumerate(speeds):
        acc = 0.0
        count = 0
        for j in range(i, -1, -1):
            if ts - speeds[j][0] > half_ms:
                break
            acc += speeds[j][1]
            count += 1
        for j in range(i + 1, len(speeds)):
            if speeds[j][0] - ts > half_ms:
                break
            acc += speeds[j][1]
            count += 1
        smoothed.append((ts, acc / count))
    return smoothed


def _detect_sprints(
    speeds: Sequence[tuple[float, float]], config: PlayerMetricsConfig
) -> list[SprintEpisode]:
    episodes: list[SprintEpisode] = []
    start: float | None = None
    end: float | None = None
    peak = 0.0
    for ts, speed in speeds:
        if speed >= config.sprint_speed_kmh:
            if start is None:
                start = ts
            end = ts
            peak = max(peak, speed)
        elif start is not None:
            assert end is not None
            if (end - start) / 1000.0 >= config.sprint_min_duration_s:
                episodes.append(SprintEpisode(start, end, peak))
            start, end, peak = None, None, 0.0
    if start is not None and end is not None:
        if (end - start) / 1000.0 >= config.sprint_min_duration_s:
            episodes.append(SprintEpisode(start, end, peak))
    return episodes


def compute_player_metrics(
    samples: Iterable[PlayerSample],
    config: PlayerMetricsConfig | None = None,
) -> dict[int, PlayerMetrics]:
    """Summarize each track id independently (no cross-track stitching)."""
    cfg = config or PlayerMetricsConfig()
    by_track: dict[int, list[PlayerSample]] = {}
    for sample in samples:
        by_track.setdefault(sample.track_id, []).append(sample)

    results: dict[int, PlayerMetrics] = {}
    for track_id, track_samples in by_track.items():
        results[track_id] = _metrics_for_track(track_id, track_samples, cfg)
    return results


def _metrics_for_track(
    track_id: int, track_samples: list[PlayerSample], cfg: PlayerMetricsConfig
) -> PlayerMetrics:
    ordered = sorted(track_samples, key=lambda s: s.timestamp_ms)
    usable = [s for s in ordered if _sample_usable(s, cfg)]
    if len(usable) < cfg.min_samples:
        return PlayerMetrics(
            track_id=track_id,
            valid=False,
            invalid_reason="insufficient_usable_samples",
            sample_count=len(ordered),
            used_sample_count=len(usable),
        )

    total_distance = 0.0
    raw_speeds: list[tuple[float, float]] = []
    prev = usable[0]
    prev_speed_ms: float | None = None
    used = 1
    for sample in usable[1:]:
        dt_ms = sample.timestamp_ms - prev.timestamp_ms
        if dt_ms <= 0.0:
            continue
        dt_s = max(dt_ms / 1000.0, _MIN_DT_S)
        step_m = math.hypot(
            sample.x_field - prev.x_field,  # type: ignore[operator]
            sample.y_field - prev.y_field,  # type: ignore[operator]
        )
        speed_ms = step_m / dt_s
        speed_kmh = speed_ms * _KMH_PER_MS
        if dt_ms > cfg.max_segment_gap_ms:
            # Discontinuity: anchor here without crediting distance or speed.
            prev = sample
            prev_speed_ms = None
            used += 1
            continue
        if speed_kmh > cfg.max_speed_kmh:
            # Physically impossible displacement: drop the sample entirely so
            # it cannot poison subsequent segments either.
            continue
        if prev_speed_ms is not None:
            accel = abs(speed_ms - prev_speed_ms) / dt_s
            if accel > cfg.max_acceleration_ms2:
                continue
        total_distance += step_m
        raw_speeds.append((sample.timestamp_ms, speed_kmh))
        prev = sample
        prev_speed_ms = speed_ms
        used += 1

    if not raw_speeds:
        return PlayerMetrics(
            track_id=track_id,
            valid=False,
            invalid_reason="no_valid_motion_segments",
            sample_count=len(ordered),
            used_sample_count=used,
        )

    smoothed = _smooth_speeds(raw_speeds, cfg.smoothing_window_s)
    speeds_only = [s for _, s in smoothed]
    # Reject isolated peaks for reported max: require min 3 consecutive
    # samples within 2 km/h of the peak, else use p95.
    arr = np.asarray(speeds_only, dtype=float) if speeds_only else np.asarray([])
    reported_max = 0.0
    if len(arr):
        p95 = float(np.nanpercentile(arr, 95))
        peak = float(np.nanmax(arr))
        consecutive_ok = False
        run = 0
        for v in arr:
            if abs(v - peak) <= 2.0:
                run += 1
                if run >= 3:
                    consecutive_ok = True
                    break
            else:
                run = 0
        reported_max = peak if consecutive_ok and peak <= cfg.max_speed_kmh else p95
        if reported_max > cfg.max_speed_kmh:
            reported_max = p95
    span_ms = ordered[-1].timestamp_ms - ordered[0].timestamp_ms
    coverage = 0.0
    if span_ms > 0.0:
        usable_span = usable[-1].timestamp_ms - usable[0].timestamp_ms
        coverage = min(usable_span / span_ms, 1.0)
    mean_conf = sum(s.confidence for s in usable) / len(usable)

    return PlayerMetrics(
        track_id=track_id,
        valid=True,
        invalid_reason=None,
        total_distance_m=total_distance,
        max_speed_kmh=reported_max if speeds_only else 0.0,
        mean_speed_kmh=sum(speeds_only) / len(speeds_only),
        sprints=tuple(_detect_sprints(smoothed, cfg)),
        sample_count=len(ordered),
        used_sample_count=used,
        coverage=coverage,
        mean_confidence=mean_conf,
        speed_series=tuple(smoothed),
    )
