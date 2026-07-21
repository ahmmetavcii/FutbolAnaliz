"""Ball trajectory estimation with gated association and bounded gap prediction.

Design goals:

- Explicit visibility states: ``detected``, ``predicted``, ``occluded_short``,
  ``airborne``, ``out_of_frame``, ``unknown``.
- Association gates: detection confidence, apparent size, field bounds, implied
  speed, and implied acceleration. Candidates failing a gate are never matched.
- Prediction across detection gaps is bounded both in wall-clock time
  (``max_gap_ms``) and in frames (``max_gap_frames``). A long loss yields a
  null position and ``unknown`` state; positions are never backfilled and a
  reappearing blob is never force-matched to the stale motion state.
- Scene cuts reset all motion state so the next detection is accepted fresh
  without motion gating against pre-cut positions.
- Motion model is selectable: exponential-blend constant velocity, or a simple
  per-axis (position, velocity) Kalman filter.

Coordinates are unit-agnostic (pixels or metres, as long as the configuration
thresholds use the same units). Timestamps are milliseconds.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Sequence

_MIN_DT_S = 1e-3


class BallState(str, Enum):
    DETECTED = "detected"
    PREDICTED = "predicted"
    OCCLUDED_SHORT = "occluded_short"
    AIRBORNE = "airborne"
    OUT_OF_FRAME = "out_of_frame"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned rectangle in trajectory coordinate units."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            self.x_min - margin <= x <= self.x_max + margin
            and self.y_min - margin <= y <= self.y_max + margin
        )


@dataclass(frozen=True)
class BallObservation:
    """A single ball detection candidate for one frame."""

    x: float
    y: float
    confidence: float
    width: float | None = None
    height: float | None = None

    @property
    def size(self) -> float | None:
        if self.width is None or self.height is None:
            return None
        if self.width <= 0.0 or self.height <= 0.0:
            return None
        return math.sqrt(self.width * self.height)


@dataclass(frozen=True)
class BallTrackerConfig:
    """Configuration for :class:`BallTrajectoryEstimator`.

    ``max_speed`` / ``max_acceleration`` are expressed in coordinate units per
    second (and per second squared) so the estimator works for pixel or metric
    coordinate spaces alike.
    """

    max_gap_ms: float = 700.0
    max_gap_frames: int = 20
    short_occlusion_frames: int = 3
    max_speed: float = 3000.0
    max_acceleration: float = 40000.0
    min_confidence: float = 0.1
    min_size: float | None = None
    max_size: float | None = None
    field_bounds: Bounds | None = None
    field_bounds_margin: float = 0.0
    frame_bounds: Bounds | None = None
    prediction_decay: float = 0.85
    airborne_size_ratio: float = 1.6
    size_history: int = 12
    filter_type: str = "constant_velocity"
    velocity_blend: float = 0.6
    kalman_process_noise: float = 200.0
    kalman_measurement_noise: float = 4.0

    def __post_init__(self) -> None:
        if self.filter_type not in ("constant_velocity", "kalman"):
            raise ValueError(f"unsupported filter_type: {self.filter_type!r}")
        if self.max_gap_ms <= 0.0 or self.max_gap_frames <= 0:
            raise ValueError("gap bounds must be positive")


@dataclass(frozen=True)
class BallEstimate:
    """Per-frame estimator output. ``x``/``y`` are ``None`` when unknown."""

    frame_id: int
    timestamp_ms: float
    x: float | None
    y: float | None
    state: BallState
    confidence: float
    velocity_x: float | None = None
    velocity_y: float | None = None


class _KalmanAxis:
    """Minimal 1D (position, velocity) Kalman filter with white-accel noise."""

    def __init__(self, position: float, process_noise: float, measurement_noise: float) -> None:
        self.x = position
        self.v = 0.0
        self._q = process_noise
        self._r = measurement_noise
        self.p11 = measurement_noise
        self.p12 = 0.0
        self.p22 = 1e4

    def predict(self, dt_s: float) -> None:
        self.x += self.v * dt_s
        q = self._q
        p11, p12, p22 = self.p11, self.p12, self.p22
        self.p11 = p11 + 2.0 * dt_s * p12 + dt_s * dt_s * p22 + q * dt_s**4 / 4.0
        self.p12 = p12 + dt_s * p22 + q * dt_s**3 / 2.0
        self.p22 = p22 + q * dt_s * dt_s

    def update(self, z: float) -> None:
        s = self.p11 + self._r
        k1 = self.p11 / s
        k2 = self.p12 / s
        innovation = z - self.x
        self.x += k1 * innovation
        self.v += k2 * innovation
        p11, p12, p22 = self.p11, self.p12, self.p22
        self.p11 = (1.0 - k1) * p11
        self.p12 = (1.0 - k1) * p12
        self.p22 = p22 - k2 * p12


class BallTrajectoryEstimator:
    """Streaming single-ball trajectory estimator.

    Call :meth:`step` once per frame with all ball detection candidates for
    that frame (possibly none). Outputs are emitted immediately and never
    revised afterwards (no backfilling).
    """

    def __init__(self, config: BallTrackerConfig | None = None) -> None:
        self._config = config or BallTrackerConfig()
        self._sizes: Deque[float] = deque(maxlen=self._config.size_history)
        self._reset_motion()

    def _reset_motion(self) -> None:
        self._last_x: float | None = None
        self._last_y: float | None = None
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._has_velocity = False
        self._last_ts: float | None = None
        self._last_frame: int | None = None
        self._last_confidence = 0.0
        self._kalman_x: _KalmanAxis | None = None
        self._kalman_y: _KalmanAxis | None = None
        self._sizes.clear()

    def reset(self) -> None:
        """Discard all motion state (e.g. on a scene cut)."""
        self._reset_motion()

    # ------------------------------------------------------------------ gates

    def _passes_static_gates(self, obs: BallObservation) -> bool:
        cfg = self._config
        if obs.confidence < cfg.min_confidence:
            return False
        size = obs.size
        if size is not None:
            if cfg.min_size is not None and size < cfg.min_size:
                return False
            if cfg.max_size is not None and size > cfg.max_size:
                return False
        if cfg.field_bounds is not None and not cfg.field_bounds.contains(
            obs.x, obs.y, cfg.field_bounds_margin
        ):
            return False
        return True

    def _passes_motion_gates(self, obs: BallObservation, dt_s: float) -> bool:
        if self._last_x is None or self._last_y is None:
            return True
        dt_s = max(dt_s, _MIN_DT_S)
        vx = (obs.x - self._last_x) / dt_s
        vy = (obs.y - self._last_y) / dt_s
        speed = math.hypot(vx, vy)
        if speed > self._config.max_speed:
            return False
        if self._has_velocity:
            accel = math.hypot(vx - self._vel_x, vy - self._vel_y) / dt_s
            if accel > self._config.max_acceleration:
                return False
        return True

    def _select_candidate(
        self, candidates: Sequence[BallObservation], dt_s: float
    ) -> BallObservation | None:
        admissible = [
            obs
            for obs in candidates
            if self._passes_static_gates(obs) and self._passes_motion_gates(obs, dt_s)
        ]
        if not admissible:
            return None
        if self._last_x is None or self._last_y is None:
            return max(admissible, key=lambda obs: obs.confidence)
        last_x, last_y = self._last_x, self._last_y
        return min(admissible, key=lambda obs: math.hypot(obs.x - last_x, obs.y - last_y))

    # ------------------------------------------------------------------ state

    def _classify_detection(self, obs: BallObservation) -> BallState:
        size = obs.size
        state = BallState.DETECTED
        if size is not None and len(self._sizes) >= 3:
            ordered = sorted(self._sizes)
            median = ordered[len(ordered) // 2]
            if median > 0.0 and size / median >= self._config.airborne_size_ratio:
                state = BallState.AIRBORNE
        if size is not None and state is BallState.DETECTED:
            self._sizes.append(size)
        return state

    def _accept(
        self, frame_id: int, timestamp_ms: float, obs: BallObservation
    ) -> BallEstimate:
        cfg = self._config
        state = self._classify_detection(obs)
        if cfg.filter_type == "kalman":
            if self._kalman_x is None or self._kalman_y is None:
                self._kalman_x = _KalmanAxis(obs.x, cfg.kalman_process_noise, cfg.kalman_measurement_noise)
                self._kalman_y = _KalmanAxis(obs.y, cfg.kalman_process_noise, cfg.kalman_measurement_noise)
            elif self._last_ts is not None:
                dt_s = max((timestamp_ms - self._last_ts) / 1000.0, _MIN_DT_S)
                self._kalman_x.predict(dt_s)
                self._kalman_y.predict(dt_s)
                self._kalman_x.update(obs.x)
                self._kalman_y.update(obs.y)
            est_x, est_y = self._kalman_x.x, self._kalman_y.x
            self._vel_x, self._vel_y = self._kalman_x.v, self._kalman_y.v
            self._has_velocity = True
        else:
            est_x, est_y = obs.x, obs.y
            if self._last_x is not None and self._last_y is not None and self._last_ts is not None:
                dt_s = max((timestamp_ms - self._last_ts) / 1000.0, _MIN_DT_S)
                raw_vx = (obs.x - self._last_x) / dt_s
                raw_vy = (obs.y - self._last_y) / dt_s
                if self._has_velocity:
                    blend = cfg.velocity_blend
                    self._vel_x = blend * raw_vx + (1.0 - blend) * self._vel_x
                    self._vel_y = blend * raw_vy + (1.0 - blend) * self._vel_y
                else:
                    self._vel_x, self._vel_y = raw_vx, raw_vy
                self._has_velocity = True
        self._last_x, self._last_y = est_x, est_y
        self._last_ts = timestamp_ms
        self._last_frame = frame_id
        self._last_confidence = obs.confidence
        return BallEstimate(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            x=est_x,
            y=est_y,
            state=state,
            confidence=obs.confidence,
            velocity_x=self._vel_x if self._has_velocity else None,
            velocity_y=self._vel_y if self._has_velocity else None,
        )

    def _unknown(self, frame_id: int, timestamp_ms: float) -> BallEstimate:
        return BallEstimate(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            x=None,
            y=None,
            state=BallState.UNKNOWN,
            confidence=0.0,
        )

    def _predict(self, frame_id: int, timestamp_ms: float) -> BallEstimate:
        cfg = self._config
        assert self._last_ts is not None and self._last_frame is not None
        assert self._last_x is not None and self._last_y is not None
        gap_ms = timestamp_ms - self._last_ts
        gap_frames = frame_id - self._last_frame
        if gap_ms > cfg.max_gap_ms or gap_frames > cfg.max_gap_frames:
            # Long loss: null position, fresh acquisition required afterwards.
            self._reset_motion()
            return self._unknown(frame_id, timestamp_ms)
        dt_s = max(gap_ms / 1000.0, 0.0)
        pred_x = self._last_x + (self._vel_x if self._has_velocity else 0.0) * dt_s
        pred_y = self._last_y + (self._vel_y if self._has_velocity else 0.0) * dt_s
        confidence = self._last_confidence * cfg.prediction_decay**max(gap_frames, 1)
        if cfg.frame_bounds is not None and not cfg.frame_bounds.contains(pred_x, pred_y):
            state = BallState.OUT_OF_FRAME
        elif gap_frames <= cfg.short_occlusion_frames:
            state = BallState.OCCLUDED_SHORT
        else:
            state = BallState.PREDICTED
        return BallEstimate(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            x=pred_x,
            y=pred_y,
            state=state,
            confidence=confidence,
            velocity_x=self._vel_x if self._has_velocity else None,
            velocity_y=self._vel_y if self._has_velocity else None,
        )

    # ------------------------------------------------------------------- step

    def step(
        self,
        frame_id: int,
        timestamp_ms: float,
        candidates: Sequence[BallObservation] = (),
        scene_cut: bool = False,
    ) -> BallEstimate:
        """Process one frame and return the current ball estimate."""
        if scene_cut:
            self._reset_motion()
        dt_s = 0.0
        if self._last_ts is not None:
            dt_s = max((timestamp_ms - self._last_ts) / 1000.0, _MIN_DT_S)
        selected = self._select_candidate(candidates, dt_s)
        if selected is not None:
            return self._accept(frame_id, timestamp_ms, selected)
        if self._last_ts is None:
            return self._unknown(frame_id, timestamp_ms)
        return self._predict(frame_id, timestamp_ms)
