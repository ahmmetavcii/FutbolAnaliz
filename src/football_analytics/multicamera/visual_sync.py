"""Visual synchronization helpers built on pre-extracted visual signals.

No frame decoding or model inference happens here. Callers supply either:

- matched event timestamps (the same visible instants, e.g. kickoff whistle
  flash or scoreboard change, located independently in each recording), or
- scalar activity signals sampled per frame (e.g. mean frame brightness),
  which are aligned by cross-correlation.

The offset convention matches the rest of the package:
``reference_time = local_time + offset_seconds``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .audio_sync import AudioSyncConfig, cross_correlate_offset
from .synchronization import OffsetEstimate, OffsetSource

_EPSILON = 1e-12


@dataclass(frozen=True)
class MatchedEventPair:
    """One visual instant located in both recordings (seconds, local clocks)."""

    reference_time_seconds: float
    camera_time_seconds: float


@dataclass(frozen=True)
class VisualSyncResult:
    camera_id: str
    offset_seconds: float
    confidence: float
    drift_seconds_per_second: float
    residual_seconds: float

    def to_offset_estimate(self) -> OffsetEstimate:
        return OffsetEstimate(
            camera_id=self.camera_id,
            offset_seconds=self.offset_seconds,
            confidence=self.confidence,
            source=OffsetSource.VISUAL,
            measured_at_seconds=0.0,
        )


def offset_from_matched_events(
    camera_id: str,
    pairs: Sequence[MatchedEventPair],
    max_residual_seconds: float = 0.5,
) -> VisualSyncResult:
    """Fit ``reference = camera + offset (+ drift * camera)`` to matched events.

    With one pair only the constant offset is estimated. With two or more, a
    linear drift term is also fit. Confidence decays with the RMS residual of
    the fit relative to ``max_residual_seconds``.
    """
    if not pairs:
        raise ValueError("at least one matched event pair is required")
    camera_times = np.asarray([p.camera_time_seconds for p in pairs], dtype=np.float64)
    reference_times = np.asarray([p.reference_time_seconds for p in pairs], dtype=np.float64)
    deltas = reference_times - camera_times

    drift = 0.0
    if len(pairs) >= 2 and float(np.ptp(camera_times)) > _EPSILON:
        centered = camera_times - camera_times.mean()
        drift = float(np.sum(centered * (deltas - deltas.mean())) / np.sum(centered * centered))
    offset = float(deltas.mean() - drift * camera_times.mean())

    predicted = offset + drift * camera_times
    residual = float(np.sqrt(np.mean((deltas - predicted) ** 2)))
    confidence = float(np.clip(1.0 - residual / max(max_residual_seconds, _EPSILON), 0.0, 1.0))
    return VisualSyncResult(
        camera_id=camera_id,
        offset_seconds=offset,
        confidence=confidence,
        drift_seconds_per_second=drift,
        residual_seconds=residual,
    )


def offset_from_activity_signals(
    camera_id: str,
    reference_signal: Sequence[float] | NDArray[np.floating],
    camera_signal: Sequence[float] | NDArray[np.floating],
    fps: float,
    max_offset_seconds: float = 30.0,
) -> VisualSyncResult:
    """Align two per-frame scalar activity signals by cross-correlation."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    config = AudioSyncConfig(sample_rate_hz=fps, max_offset_seconds=max_offset_seconds)
    offset, confidence = cross_correlate_offset(reference_signal, camera_signal, config)
    return VisualSyncResult(
        camera_id=camera_id,
        offset_seconds=offset,
        confidence=confidence,
        drift_seconds_per_second=0.0,
        residual_seconds=0.0,
    )
