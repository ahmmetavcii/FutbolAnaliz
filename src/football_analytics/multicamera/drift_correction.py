"""Clock drift modelling and correction for long recordings.

Consumer cameras drift by a few parts per million to parts per thousand;
over a 90+ minute match this can amount to several frames. This module fits a
robust linear drift model to repeated offset measurements and applies it to
timestamps, and can push the fitted model into a
:class:`~football_analytics.multicamera.synchronization.TimelineSynchronizer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .synchronization import OffsetEstimate, TimelineSynchronizer

_EPSILON = 1e-12


@dataclass(frozen=True)
class DriftModel:
    """Linear model ``offset(t) = base_offset + rate * (t - anchor)``."""

    camera_id: str
    base_offset_seconds: float
    drift_rate_seconds_per_second: float
    anchor_seconds: float
    residual_seconds: float
    sample_count: int

    def offset_at(self, local_time_seconds: float) -> float:
        return self.base_offset_seconds + self.drift_rate_seconds_per_second * (
            local_time_seconds - self.anchor_seconds
        )

    def correct(self, local_time_seconds: float) -> float:
        """Map a local timestamp onto the reference timeline."""
        return local_time_seconds + self.offset_at(local_time_seconds)


def fit_drift_model(
    camera_id: str,
    measurements: Sequence[OffsetEstimate],
    minimum_confidence: float = 0.0,
) -> DriftModel:
    """Weighted least-squares linear fit over timed offset measurements.

    Measurements without ``measured_at_seconds`` or for other cameras are
    ignored. At least one usable measurement is required; a single measurement
    yields a zero drift rate.
    """
    usable = [
        m
        for m in measurements
        if m.camera_id == camera_id
        and m.measured_at_seconds is not None
        and m.confidence >= minimum_confidence
    ]
    if not usable:
        raise ValueError(f"no usable offset measurements for camera {camera_id!r}")

    times = np.asarray([m.measured_at_seconds for m in usable], dtype=np.float64)
    offsets = np.asarray([m.offset_seconds for m in usable], dtype=np.float64)
    weights = np.asarray([max(m.confidence, _EPSILON) for m in usable], dtype=np.float64)

    anchor = float(np.average(times, weights=weights))
    mean_offset = float(np.average(offsets, weights=weights))
    rate = 0.0
    if len(usable) >= 2:
        centered_t = times - anchor
        denominator = float(np.sum(weights * centered_t * centered_t))
        if denominator > _EPSILON:
            rate = float(
                np.sum(weights * centered_t * (offsets - mean_offset)) / denominator
            )
    predicted = mean_offset + rate * (times - anchor)
    residual = float(np.sqrt(np.average((offsets - predicted) ** 2, weights=weights)))
    return DriftModel(
        camera_id=camera_id,
        base_offset_seconds=mean_offset,
        drift_rate_seconds_per_second=rate,
        anchor_seconds=anchor,
        residual_seconds=residual,
        sample_count=len(usable),
    )


def apply_drift_model(synchronizer: TimelineSynchronizer, model: DriftModel) -> None:
    """Install a fitted drift model into a synchronizer."""
    synchronizer.set_drift_rate(model.camera_id, model.drift_rate_seconds_per_second)
