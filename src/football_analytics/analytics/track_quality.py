"""Track-level quality metrics for downstream analytics gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class TrackQuality:
    observations: int
    frame_span: int
    coverage: float
    fragmentation_count: int
    continuity: float
    mean_tracking_confidence: float
    valid_bbox_fraction: float
    mean_visibility: float
    mean_foot_point_confidence: float
    quality_score: float

    @property
    def fragments(self) -> int:
        return self.fragmentation_count + (1 if self.observations else 0)


def compute_track_quality(
    frame_ids: Sequence[int],
    *,
    total_frames: int | None = None,
    tracking_confidences: Sequence[float] | None = None,
    bbox_valid: Sequence[bool] | None = None,
    visibility: Sequence[float] | None = None,
    foot_point_confidences: Sequence[float] | None = None,
) -> TrackQuality:
    """Calculate coverage, fragmentation, continuity, and observation quality."""
    frames = np.asarray(frame_ids, dtype=np.int64)
    if frames.ndim != 1:
        raise ValueError("frame_ids must be one-dimensional")
    frames = np.unique(frames)
    observations = int(frames.size)
    if observations:
        span = int(frames[-1] - frames[0] + 1)
        gaps = np.diff(frames)
        fragmentation = int(np.count_nonzero(gaps > 1))
        continuity = observations / span
    else:
        span, fragmentation, continuity = 0, 0, 0.0
    denominator = total_frames if total_frames is not None else span
    if denominator < span or denominator < 0:
        raise ValueError("total_frames must be non-negative and at least the track span")
    coverage = observations / denominator if denominator else 0.0

    mean_confidence = _mean_metric(tracking_confidences, len(frame_ids), default=0.0)
    valid_fraction = _mean_metric(bbox_valid, len(frame_ids), default=1.0 if observations else 0.0)
    mean_visibility = _mean_metric(visibility, len(frame_ids), default=1.0 if observations else 0.0)
    mean_foot = _mean_metric(
        foot_point_confidences, len(frame_ids), default=1.0 if observations else 0.0
    )
    quality = float(
        np.clip(
            coverage ** 0.20
            * continuity ** 0.25
            * mean_confidence ** 0.20
            * valid_fraction ** 0.10
            * mean_visibility ** 0.10
            * mean_foot ** 0.15,
            0.0,
            1.0,
        )
    )
    return TrackQuality(
        observations=observations,
        frame_span=span,
        coverage=float(coverage),
        fragmentation_count=fragmentation,
        continuity=float(continuity),
        mean_tracking_confidence=mean_confidence,
        valid_bbox_fraction=valid_fraction,
        mean_visibility=mean_visibility,
        mean_foot_point_confidence=mean_foot,
        quality_score=quality,
    )


def summarize_track(
    records: Iterable[Mapping[str, object]], *, total_frames: int | None = None
) -> TrackQuality:
    """Summarize canonical track rows without requiring pandas."""
    rows = list(records)
    return compute_track_quality(
        [int(row["frame_id"]) for row in rows],
        total_frames=total_frames,
        tracking_confidences=[float(row.get("tracking_confidence", 0.0)) for row in rows],
        bbox_valid=[bool(row.get("bbox_valid", True)) for row in rows],
        visibility=[float(row.get("visibility", row.get("visible_fraction", 1.0))) for row in rows],
        foot_point_confidences=[
            float(row.get("foot_point_confidence", 1.0)) for row in rows
        ],
    )


def _mean_metric(
    values: Sequence[float] | Sequence[bool] | None, expected: int, *, default: float
) -> float:
    if values is None:
        return default
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != expected:
        raise ValueError("metric sequences must align with frame_ids")
    if array.size == 0:
        return default
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("quality metrics must be finite values in [0, 1]")
    return float(np.mean(array))
