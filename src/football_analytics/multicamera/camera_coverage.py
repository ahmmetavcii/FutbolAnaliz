"""Coverage statistics: how much each camera contributes and where gaps are."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .local_tracking import LocalObservation


@dataclass(frozen=True)
class CameraCoverage:
    """Contribution summary for one camera."""

    camera_id: str
    observation_count: int
    track_count: int
    first_time_seconds: float | None
    last_time_seconds: float | None
    pitch_localized_fraction: float
    mean_detection_confidence: float

    @property
    def span_seconds(self) -> float:
        if self.first_time_seconds is None or self.last_time_seconds is None:
            return 0.0
        return self.last_time_seconds - self.first_time_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "observation_count": self.observation_count,
            "track_count": self.track_count,
            "first_time_seconds": self.first_time_seconds,
            "last_time_seconds": self.last_time_seconds,
            "span_seconds": self.span_seconds,
            "pitch_localized_fraction": self.pitch_localized_fraction,
            "mean_detection_confidence": self.mean_detection_confidence,
        }


def compute_camera_coverage(
    observations: Iterable[LocalObservation],
) -> dict[str, CameraCoverage]:
    """Aggregate per-camera counts, spans, and localization rates."""
    by_camera: dict[str, list[LocalObservation]] = {}
    for observation in observations:
        by_camera.setdefault(observation.camera_id, []).append(observation)

    coverage: dict[str, CameraCoverage] = {}
    for camera_id, rows in sorted(by_camera.items()):
        times = [obs.reference_time_seconds for obs in rows]
        localized = sum(1 for obs in rows if obs.pitch_xy_m is not None)
        coverage[camera_id] = CameraCoverage(
            camera_id=camera_id,
            observation_count=len(rows),
            track_count=len({obs.local_track_id for obs in rows}),
            first_time_seconds=min(times),
            last_time_seconds=max(times),
            pitch_localized_fraction=localized / len(rows),
            mean_detection_confidence=float(
                np.mean([obs.detection_confidence for obs in rows])
            ),
        )
    return coverage


def find_coverage_gaps(
    observations: Iterable[LocalObservation],
    expected_cameras: Sequence[str],
    duration_seconds: float,
    bin_seconds: float = 10.0,
    minimum_active_cameras: int = 1,
) -> list[dict[str, Any]]:
    """Locate time bins where fewer than the required cameras were observing."""
    if bin_seconds <= 0 or duration_seconds <= 0:
        raise ValueError("bin_seconds and duration_seconds must be positive")
    n_bins = int(np.ceil(duration_seconds / bin_seconds))
    active: dict[int, set[str]] = {index: set() for index in range(n_bins)}
    for observation in observations:
        index = int(observation.reference_time_seconds // bin_seconds)
        if 0 <= index < n_bins:
            active[index].add(observation.camera_id)

    expected = set(expected_cameras)
    gaps: list[dict[str, Any]] = []
    for index in range(n_bins):
        cameras = active[index] & expected if expected else active[index]
        if len(cameras) < minimum_active_cameras:
            gaps.append(
                {
                    "start_seconds": index * bin_seconds,
                    "end_seconds": min((index + 1) * bin_seconds, duration_seconds),
                    "active_cameras": sorted(cameras),
                    "missing_cameras": sorted(expected - cameras),
                }
            )
    return gaps


def coverage_report(
    observations: Sequence[LocalObservation],
    expected_cameras: Sequence[str],
    duration_seconds: float,
) -> dict[str, Any]:
    """Full coverage report: per-camera stats plus gap list."""
    coverage = compute_camera_coverage(observations)
    gaps = find_coverage_gaps(observations, expected_cameras, duration_seconds)
    return {
        "cameras": {camera_id: item.to_dict() for camera_id, item in coverage.items()},
        "missing_cameras": sorted(set(expected_cameras) - set(coverage)),
        "gaps": gaps,
        "gap_seconds": sum(gap["end_seconds"] - gap["start_seconds"] for gap in gaps),
    }
