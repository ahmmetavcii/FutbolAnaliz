"""Per-track visibility and temporal coverage summaries.

Wraps :mod:`football_analytics.analytics.track_quality` so downstream
summaries can gate on how much of the match a track was actually observed.
Replay frames are excluded before any coverage math: footage that repeats a
moment must not inflate visibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from football_analytics.analytics.track_quality import TrackQuality, compute_track_quality


@dataclass(frozen=True)
class VisibilitySample:
    """One frame-level observation of a track."""

    track_id: int
    frame_id: int
    tracking_confidence: float = 1.0
    visibility: float = 1.0
    bbox_valid: bool = True
    replay: bool = False


@dataclass(frozen=True)
class TrackVisibility:
    track_id: int
    quality: TrackQuality
    observed_frames: int
    replay_frames_excluded: int

    @property
    def coverage(self) -> float:
        return self.quality.coverage

    @property
    def quality_score(self) -> float:
        return self.quality.quality_score


def summarize_visibility(
    samples: Iterable[VisibilitySample],
    *,
    total_frames: int | None = None,
) -> dict[int, TrackVisibility]:
    """Summarize visibility per track, excluding replay frames."""
    by_track: dict[int, list[VisibilitySample]] = {}
    replay_counts: dict[int, int] = {}
    for sample in samples:
        replay_counts.setdefault(sample.track_id, 0)
        if sample.replay:
            replay_counts[sample.track_id] += 1
            continue
        by_track.setdefault(sample.track_id, []).append(sample)

    results: dict[int, TrackVisibility] = {}
    for track_id in sorted(set(by_track) | set(replay_counts)):
        live = sorted(by_track.get(track_id, []), key=lambda s: s.frame_id)
        quality = compute_track_quality(
            [s.frame_id for s in live],
            total_frames=total_frames,
            tracking_confidences=[s.tracking_confidence for s in live],
            bbox_valid=[s.bbox_valid for s in live],
            visibility=[s.visibility for s in live],
        )
        results[track_id] = TrackVisibility(
            track_id=track_id,
            quality=quality,
            observed_frames=len(live),
            replay_frames_excluded=replay_counts.get(track_id, 0),
        )
    return results
