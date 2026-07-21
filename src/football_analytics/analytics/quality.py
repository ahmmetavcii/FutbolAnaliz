"""Match-level data-quality reporting.

Downstream consumers need to know how much to trust the numbers. This module
aggregates per-frame gating outcomes (calibration validity, replay share,
accepted shot types) and per-track quality into one report. It reuses
:mod:`football_analytics.analytics.track_quality` for track scoring instead
of inventing another metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from football_analytics.analytics.track_quality import TrackQuality


@dataclass(frozen=True)
class FrameGateRecord:
    """Per-frame gating outcome used for coverage accounting."""

    frame_id: int
    calibration_valid: bool
    replay: bool
    shot_type: str = "wide"


@dataclass(frozen=True)
class MatchQualityReport:
    total_frames: int
    calibrated_fraction: float
    replay_fraction: float
    accepted_shot_fraction: float
    usable_fraction: float
    track_count: int
    mean_track_quality: float
    low_quality_track_ids: tuple[int, ...]

    @property
    def physical_metrics_trustworthy(self) -> bool:
        """Coarse gate: enough calibrated, live, accepted footage to report."""
        return self.usable_fraction >= 0.5 and self.calibrated_fraction >= 0.5


def build_quality_report(
    frames: Iterable[FrameGateRecord],
    track_qualities: Mapping[int, TrackQuality],
    *,
    accepted_shot_types: frozenset[str] = frozenset({"wide", "tactical"}),
    low_quality_threshold: float = 0.3,
) -> MatchQualityReport:
    frame_list = list(frames)
    total = len(frame_list)
    if total == 0:
        calibrated = replay = accepted = usable = 0.0
    else:
        calibrated = sum(1 for f in frame_list if f.calibration_valid) / total
        replay = sum(1 for f in frame_list if f.replay) / total
        accepted = sum(1 for f in frame_list if f.shot_type in accepted_shot_types) / total
        usable = (
            sum(
                1
                for f in frame_list
                if f.calibration_valid and not f.replay and f.shot_type in accepted_shot_types
            )
            / total
        )

    scores = {tid: tq.quality_score for tid, tq in track_qualities.items()}
    mean_quality = sum(scores.values()) / len(scores) if scores else 0.0
    low = tuple(sorted(tid for tid, s in scores.items() if s < low_quality_threshold))

    return MatchQualityReport(
        total_frames=total,
        calibrated_fraction=calibrated,
        replay_fraction=replay,
        accepted_shot_fraction=accepted,
        usable_fraction=usable,
        track_count=len(scores),
        mean_track_quality=mean_quality,
        low_quality_track_ids=low,
    )
