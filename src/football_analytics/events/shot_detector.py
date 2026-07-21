"""Shot candidate detection from ball-motion and context evidence.

A "shot" candidate is surfaced when the ball accelerates toward goal from a
plausible striking position. This is an evidence aggregation, not a claim of
real shot-detection accuracy: ambiguous clearances, crosses, and deflections
look similar, so candidates default to review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from football_analytics.events.event_detector import EventDetector, EventDetectorConfig
from football_analytics.events.event_evidence import EvidenceItem
from football_analytics.events.schemas import EventType, MatchEvent


@dataclass(frozen=True)
class ShotSignals:
    """Upstream signals around one shot-candidate moment (scores in [0, 1])."""

    timestamp_ms: float
    team_id: int | None = None
    ball_toward_goal_score: float | None = None
    ball_speed_spike_score: float | None = None
    striking_pose_score: float | None = None
    goalkeeper_reaction_score: float | None = None
    from_replay: bool = False
    shooter_track_id: int | None = None
    shooter_attribution_score: float = 0.0
    on_target: bool | None = None


@dataclass(frozen=True)
class ShotDetectorConfig:
    detector: EventDetectorConfig = EventDetectorConfig()
    min_shooter_attribution: float = 0.7


class ShotDetector(EventDetector):
    event_type = EventType.SHOT

    def __init__(self, config: ShotDetectorConfig | None = None) -> None:
        self.shot_config = config or ShotDetectorConfig()
        super().__init__(self.shot_config.detector)

    def detect(self, signals: ShotSignals) -> MatchEvent | None:
        items: list[EvidenceItem] = []
        for source, score in (
            ("ball_toward_goal", signals.ball_toward_goal_score),
            ("ball_speed_spike", signals.ball_speed_spike_score),
            ("striking_pose", signals.striking_pose_score),
            ("goalkeeper_reaction", signals.goalkeeper_reaction_score),
        ):
            if score is not None:
                items.append(
                    EvidenceItem(
                        source=source,
                        score=score,
                        timestamp_ms=signals.timestamp_ms,
                        from_replay=signals.from_replay,
                    )
                )
        if not items:
            return None

        shooter: int | None = None
        if (
            signals.shooter_track_id is not None
            and signals.shooter_attribution_score >= self.shot_config.min_shooter_attribution
            and not signals.from_replay
        ):
            shooter = signals.shooter_track_id

        attributes: dict = {}
        if signals.on_target is not None:
            attributes["on_target"] = signals.on_target
        return self.build_event(
            timestamp_ms=signals.timestamp_ms,
            evidence=items,
            team_id=signals.team_id,
            scorer_track_id=shooter,
            attributes=attributes,
        )

    def detect_all(self, signals: Sequence[ShotSignals]) -> list[MatchEvent]:
        events = [self.detect(s) for s in signals]
        return [event for event in events if event is not None]
