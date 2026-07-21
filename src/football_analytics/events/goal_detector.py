"""Goal candidate detection from aggregated, conservative evidence.

Signals that typically feed a goal candidate: estimated ball-crossed-line
geometry, a scoreboard/graphic score change, a centre-circle restart shortly
after, and a sustained possession/celebration break. None of these alone is
proof, so single-signal candidates stay below auto-confirmation by design.

Attribution rules enforced here:

- Scorer stays ``None`` unless attribution evidence is itself strong; an
  unresolved scorer is represented as ``None``, never guessed.
- Own goals and direct penalty goals never carry an assist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from football_analytics.events.event_detector import EventDetector, EventDetectorConfig
from football_analytics.events.event_evidence import EvidenceBundle, EvidenceItem
from football_analytics.events.schemas import EventType, MatchEvent


@dataclass(frozen=True)
class GoalSignals:
    """Pre-computed upstream signals around one goal-candidate moment.

    Scores are in ``[0, 1]``; ``None`` means the signal was unavailable.
    Replay-derived observations must be flagged so they are excluded.
    """

    timestamp_ms: float
    team_id: int | None = None
    ball_crossed_line_score: float | None = None
    scoreboard_change_score: float | None = None
    kickoff_restart_score: float | None = None
    celebration_break_score: float | None = None
    from_replay: bool = False
    #: Attribution evidence: last confirmed toucher before the ball crossed.
    scorer_track_id: int | None = None
    scorer_attribution_score: float = 0.0
    own_goal: bool = False
    penalty: bool = False


@dataclass(frozen=True)
class GoalDetectorConfig:
    detector: EventDetectorConfig = EventDetectorConfig()
    #: Minimum attribution score to attach a scorer to a confirmed goal.
    min_scorer_attribution: float = 0.7


class GoalDetector(EventDetector):
    event_type = EventType.GOAL

    def __init__(self, config: GoalDetectorConfig | None = None) -> None:
        self.goal_config = config or GoalDetectorConfig()
        super().__init__(self.goal_config.detector)

    def detect(self, signals: GoalSignals) -> MatchEvent | None:
        items: list[EvidenceItem] = []
        for source, score in (
            ("ball_crossed_line", signals.ball_crossed_line_score),
            ("scoreboard_change", signals.scoreboard_change_score),
            ("kickoff_restart", signals.kickoff_restart_score),
            ("celebration_break", signals.celebration_break_score),
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

        scorer: int | None = None
        if (
            signals.scorer_track_id is not None
            and signals.scorer_attribution_score >= self.goal_config.min_scorer_attribution
            and not signals.from_replay
        ):
            scorer = signals.scorer_track_id

        return self.build_event(
            timestamp_ms=signals.timestamp_ms,
            evidence=items,
            team_id=signals.team_id,
            scorer_track_id=scorer,
            # Assist is never decided here; the assist detector links passes to
            # confirmed goals, and own goals / penalties forbid assists anyway.
            assist_track_id=None,
            attributes={
                "own_goal": signals.own_goal,
                "penalty": signals.penalty,
            },
        )

    def detect_all(self, signals: Sequence[GoalSignals]) -> list[MatchEvent]:
        events = [self.detect(s) for s in signals]
        return [event for event in events if event is not None]


def goal_evidence_bundle(signals: GoalSignals) -> EvidenceBundle:
    """Expose the evidence bundle a :class:`GoalDetector` would score."""
    detector = GoalDetector()
    event = detector.detect(signals)
    return event.evidence if event is not None else EvidenceBundle()
