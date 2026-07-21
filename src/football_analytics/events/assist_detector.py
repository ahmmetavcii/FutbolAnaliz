"""Assist candidates linked to confirmed goals.

An assist is proposed only when:

- the goal itself is confirmed (auto or manual) — no assists for candidates;
- the goal is neither an own goal nor a direct penalty;
- a pass to the scorer completed within the linking window with sufficient
  pass-detection confidence;
- the passer is a different player from the scorer.

If any condition fails the assist stays ``None``. The proposed assist is
itself an event needing review; it never silently mutates the goal record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from football_analytics.events.event_detector import EventDetector, EventDetectorConfig
from football_analytics.events.event_evidence import EvidenceItem
from football_analytics.events.schemas import EventType, MatchEvent


@dataclass(frozen=True)
class PassObservation:
    """One detected completed pass (evidence from the possession chain)."""

    timestamp_ms: float
    passer_track_id: int
    receiver_track_id: int
    team_id: int | None = None
    confidence: float = 0.0
    from_replay: bool = False


@dataclass(frozen=True)
class AssistDetectorConfig:
    detector: EventDetectorConfig = EventDetectorConfig()
    #: Maximum time between the assisting pass and the goal.
    max_link_window_ms: float = 15_000.0
    #: Minimum pass confidence to even propose an assist.
    min_pass_confidence: float = 0.5


class AssistDetector(EventDetector):
    event_type = EventType.ASSIST

    def __init__(self, config: AssistDetectorConfig | None = None) -> None:
        self.assist_config = config or AssistDetectorConfig()
        super().__init__(self.assist_config.detector)

    def detect_for_goal(
        self, goal: MatchEvent, passes: Sequence[PassObservation]
    ) -> MatchEvent | None:
        if goal.event_type is not EventType.GOAL:
            raise ValueError("assist detection requires a goal event")
        if not goal.counts_as_confirmed:
            return None
        if goal.own_goal or goal.penalty:
            return None
        if goal.scorer_track_id is None:
            # Unresolved scorer: the assist cannot be resolved either.
            return None

        cfg = self.assist_config
        candidates = [
            p
            for p in passes
            if not p.from_replay
            and p.receiver_track_id == goal.scorer_track_id
            and p.passer_track_id != goal.scorer_track_id
            and p.confidence >= cfg.min_pass_confidence
            and 0.0 <= goal.timestamp_ms - p.timestamp_ms <= cfg.max_link_window_ms
        ]
        if not candidates:
            return None
        last_pass = max(candidates, key=lambda p: p.timestamp_ms)

        recency = 1.0 - (goal.timestamp_ms - last_pass.timestamp_ms) / cfg.max_link_window_ms
        items = [
            EvidenceItem(
                source="linked_pass",
                score=last_pass.confidence,
                timestamp_ms=last_pass.timestamp_ms,
                description=f"pass to scorer track {goal.scorer_track_id}",
            ),
            EvidenceItem(
                source="pass_recency",
                score=max(0.0, min(1.0, recency)),
                timestamp_ms=last_pass.timestamp_ms,
            ),
        ]
        return self.build_event(
            timestamp_ms=last_pass.timestamp_ms,
            evidence=items,
            team_id=last_pass.team_id if last_pass.team_id is not None else goal.team_id,
            assist_track_id=last_pass.passer_track_id,
            attributes={"goal_event_id": goal.event_id},
            # The passer *is* the subject of an assist candidate; without it
            # the event would be meaningless to review. Counting still only
            # happens after confirmation.
            attribution_is_subject=True,
        )
