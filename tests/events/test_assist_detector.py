from __future__ import annotations

import pytest

from football_analytics.events import (
    AssistDetector,
    EventStatus,
    EventType,
    MatchEvent,
    PassObservation,
)


def goal(
    *,
    status: EventStatus = EventStatus.AUTO_CONFIRMED,
    scorer: int | None = 10,
    own_goal: bool = False,
    penalty: bool = False,
    timestamp_ms: float = 60_000.0,
) -> MatchEvent:
    return MatchEvent(
        event_id="goal-1",
        event_type=EventType.GOAL,
        status=status,
        timestamp_ms=timestamp_ms,
        team_id=0,
        scorer_track_id=scorer,
        attributes={"own_goal": own_goal, "penalty": penalty},
    )


def pass_to(receiver: int, *, passer: int = 8, ts: float = 55_000.0, conf: float = 0.9):
    return PassObservation(
        timestamp_ms=ts,
        passer_track_id=passer,
        receiver_track_id=receiver,
        team_id=0,
        confidence=conf,
    )


class TestAssistDetector:
    def test_assist_proposed_for_confirmed_goal_with_linked_pass(self):
        event = AssistDetector().detect_for_goal(goal(), [pass_to(10)])
        assert event is not None
        assert event.assist_track_id == 8
        assert event.attributes["goal_event_id"] == "goal-1"
        # An assist candidate still needs review before it counts.
        assert not event.counts_as_confirmed

    def test_no_assist_for_unconfirmed_goal(self):
        candidate = goal(status=EventStatus.CANDIDATE_REVIEW_REQUIRED)
        assert AssistDetector().detect_for_goal(candidate, [pass_to(10)]) is None

    def test_no_assist_for_own_goal(self):
        assert AssistDetector().detect_for_goal(goal(own_goal=True), [pass_to(10)]) is None

    def test_no_assist_for_direct_penalty(self):
        assert AssistDetector().detect_for_goal(goal(penalty=True), [pass_to(10)]) is None

    def test_unresolved_scorer_means_no_assist(self):
        assert AssistDetector().detect_for_goal(goal(scorer=None), [pass_to(10)]) is None

    def test_pass_to_someone_else_is_not_linked(self):
        assert AssistDetector().detect_for_goal(goal(), [pass_to(99)]) is None

    def test_self_pass_is_not_an_assist(self):
        assert (
            AssistDetector().detect_for_goal(goal(), [pass_to(10, passer=10)]) is None
        )

    def test_pass_outside_link_window_is_not_linked(self):
        stale = pass_to(10, ts=10_000.0)  # 50 s before the goal
        assert AssistDetector().detect_for_goal(goal(), [stale]) is None

    def test_replay_pass_is_ignored(self):
        replay_pass = PassObservation(
            timestamp_ms=55_000.0,
            passer_track_id=8,
            receiver_track_id=10,
            confidence=0.9,
            from_replay=True,
        )
        assert AssistDetector().detect_for_goal(goal(), [replay_pass]) is None

    def test_low_confidence_pass_is_ignored(self):
        assert AssistDetector().detect_for_goal(goal(), [pass_to(10, conf=0.2)]) is None

    def test_most_recent_qualifying_pass_wins(self):
        earlier = pass_to(10, passer=3, ts=50_000.0)
        later = pass_to(10, passer=8, ts=58_000.0)
        event = AssistDetector().detect_for_goal(goal(), [earlier, later])
        assert event is not None
        assert event.assist_track_id == 8

    def test_non_goal_event_is_rejected(self):
        shot = MatchEvent(
            event_id="shot-1",
            event_type=EventType.SHOT,
            status=EventStatus.AUTO_CONFIRMED,
            timestamp_ms=1000.0,
        )
        with pytest.raises(ValueError):
            AssistDetector().detect_for_goal(shot, [])
