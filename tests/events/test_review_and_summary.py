from __future__ import annotations

from football_analytics.events import (
    EventStatus,
    EventType,
    MatchEvent,
    ReviewLog,
    apply_review,
    recompute_summary,
    summarize_events,
)


def event(
    event_id: str,
    event_type: EventType = EventType.GOAL,
    status: EventStatus = EventStatus.CANDIDATE_REVIEW_REQUIRED,
    *,
    team_id: int | None = 0,
    scorer: int | None = None,
    assist: int | None = None,
    ts: float = 1000.0,
    attributes: dict | None = None,
) -> MatchEvent:
    return MatchEvent(
        event_id=event_id,
        event_type=event_type,
        status=status,
        timestamp_ms=ts,
        team_id=team_id,
        scorer_track_id=scorer,
        assist_track_id=assist,
        attributes=attributes or {},
    )


class TestReview:
    def test_confirm_and_reject(self):
        events = [event("goal-1"), event("goal-2")]
        log = ReviewLog()
        log.confirm("goal-1", reviewer="ana")
        log.reject("goal-2", reviewer="ana")
        result = apply_review(events, log)
        statuses = {e.event_id: e.status for e in result.events}
        assert statuses["goal-1"] is EventStatus.MANUALLY_CONFIRMED
        assert statuses["goal-2"] is EventStatus.MANUALLY_REJECTED

    def test_corrections_do_not_mutate_originals(self):
        original = event("goal-1")
        log = ReviewLog()
        log.confirm("goal-1")
        apply_review([original], log)
        assert original.status is EventStatus.CANDIDATE_REVIEW_REQUIRED

    def test_scorer_can_be_set_and_nulled(self):
        events = [event("goal-1", scorer=None)]
        log = ReviewLog()
        log.set_scorer("goal-1", 12)
        reviewed = apply_review(events, log).events[0]
        assert reviewed.scorer_track_id == 12
        log.set_scorer("goal-1", None)
        reviewed = apply_review(events, log).events[0]
        assert reviewed.scorer_track_id is None

    def test_assist_on_own_goal_is_rejected_with_reason(self):
        events = [event("goal-1", attributes={"own_goal": True})]
        log = ReviewLog()
        log.set_assist("goal-1", 5)
        result = apply_review(events, log)
        assert result.events[0].assist_track_id is None
        assert len(result.rejected_corrections) == 1
        assert "assist" in result.rejected_corrections[0][1]

    def test_unknown_event_id_is_reported(self):
        log = ReviewLog()
        log.confirm("missing-1")
        result = apply_review([event("goal-1")], log)
        assert result.unmatched_event_ids == ("missing-1",)


class TestSummary:
    def test_candidates_are_never_counted_as_confirmed(self):
        events = [
            event("goal-1", status=EventStatus.CANDIDATE_REVIEW_REQUIRED),
            event("goal-2", status=EventStatus.UNRESOLVED),
            event("goal-3", status=EventStatus.MANUALLY_REJECTED),
        ]
        summary = summarize_events(events)
        assert summary.confirmed_goals_by_team == {}
        assert summary.pending_review_event_ids == ("goal-1",)
        assert summary.unresolved_event_ids == ("goal-2",)

    def test_confirmed_goal_with_null_scorer_counts_for_team_only(self):
        events = [event("goal-1", status=EventStatus.AUTO_CONFIRMED, scorer=None)]
        summary = summarize_events(events)
        assert summary.confirmed_goals_by_team == {0: 1}
        assert summary.confirmed_goals_by_scorer == {}
        assert summary.unattributed_confirmed_goals == 1

    def test_own_goal_credits_opposing_team_and_no_scorer(self):
        events = [
            event(
                "goal-1",
                status=EventStatus.MANUALLY_CONFIRMED,
                team_id=1,
                scorer=9,
                attributes={"own_goal": True},
            )
        ]
        summary = summarize_events(events, team_ids=(0, 1))
        assert summary.confirmed_goals_by_team == {0: 1}
        assert summary.confirmed_goals_by_scorer == {}

    def test_assists_count_only_when_confirmed(self):
        events = [
            event(
                "assist-1",
                event_type=EventType.ASSIST,
                status=EventStatus.CANDIDATE_REVIEW_REQUIRED,
                assist=8,
            ),
            event(
                "assist-2",
                event_type=EventType.ASSIST,
                status=EventStatus.MANUALLY_CONFIRMED,
                assist=8,
            ),
        ]
        summary = summarize_events(events)
        assert summary.confirmed_assists_by_player == {8: 1}

    def test_shots_and_substitutions_by_team(self):
        events = [
            event("shot-1", event_type=EventType.SHOT, status=EventStatus.AUTO_CONFIRMED),
            event(
                "sub-1",
                event_type=EventType.SUBSTITUTION,
                status=EventStatus.MANUALLY_CONFIRMED,
                team_id=1,
            ),
        ]
        summary = summarize_events(events)
        assert summary.confirmed_shots_by_team == {0: 1}
        assert summary.confirmed_substitutions_by_team == {1: 1}


class TestRecompute:
    def test_recompute_reapplies_corrections_to_fresh_detections(self):
        log = ReviewLog()
        log.confirm("goal-1")
        log.set_scorer("goal-1", 12)

        first_detection = [event("goal-1")]
        summary_one = recompute_summary(first_detection, log)
        assert summary_one.confirmed_goals_by_team == {0: 1}
        assert summary_one.confirmed_goals_by_scorer == {12: 1}

        # Re-run detection (e.g. new model), same event id, extra candidate.
        second_detection = [event("goal-1"), event("goal-9")]
        summary_two = recompute_summary(second_detection, log)
        assert summary_two.confirmed_goals_by_team == {0: 1}
        assert summary_two.confirmed_goals_by_scorer == {12: 1}
        assert summary_two.pending_review_event_ids == ("goal-9",)

    def test_recompute_is_deterministic(self):
        log = ReviewLog()
        log.confirm("goal-1")
        events = [event("goal-1"), event("goal-2")]
        assert recompute_summary(events, log) == recompute_summary(events, log)
