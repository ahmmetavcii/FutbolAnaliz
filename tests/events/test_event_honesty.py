from __future__ import annotations

from football_analytics.events.goal_detector import GoalDetector, GoalSignals
from football_analytics.events.event_summary import summarize_events
from football_analytics.events.schemas import EventStatus


def test_unresolved_scorer_stays_null():
    event = GoalDetector().detect(
        GoalSignals(
            timestamp_ms=1000.0,
            team_id=0,
            ball_crossed_line_score=0.8,
            scoreboard_change_score=0.8,
            scorer_track_id=None,
        )
    )
    assert event is not None
    assert event.scorer_track_id is None


def test_candidates_excluded_from_confirmed_totals():
    event = GoalDetector().detect(
        GoalSignals(
            timestamp_ms=1000.0,
            team_id=0,
            ball_crossed_line_score=0.6,
            scoreboard_change_score=0.5,
            scorer_track_id=1,
            scorer_attribution_score=0.9,
        )
    )
    assert event is not None
    event = event.with_status(EventStatus.CANDIDATE_REVIEW_REQUIRED)
    summary = summarize_events([event])
    assert summary.confirmed_goals_by_team == {}
    assert summary.confirmed_goals_by_scorer == {}
    assert summary.pending_review_event_ids == (event.event_id,)
    # Candidates must not be counted as confirmed.
    assert not event.counts_as_confirmed


def test_own_goal_no_assist():
    event = GoalDetector().detect(
        GoalSignals(
            timestamp_ms=2000.0,
            team_id=0,
            ball_crossed_line_score=0.9,
            scoreboard_change_score=0.9,
            kickoff_restart_score=0.8,
            scorer_track_id=9,
            scorer_attribution_score=0.95,
            own_goal=True,
        )
    )
    assert event is not None
    assert event.assist_track_id is None
    assert event.own_goal is True
