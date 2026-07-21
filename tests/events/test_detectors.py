from __future__ import annotations

from football_analytics.events import (
    EventStatus,
    GoalDetector,
    GoalSignals,
    PitchEntry,
    PitchExit,
    ShotDetector,
    ShotSignals,
    SubstitutionDetector,
    interval_of,
)


class TestGoalDetector:
    def test_single_strong_signal_is_never_auto_confirmed(self):
        event = GoalDetector().detect(
            GoalSignals(timestamp_ms=60_000.0, team_id=0, ball_crossed_line_score=1.0)
        )
        assert event is not None
        assert event.status is EventStatus.CANDIDATE_REVIEW_REQUIRED
        assert not event.counts_as_confirmed

    def test_corroborated_strong_signals_auto_confirm(self):
        event = GoalDetector().detect(
            GoalSignals(
                timestamp_ms=60_000.0,
                team_id=0,
                ball_crossed_line_score=0.9,
                scoreboard_change_score=0.9,
                kickoff_restart_score=0.9,
            )
        )
        assert event is not None
        assert event.status is EventStatus.AUTO_CONFIRMED

    def test_weak_evidence_is_unresolved(self):
        event = GoalDetector().detect(
            GoalSignals(timestamp_ms=60_000.0, ball_crossed_line_score=0.3)
        )
        assert event is not None
        assert event.status is EventStatus.UNRESOLVED

    def test_replay_signals_produce_no_event(self):
        event = GoalDetector().detect(
            GoalSignals(
                timestamp_ms=60_000.0,
                ball_crossed_line_score=0.9,
                scoreboard_change_score=0.9,
                from_replay=True,
            )
        )
        assert event is None

    def test_unresolved_scorer_stays_null_even_when_confirmed(self):
        event = GoalDetector().detect(
            GoalSignals(
                timestamp_ms=60_000.0,
                team_id=0,
                ball_crossed_line_score=0.9,
                scoreboard_change_score=0.9,
                kickoff_restart_score=0.9,
                scorer_track_id=12,
                scorer_attribution_score=0.4,  # below attribution threshold
            )
        )
        assert event is not None
        assert event.status is EventStatus.AUTO_CONFIRMED
        assert event.scorer_track_id is None

    def test_confident_scorer_is_attached_to_confirmed_goal(self):
        event = GoalDetector().detect(
            GoalSignals(
                timestamp_ms=60_000.0,
                team_id=0,
                ball_crossed_line_score=0.9,
                scoreboard_change_score=0.9,
                kickoff_restart_score=0.9,
                scorer_track_id=12,
                scorer_attribution_score=0.9,
            )
        )
        assert event is not None
        assert event.scorer_track_id == 12

    def test_candidate_keeps_scorer_only_as_suggestion(self):
        event = GoalDetector().detect(
            GoalSignals(
                timestamp_ms=60_000.0,
                ball_crossed_line_score=1.0,
                scorer_track_id=12,
                scorer_attribution_score=0.9,
            )
        )
        assert event is not None
        assert event.status is EventStatus.CANDIDATE_REVIEW_REQUIRED
        assert event.scorer_track_id is None
        assert event.attributes["suggested_scorer_track_id"] == 12

    def test_own_goal_flag_is_recorded_without_assist(self):
        event = GoalDetector().detect(
            GoalSignals(
                timestamp_ms=60_000.0,
                team_id=1,
                ball_crossed_line_score=0.9,
                scoreboard_change_score=0.9,
                own_goal=True,
            )
        )
        assert event is not None
        assert event.own_goal
        assert event.assist_track_id is None


class TestShotDetector:
    def test_shot_candidate_from_ball_motion(self):
        event = ShotDetector().detect(
            ShotSignals(
                timestamp_ms=30_000.0,
                team_id=1,
                ball_toward_goal_score=0.8,
                ball_speed_spike_score=0.7,
                on_target=True,
            )
        )
        assert event is not None
        assert event.status is EventStatus.CANDIDATE_REVIEW_REQUIRED
        assert event.attributes["on_target"] is True

    def test_no_signals_no_event(self):
        assert ShotDetector().detect(ShotSignals(timestamp_ms=30_000.0)) is None


class TestSubstitutionDetector:
    def test_paired_exit_and_entry_form_an_interval(self):
        detector = SubstitutionDetector()
        events = detector.detect(
            exits=[PitchExit(track_id=7, timestamp_ms=100_000.0, team_id=0, confidence=0.8)],
            entries=[
                PitchEntry(
                    track_id=21,
                    timestamp_ms=112_000.0,
                    team_id=0,
                    confidence=0.8,
                    came_from_bench=True,
                )
            ],
        )
        assert len(events) == 1
        event = events[0]
        interval = interval_of(event)
        assert interval is not None
        assert interval.start_ms == 100_000.0
        assert interval.end_ms == 112_000.0
        assert event.timestamp_ms == interval.midpoint_ms
        assert event.attributes["player_off_track_id"] == 7
        assert event.attributes["player_on_track_id"] == 21

    def test_unpaired_exit_never_becomes_a_candidate(self):
        detector = SubstitutionDetector()
        events = detector.detect(
            exits=[PitchExit(track_id=7, timestamp_ms=100_000.0, team_id=0, confidence=0.9)],
            entries=[],
        )
        assert len(events) == 1
        assert events[0].status is EventStatus.UNRESOLVED
        assert events[0].attributes["unpaired_exit"] is True

    def test_entries_from_other_team_are_not_paired(self):
        detector = SubstitutionDetector()
        events = detector.detect(
            exits=[PitchExit(track_id=7, timestamp_ms=100_000.0, team_id=0, confidence=0.8)],
            entries=[PitchEntry(track_id=21, timestamp_ms=101_000.0, team_id=1, confidence=0.8)],
        )
        assert len(events) == 1
        assert events[0].attributes.get("unpaired_exit") is True

    def test_replay_observations_are_ignored(self):
        detector = SubstitutionDetector()
        events = detector.detect(
            exits=[
                PitchExit(
                    track_id=7, timestamp_ms=100_000.0, team_id=0, confidence=0.9, from_replay=True
                )
            ],
            entries=[],
        )
        assert events == []
