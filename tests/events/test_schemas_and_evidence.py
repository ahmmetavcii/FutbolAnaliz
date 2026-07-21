from __future__ import annotations

import pytest

from football_analytics.events import (
    EventStatus,
    EventType,
    EvidenceBundle,
    EvidenceItem,
    MatchEvent,
    SubstitutionInterval,
    is_confirmed,
)


class TestStatuses:
    def test_all_required_statuses_exist(self):
        assert {s.value for s in EventStatus} == {
            "auto_confirmed",
            "candidate_review_required",
            "manually_confirmed",
            "manually_rejected",
            "unresolved",
        }

    def test_only_confirmed_statuses_count(self):
        assert is_confirmed(EventStatus.AUTO_CONFIRMED)
        assert is_confirmed(EventStatus.MANUALLY_CONFIRMED)
        assert not is_confirmed(EventStatus.CANDIDATE_REVIEW_REQUIRED)
        assert not is_confirmed(EventStatus.MANUALLY_REJECTED)
        assert not is_confirmed(EventStatus.UNRESOLVED)


class TestMatchEvent:
    def test_unresolved_attribution_defaults_to_none(self):
        event = MatchEvent(
            event_id="goal-1",
            event_type=EventType.GOAL,
            status=EventStatus.UNRESOLVED,
            timestamp_ms=1000.0,
        )
        assert event.scorer_track_id is None
        assert event.assist_track_id is None

    def test_own_goal_with_assist_is_rejected(self):
        with pytest.raises(ValueError, match="assist"):
            MatchEvent(
                event_id="goal-1",
                event_type=EventType.GOAL,
                status=EventStatus.MANUALLY_CONFIRMED,
                timestamp_ms=1000.0,
                assist_track_id=5,
                attributes={"own_goal": True},
            )

    def test_penalty_with_assist_is_rejected(self):
        with pytest.raises(ValueError, match="assist"):
            MatchEvent(
                event_id="goal-1",
                event_type=EventType.GOAL,
                status=EventStatus.MANUALLY_CONFIRMED,
                timestamp_ms=1000.0,
                assist_track_id=5,
                attributes={"penalty": True},
            )


class TestSubstitutionInterval:
    def test_interval_orders_and_midpoint(self):
        interval = SubstitutionInterval(start_ms=1000.0, end_ms=5000.0)
        assert interval.midpoint_ms == 3000.0
        assert interval.duration_ms == 4000.0

    def test_inverted_interval_rejected(self):
        with pytest.raises(ValueError):
            SubstitutionInterval(start_ms=5000.0, end_ms=1000.0)


class TestEvidenceBundle:
    def test_replay_evidence_is_excluded_from_scoring(self):
        bundle = EvidenceBundle(
            (
                EvidenceItem("scoreboard_change", 0.9, from_replay=True),
                EvidenceItem("ball_crossed_line", 0.9, from_replay=True),
            )
        )
        assert bundle.aggregate_score() == 0.0
        assert not bundle.corroborated

    def test_single_source_is_capped(self):
        bundle = EvidenceBundle((EvidenceItem("ball_crossed_line", 1.0),))
        assert bundle.aggregate_score() <= bundle.single_source_cap
        assert not bundle.corroborated

    def test_corroboration_requires_two_distinct_sources(self):
        same_source = EvidenceBundle(
            (
                EvidenceItem("ball_crossed_line", 0.9),
                EvidenceItem("ball_crossed_line", 0.95),
            )
        )
        assert not same_source.corroborated
        two_sources = EvidenceBundle(
            (
                EvidenceItem("ball_crossed_line", 0.9),
                EvidenceItem("scoreboard_change", 0.9),
            )
        )
        assert two_sources.corroborated
        assert two_sources.aggregate_score() > same_source.aggregate_score()

    def test_contradicting_evidence_reduces_score(self):
        supported = EvidenceBundle(
            (
                EvidenceItem("ball_crossed_line", 0.9),
                EvidenceItem("scoreboard_change", 0.9),
            )
        )
        contradicted = supported.with_items(
            [EvidenceItem("play_continued", -0.8, description="no restart observed")]
        )
        assert contradicted.aggregate_score() < supported.aggregate_score()
