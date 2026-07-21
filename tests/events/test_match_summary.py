from __future__ import annotations

import pytest

from football_analytics.analytics.match_summary import build_match_summary
from football_analytics.analytics.officials_summary import OfficialSummary
from football_analytics.analytics.player_summary import PlayerSummary
from football_analytics.events import EventStatus, EventType, MatchEvent
from football_analytics.roles import PersonRole


def player(track_id: int, team_id: int, distance: float = 5000.0) -> PlayerSummary:
    return PlayerSummary(
        track_id=track_id,
        role=PersonRole.OUTFIELD_PLAYER,
        team_id=team_id,
        role_vote_share=0.9,
        total_distance_m=distance,
        max_speed_kmh=28.0,
        mean_speed_kmh=7.0,
        sprint_count=3,
        physical_metrics_valid=True,
        invalid_reason=None,
    )


def official(track_id: int) -> OfficialSummary:
    return OfficialSummary(
        track_id=track_id,
        role=PersonRole.REFEREE,
        role_vote_share=0.95,
        total_distance_m=9000.0,
        max_speed_kmh=24.0,
        invalid_reason=None,
    )


def goal(event_id: str, status: EventStatus, team_id: int = 0) -> MatchEvent:
    return MatchEvent(
        event_id=event_id,
        event_type=EventType.GOAL,
        status=status,
        timestamp_ms=60_000.0,
        team_id=team_id,
    )


class TestMatchSummary:
    def test_score_counts_only_confirmed_goals(self):
        summary = build_match_summary(
            player_summaries={1: player(1, 0)},
            official_summaries={99: official(99)},
            events=[
                goal("goal-1", EventStatus.AUTO_CONFIRMED, team_id=0),
                goal("goal-2", EventStatus.CANDIDATE_REVIEW_REQUIRED, team_id=1),
                goal("goal-3", EventStatus.MANUALLY_REJECTED, team_id=1),
            ],
        )
        assert summary.score == {0: 1}

    def test_officials_are_a_separate_section_not_in_teams(self):
        summary = build_match_summary(
            player_summaries={1: player(1, 0), 2: player(2, 0)},
            official_summaries={99: official(99)},
            events=[],
        )
        assert 99 in summary.officials
        assert summary.teams[0].player_count == 2
        assert summary.teams[0].total_distance_m == 10_000.0  # referee's 9 km absent

    def test_track_in_both_sections_is_an_error(self):
        with pytest.raises(ValueError, match="both players and officials"):
            build_match_summary(
                player_summaries={1: player(1, 0)},
                official_summaries={1: official(1)},
                events=[],
            )
