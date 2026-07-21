"""Role-driven summary rules: officials excluded, calibration gating."""

from __future__ import annotations

from football_analytics.analytics.officials_summary import summarize_officials
from football_analytics.analytics.player_metrics import PlayerSample
from football_analytics.analytics.player_summary import summarize_players
from football_analytics.analytics.team_summary import summarize_teams
from football_analytics.roles import PersonRole, RoleVote


def _walk_samples(track_id: int, *, calibration_valid: bool = True) -> list[PlayerSample]:
    # 1 m per second, one sample per second, 10 samples.
    return [
        PlayerSample(
            track_id=track_id,
            timestamp_ms=1000.0 * i,
            x_field=float(i) if calibration_valid else None,
            y_field=0.0 if calibration_valid else None,
            calibration_valid=calibration_valid,
        )
        for i in range(10)
    ]


def _vote(track_id: int, role: PersonRole) -> RoleVote:
    return RoleVote(track_id=track_id, role=role, vote_share=0.9, observations=30)


class TestOfficialsExcludedFromTeamTotals:
    def test_referee_distance_never_reaches_team_totals(self):
        votes = {
            1: _vote(1, PersonRole.OUTFIELD_PLAYER),
            2: _vote(2, PersonRole.GOALKEEPER),
            3: _vote(3, PersonRole.REFEREE),
        }
        samples = _walk_samples(1) + _walk_samples(2) + _walk_samples(3)
        team_ids = {1: 0, 2: 0, 3: None}

        players = summarize_players(votes, team_ids, samples)
        officials = summarize_officials(votes, samples)
        teams = summarize_teams(players)

        assert 3 not in players
        assert 3 in officials
        assert officials[3].total_distance_m is not None
        assert teams[0].player_count == 2
        # Team total covers exactly the two team members' distance.
        expected = players[1].total_distance_m + players[2].total_distance_m
        assert teams[0].total_distance_m == expected

    def test_goalkeeper_is_included_in_team_totals(self):
        votes = {2: _vote(2, PersonRole.GOALKEEPER)}
        players = summarize_players(votes, {2: 1}, _walk_samples(2))
        teams = summarize_teams(players)
        assert teams[1].player_count == 1
        assert teams[1].goalkeeper_track_ids == (2,)

    def test_substitute_summary_exists_but_does_not_count(self):
        votes = {5: _vote(5, PersonRole.SUBSTITUTE)}
        players = summarize_players(votes, {5: 0}, _walk_samples(5))
        assert 5 in players
        assert not players[5].counts_toward_team_totals
        assert summarize_teams(players) == {}


class TestCalibrationGating:
    def test_invalid_calibration_gives_null_physical_metrics(self):
        votes = {1: _vote(1, PersonRole.OUTFIELD_PLAYER)}
        samples = _walk_samples(1, calibration_valid=False)
        players = summarize_players(votes, {1: 0}, samples)
        summary = players[1]
        assert summary.total_distance_m is None
        assert summary.max_speed_kmh is None
        assert summary.mean_speed_kmh is None
        assert not summary.physical_metrics_valid
        assert summary.invalid_reason == "insufficient_usable_samples"

    def test_invalid_metrics_do_not_poison_team_totals(self):
        votes = {
            1: _vote(1, PersonRole.OUTFIELD_PLAYER),
            2: _vote(2, PersonRole.OUTFIELD_PLAYER),
        }
        samples = _walk_samples(1) + _walk_samples(2, calibration_valid=False)
        players = summarize_players(votes, {1: 0, 2: 0}, samples)
        teams = summarize_teams(players)
        assert teams[0].player_count == 2
        assert teams[0].players_without_valid_metrics == 1
        assert teams[0].total_distance_m == players[1].total_distance_m

    def test_replay_samples_are_excluded(self):
        votes = {1: _vote(1, PersonRole.OUTFIELD_PLAYER)}
        samples = _walk_samples(1)
        replay_flags = [True] * len(samples)
        players = summarize_players(votes, {1: 0}, samples, replay_flags=replay_flags)
        assert players[1].total_distance_m is None
        assert not players[1].physical_metrics_valid
