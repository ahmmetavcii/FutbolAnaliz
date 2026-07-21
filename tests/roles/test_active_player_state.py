from __future__ import annotations

from football_analytics.roles import (
    ActivePlayerStateConfig,
    ActivePlayerStateTracker,
    ParticipationState,
    PersonRole,
    RoleVote,
)


def vote(track_id: int, role: PersonRole, share: float = 0.9, observations: int = 20):
    return RoleVote(track_id=track_id, role=role, vote_share=share, observations=observations)


class TestGoalkeeperStickiness:
    def test_confirmed_goalkeeper_stays_goalkeeper_during_unknown_spells(self):
        tracker = ActivePlayerStateTracker()
        tracker.update(vote(1, PersonRole.GOALKEEPER), team_id=0)
        # Keeper joins a corner: penalty-area evidence vanishes.
        record = tracker.update(vote(1, PersonRole.UNKNOWN_PERSON, share=0.0))
        assert record.role is PersonRole.GOALKEEPER
        assert record.goalkeeper_sticky

    def test_goalkeeper_survives_outfield_looking_evidence(self):
        tracker = ActivePlayerStateTracker()
        tracker.update(vote(1, PersonRole.GOALKEEPER), team_id=0)
        record = tracker.update(vote(1, PersonRole.OUTFIELD_PLAYER, share=0.8), team_id=0)
        assert record.role is PersonRole.GOALKEEPER

    def test_sustained_official_evidence_eventually_demotes(self):
        config = ActivePlayerStateConfig(goalkeeper_demotion_votes=3)
        tracker = ActivePlayerStateTracker(config)
        tracker.update(vote(1, PersonRole.GOALKEEPER), team_id=0)
        for _ in range(2):
            record = tracker.update(vote(1, PersonRole.REFEREE, share=0.9))
            assert record.role is PersonRole.GOALKEEPER
        record = tracker.update(vote(1, PersonRole.REFEREE, share=0.9))
        assert record.role is PersonRole.REFEREE
        assert not record.goalkeeper_sticky

    def test_low_share_goalkeeper_vote_is_not_sticky(self):
        tracker = ActivePlayerStateTracker()
        record = tracker.update(vote(1, PersonRole.GOALKEEPER, share=0.4), team_id=0)
        assert not record.goalkeeper_sticky

    def test_sticky_goalkeeper_keeps_team_membership(self):
        tracker = ActivePlayerStateTracker()
        tracker.update(vote(1, PersonRole.GOALKEEPER), team_id=0)
        record = tracker.update(vote(1, PersonRole.UNKNOWN_PERSON, share=0.0))
        assert record.team_id == 0
        assert record.counts_toward_team_totals


class TestOfficialsExclusion:
    def test_officials_never_carry_a_team_or_count(self):
        tracker = ActivePlayerStateTracker()
        for role in (
            PersonRole.REFEREE,
            PersonRole.ASSISTANT_REFEREE,
            PersonRole.FOURTH_OFFICIAL,
        ):
            record = tracker.update(vote(hash(role) % 1000 + 1, role), team_id=0)
            assert record.team_id is None
            assert record.state is ParticipationState.NOT_PLAYING
            assert not record.counts_toward_team_totals

    def test_active_players_excludes_officials_and_bench(self):
        tracker = ActivePlayerStateTracker()
        tracker.update(vote(1, PersonRole.OUTFIELD_PLAYER), team_id=0)
        tracker.update(vote(2, PersonRole.GOALKEEPER), team_id=0)
        tracker.update(vote(3, PersonRole.REFEREE))
        tracker.update(vote(4, PersonRole.SUBSTITUTE), team_id=0)
        tracker.update(vote(5, PersonRole.STAFF))
        active = tracker.active_players(team_id=0)
        assert sorted(r.track_id for r in active) == [1, 2]


class TestSubstitutionTransitions:
    def test_substitute_coming_on_becomes_active_outfield(self):
        tracker = ActivePlayerStateTracker()
        tracker.update(vote(4, PersonRole.SUBSTITUTE), team_id=1)
        record = tracker.mark_substituted_on(4)
        assert record.state is ParticipationState.ACTIVE
        assert record.role is PersonRole.OUTFIELD_PLAYER
        assert record.counts_toward_team_totals

    def test_substituted_off_is_terminal_for_updates(self):
        tracker = ActivePlayerStateTracker()
        tracker.update(vote(7, PersonRole.OUTFIELD_PLAYER), team_id=0)
        tracker.mark_substituted_off(7)
        record = tracker.update(vote(7, PersonRole.OUTFIELD_PLAYER), team_id=0)
        assert record.state is ParticipationState.SUBSTITUTED_OFF
        assert not record.counts_toward_team_totals
