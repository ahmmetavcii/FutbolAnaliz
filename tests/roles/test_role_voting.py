from __future__ import annotations

from football_analytics.roles import (
    PersonRole,
    RoleObservation,
    RoleVoter,
    RoleVotingConfig,
    vote_roles,
)


def obs(track_id: int, role: PersonRole, score: float, ts: float, replay: bool = False):
    return RoleObservation(
        track_id=track_id,
        timestamp_ms=ts,
        scores={role: score},
        replay=replay,
    )


class TestRoleVoting:
    def test_consistent_evidence_wins(self):
        observations = [obs(1, PersonRole.REFEREE, 0.8, float(i)) for i in range(10)]
        votes = vote_roles(observations)
        assert votes[1].role is PersonRole.REFEREE
        assert votes[1].vote_share > 0.9

    def test_too_few_observations_stays_unknown(self):
        config = RoleVotingConfig(min_observations=5)
        observations = [obs(1, PersonRole.GOALKEEPER, 0.9, float(i)) for i in range(4)]
        votes = vote_roles(observations, config)
        assert votes[1].role is PersonRole.UNKNOWN_PERSON

    def test_replay_observations_do_not_vote(self):
        config = RoleVotingConfig(min_observations=5)
        live = [obs(1, PersonRole.OUTFIELD_PLAYER, 0.8, float(i)) for i in range(4)]
        replays = [obs(1, PersonRole.OUTFIELD_PLAYER, 0.9, 100.0 + i, replay=True) for i in range(10)]
        votes = vote_roles(live + replays, config)
        # Replay evidence must not push the track over min_observations.
        assert votes[1].observations == 4
        assert votes[1].role is PersonRole.UNKNOWN_PERSON

    def test_ambiguous_evidence_stays_unknown(self):
        observations = []
        for i in range(20):
            role = PersonRole.REFEREE if i % 2 == 0 else PersonRole.OUTFIELD_PLAYER
            observations.append(obs(1, role, 0.8, float(i)))
        votes = vote_roles(observations)
        assert votes[1].role is PersonRole.UNKNOWN_PERSON
        assert votes[1].vote_share < 0.55

    def test_single_noisy_frame_does_not_flip_decision(self):
        voter = RoleVoter()
        for i in range(30):
            voter.add_observation(obs(1, PersonRole.GOALKEEPER, 0.8, float(i)))
        decision = voter.add_observation(obs(1, PersonRole.REFEREE, 0.9, 30.0))
        assert decision.role is PersonRole.GOALKEEPER

    def test_sustained_change_eventually_adapts(self):
        voter = RoleVoter(RoleVotingConfig(decay=0.9))
        for i in range(10):
            voter.add_observation(obs(1, PersonRole.SUBSTITUTE, 0.8, float(i)))
        for i in range(60):
            voter.add_observation(obs(1, PersonRole.OUTFIELD_PLAYER, 0.8, 10.0 + i))
        assert voter.decide(1).role is PersonRole.OUTFIELD_PLAYER

    def test_tracks_are_independent(self):
        observations = [obs(1, PersonRole.REFEREE, 0.8, float(i)) for i in range(10)]
        observations += [obs(2, PersonRole.OUTFIELD_PLAYER, 0.8, float(i)) for i in range(10)]
        votes = vote_roles(observations)
        assert votes[1].role is PersonRole.REFEREE
        assert votes[2].role is PersonRole.OUTFIELD_PLAYER
