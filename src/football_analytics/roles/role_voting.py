"""Temporal voting over per-frame role observations.

Each frame contributes a weighted vote per role; the winner must both reach a
minimum share of the total vote mass and a minimum number of observations,
otherwise the track stays UNKNOWN_PERSON. Replay frames are excluded from
voting entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from football_analytics.roles.role_classifier import PersonRole, RoleObservation


@dataclass(frozen=True)
class RoleVotingConfig:
    #: Minimum non-replay observations before a role decision is made.
    min_observations: int = 5
    #: Winning role must hold at least this share of total vote mass.
    min_vote_share: float = 0.55
    #: Exponential decay applied to prior votes each new observation, so the
    #: vote adapts to genuine role changes without flip-flopping.
    decay: float = 0.98

    def __post_init__(self) -> None:
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if self.min_observations < 1:
            raise ValueError("min_observations must be >= 1")


@dataclass(frozen=True)
class RoleVote:
    """Outcome of temporal voting for one track."""

    track_id: int
    role: PersonRole
    vote_share: float
    observations: int
    votes: Mapping[PersonRole, float] = field(default_factory=dict)


class RoleVoter:
    """Accumulate per-frame role evidence into stable per-track decisions."""

    def __init__(self, config: RoleVotingConfig | None = None) -> None:
        self.config = config or RoleVotingConfig()
        self._votes: dict[int, dict[PersonRole, float]] = {}
        self._observations: dict[int, int] = {}

    def add_observation(self, observation: RoleObservation) -> RoleVote:
        if observation.replay:
            # Replay footage repeats moments and often shows different camera
            # angles; it must not influence role identity.
            return self.decide(observation.track_id)
        votes = self._votes.setdefault(observation.track_id, {})
        for role in votes:
            votes[role] *= self.config.decay
        for role, score in observation.scores.items():
            votes[role] = votes.get(role, 0.0) + score
        self._observations[observation.track_id] = (
            self._observations.get(observation.track_id, 0) + 1
        )
        return self.decide(observation.track_id)

    def decide(self, track_id: int) -> RoleVote:
        votes = self._votes.get(track_id, {})
        observations = self._observations.get(track_id, 0)
        total = sum(votes.values())
        if observations < self.config.min_observations or total <= 0.0:
            return RoleVote(track_id, PersonRole.UNKNOWN_PERSON, 0.0, observations, dict(votes))
        role, mass = max(votes.items(), key=lambda item: item[1])
        share = mass / total
        if share < self.config.min_vote_share:
            return RoleVote(track_id, PersonRole.UNKNOWN_PERSON, share, observations, dict(votes))
        return RoleVote(track_id, role, share, observations, dict(votes))

    def decide_all(self) -> dict[int, RoleVote]:
        track_ids = set(self._votes) | set(self._observations)
        return {track_id: self.decide(track_id) for track_id in sorted(track_ids)}

    def reset(self) -> None:
        self._votes.clear()
        self._observations.clear()


def vote_roles(
    observations: Iterable[RoleObservation],
    config: RoleVotingConfig | None = None,
) -> dict[int, RoleVote]:
    """Batch helper: run temporal voting over a full observation stream."""
    voter = RoleVoter(config)
    for observation in observations:
        voter.add_observation(observation)
    return voter.decide_all()
