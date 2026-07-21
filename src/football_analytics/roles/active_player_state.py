"""Per-track participation state with sticky goalkeeper identity.

Tracks move through participation states (active on pitch, on the bench,
substituted off, non-playing for officials/staff). The goalkeeper role is
*sticky*: once a track is confirmed as a goalkeeper it keeps the role while it
remains a team member, even during spells away from the penalty area (corner
kicks, sweeper-keeper play), unless sustained contrary evidence demotes it.
Officials are never active players and never carry a team id.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from football_analytics.roles.role_classifier import (
    PersonRole,
    is_official,
    is_team_member,
)
from football_analytics.roles.role_voting import RoleVote


class ParticipationState(str, Enum):
    ACTIVE = "active"
    BENCH = "bench"
    SUBSTITUTED_OFF = "substituted_off"
    NOT_PLAYING = "not_playing"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ActivePlayerRecord:
    track_id: int
    role: PersonRole
    state: ParticipationState
    team_id: int | None
    goalkeeper_sticky: bool = False

    @property
    def counts_toward_team_totals(self) -> bool:
        return (
            self.state is ParticipationState.ACTIVE
            and self.role in (PersonRole.OUTFIELD_PLAYER, PersonRole.GOALKEEPER)
            and self.team_id is not None
        )


@dataclass(frozen=True)
class ActivePlayerStateConfig:
    #: Vote share needed to first confirm a goalkeeper (makes the role sticky).
    goalkeeper_confirm_share: float = 0.60
    #: Consecutive contrary decisions needed to demote a sticky goalkeeper.
    goalkeeper_demotion_votes: int = 30


class ActivePlayerStateTracker:
    """Maintain participation state and sticky goalkeeper role per track."""

    def __init__(self, config: ActivePlayerStateConfig | None = None) -> None:
        self.config = config or ActivePlayerStateConfig()
        self._records: dict[int, ActivePlayerRecord] = {}
        self._gk_contrary: dict[int, int] = {}

    def update(
        self,
        vote: RoleVote,
        *,
        team_id: int | None = None,
        on_pitch: bool = True,
    ) -> ActivePlayerRecord:
        previous = self._records.get(vote.track_id)
        role = vote.role
        sticky = bool(previous and previous.goalkeeper_sticky)

        if sticky:
            if role is PersonRole.GOALKEEPER or is_team_member(role) or (
                role is PersonRole.UNKNOWN_PERSON
            ):
                # Team-member or inconclusive evidence: the keeper stays a
                # keeper even while roaming outside the box.
                self._gk_contrary[vote.track_id] = 0
                role = PersonRole.GOALKEEPER
            else:
                contrary = self._gk_contrary.get(vote.track_id, 0) + 1
                self._gk_contrary[vote.track_id] = contrary
                if contrary < self.config.goalkeeper_demotion_votes:
                    role = PersonRole.GOALKEEPER
                else:
                    sticky = False
        elif (
            role is PersonRole.GOALKEEPER
            and vote.vote_share >= self.config.goalkeeper_confirm_share
        ):
            sticky = True
            self._gk_contrary[vote.track_id] = 0

        if is_official(role) or role is PersonRole.STAFF:
            state = ParticipationState.NOT_PLAYING
            resolved_team: int | None = None
        elif role is PersonRole.SUBSTITUTE:
            state = ParticipationState.BENCH
            resolved_team = team_id if team_id is not None else _prior_team(previous)
        elif role in (PersonRole.OUTFIELD_PLAYER, PersonRole.GOALKEEPER):
            state = ParticipationState.ACTIVE if on_pitch else ParticipationState.BENCH
            resolved_team = team_id if team_id is not None else _prior_team(previous)
        else:
            state = ParticipationState.UNKNOWN
            resolved_team = team_id if team_id is not None else _prior_team(previous)

        # Substituted-off is terminal until an explicit correction.
        if previous and previous.state is ParticipationState.SUBSTITUTED_OFF:
            state = ParticipationState.SUBSTITUTED_OFF

        record = ActivePlayerRecord(
            track_id=vote.track_id,
            role=role,
            state=state,
            team_id=resolved_team,
            goalkeeper_sticky=sticky,
        )
        self._records[vote.track_id] = record
        return record

    def mark_substituted_on(self, track_id: int) -> ActivePlayerRecord:
        record = self._require(track_id)
        record = replace(record, state=ParticipationState.ACTIVE)
        if record.role is PersonRole.SUBSTITUTE:
            record = replace(record, role=PersonRole.OUTFIELD_PLAYER)
        self._records[track_id] = record
        return record

    def mark_substituted_off(self, track_id: int) -> ActivePlayerRecord:
        record = replace(self._require(track_id), state=ParticipationState.SUBSTITUTED_OFF)
        self._records[track_id] = record
        return record

    def get(self, track_id: int) -> ActivePlayerRecord | None:
        return self._records.get(track_id)

    def active_players(self, team_id: int | None = None) -> list[ActivePlayerRecord]:
        return [
            record
            for record in self._records.values()
            if record.counts_toward_team_totals
            and (team_id is None or record.team_id == team_id)
        ]

    def _require(self, track_id: int) -> ActivePlayerRecord:
        record = self._records.get(track_id)
        if record is None:
            raise KeyError(f"unknown track_id {track_id}")
        return record


def _prior_team(previous: ActivePlayerRecord | None) -> int | None:
    return previous.team_id if previous else None
