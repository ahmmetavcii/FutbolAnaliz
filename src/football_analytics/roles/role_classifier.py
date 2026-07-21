"""Person-role taxonomy and per-frame role evidence fusion.

The classifier produces *evidence*, not truth: each frame yields a
:class:`RoleObservation` with per-role scores in ``[0, 1]``. Temporal voting
(:mod:`football_analytics.roles.role_voting`) turns those observations into a
stable decision. Officials are a first-class concept so that downstream
aggregation can guarantee they are never mixed into team or player totals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from football_analytics.roles.goalkeeper_classifier import (
    GoalkeeperClassifier,
    GoalkeeperFeatures,
)
from football_analytics.roles.referee_classifier import (
    RefereeClassifier,
    RefereeFeatures,
)


class PersonRole(str, Enum):
    OUTFIELD_PLAYER = "outfield_player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    ASSISTANT_REFEREE = "assistant_referee"
    FOURTH_OFFICIAL = "fourth_official"
    SUBSTITUTE = "substitute"
    STAFF = "staff"
    UNKNOWN_PERSON = "unknown_person"


OFFICIAL_ROLES: frozenset[PersonRole] = frozenset(
    {PersonRole.REFEREE, PersonRole.ASSISTANT_REFEREE, PersonRole.FOURTH_OFFICIAL}
)

#: Roles that belong to a team roster (the goalkeeper is a team member).
TEAM_ROLES: frozenset[PersonRole] = frozenset(
    {PersonRole.OUTFIELD_PLAYER, PersonRole.GOALKEEPER, PersonRole.SUBSTITUTE}
)

#: Roles whose on-pitch activity counts toward team/player totals.
COUNTABLE_ROLES: frozenset[PersonRole] = frozenset(
    {PersonRole.OUTFIELD_PLAYER, PersonRole.GOALKEEPER}
)


def is_official(role: PersonRole) -> bool:
    return role in OFFICIAL_ROLES


def is_team_member(role: PersonRole) -> bool:
    return role in TEAM_ROLES


def counts_toward_team_totals(role: PersonRole) -> bool:
    """Officials, staff, bench substitutes and unknowns never count."""
    return role in COUNTABLE_ROLES


@dataclass(frozen=True)
class PersonFrameFeatures:
    """Per-frame appearance/position features for one tracked person.

    All similarity/occupancy values are in ``[0, 1]``. ``None`` means the
    feature could not be measured this frame and contributes no evidence.
    """

    track_id: int
    timestamp_ms: float
    on_pitch: bool = True
    in_technical_area: bool = False
    in_bench_area: bool = False
    near_touchline: bool = False
    replay: bool = False
    kit_similarity_officials: float | None = None
    kit_similarity_team0: float | None = None
    kit_similarity_team1: float | None = None
    goalkeeper_kit_distinctiveness: float | None = None
    own_penalty_area_occupancy: float | None = None
    is_deepest_teammate: bool = False
    team_id: int | None = None


@dataclass(frozen=True)
class RoleObservation:
    """Per-frame role evidence emitted by :class:`RoleClassifier`."""

    track_id: int
    timestamp_ms: float
    scores: Mapping[PersonRole, float]
    replay: bool = False

    def __post_init__(self) -> None:
        for role, score in self.scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"score for {role} must be in [0, 1], got {score}")
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))

    @property
    def best_role(self) -> PersonRole:
        if not self.scores:
            return PersonRole.UNKNOWN_PERSON
        role, score = max(self.scores.items(), key=lambda item: item[1])
        return role if score > 0.0 else PersonRole.UNKNOWN_PERSON

    @property
    def best_score(self) -> float:
        return max(self.scores.values(), default=0.0)


@dataclass(frozen=True)
class RoleClassifierConfig:
    #: Below this best score the frame is labelled UNKNOWN_PERSON.
    min_score: float = 0.30
    #: Team-kit similarity required to call someone an outfield player.
    min_team_kit_similarity: float = 0.50
    referee: RefereeClassifier = field(default_factory=RefereeClassifier)
    goalkeeper: GoalkeeperClassifier = field(default_factory=GoalkeeperClassifier)


class RoleClassifier:
    """Fuse referee/goalkeeper/appearance evidence into per-frame role scores.

    Deliberately conservative: when signals conflict or are missing, scores
    stay low and the frame contributes UNKNOWN_PERSON evidence instead of a
    confident wrong label.
    """

    def __init__(self, config: RoleClassifierConfig | None = None) -> None:
        self.config = config or RoleClassifierConfig()

    def classify_frame(self, features: PersonFrameFeatures) -> RoleObservation:
        cfg = self.config
        scores: dict[PersonRole, float] = {role: 0.0 for role in PersonRole}
        scores.pop(PersonRole.UNKNOWN_PERSON)

        official_scores = cfg.referee.score(
            RefereeFeatures(
                kit_similarity_officials=features.kit_similarity_officials,
                kit_similarity_team0=features.kit_similarity_team0,
                kit_similarity_team1=features.kit_similarity_team1,
                on_pitch=features.on_pitch,
                near_touchline=features.near_touchline,
                in_technical_area=features.in_technical_area,
            )
        )
        scores[PersonRole.REFEREE] = official_scores.referee
        scores[PersonRole.ASSISTANT_REFEREE] = official_scores.assistant_referee
        scores[PersonRole.FOURTH_OFFICIAL] = official_scores.fourth_official

        gk_score = cfg.goalkeeper.score(
            GoalkeeperFeatures(
                kit_distinct_from_teammates=features.goalkeeper_kit_distinctiveness,
                own_penalty_area_occupancy=features.own_penalty_area_occupancy,
                is_deepest_teammate=features.is_deepest_teammate,
                team_id=features.team_id,
                on_pitch=features.on_pitch,
            )
        )
        scores[PersonRole.GOALKEEPER] = gk_score

        team_kit = max(
            features.kit_similarity_team0 or 0.0,
            features.kit_similarity_team1 or 0.0,
        )
        official_best = max(
            official_scores.referee,
            official_scores.assistant_referee,
            official_scores.fourth_official,
        )
        if team_kit >= cfg.min_team_kit_similarity and official_best < team_kit:
            player_score = team_kit * (1.0 - gk_score)
            if features.on_pitch:
                scores[PersonRole.OUTFIELD_PLAYER] = player_score
            elif features.in_bench_area:
                scores[PersonRole.SUBSTITUTE] = player_score

        if not features.on_pitch and features.in_technical_area and team_kit < 0.3:
            # Non-playing kit in the technical area and no official evidence.
            scores[PersonRole.STAFF] = max(
                scores[PersonRole.STAFF], 0.5 * (1.0 - official_best)
            )

        best = max(scores.values(), default=0.0)
        if best < cfg.min_score:
            scores = {}
        return RoleObservation(
            track_id=features.track_id,
            timestamp_ms=features.timestamp_ms,
            scores={r: s for r, s in scores.items() if s > 0.0},
            replay=features.replay,
        )
