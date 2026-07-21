"""Cue-based scoring for matching local tracks across cameras.

A candidate observation is compared against the running summary of a global
identity using six cues: appearance (re-id embedding), team, jersey number,
pitch position feasibility, temporal proximity, and role compatibility.

Two safety properties are enforced at this layer:

- a jersey number alone is never sufficient evidence
  (:attr:`MatchScore.supported_by_non_jersey` stays False), and
- physically impossible matches — the identity was seen almost simultaneously
  at a pitch position unreachable at plausible player speed — are hard-rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .local_tracking import LocalObservation, PlayerRole, cosine_similarity


@dataclass(frozen=True)
class ReidMatchConfig:
    weight_reid: float = 0.35
    weight_team: float = 0.15
    weight_jersey: float = 0.20
    weight_position: float = 0.20
    weight_time: float = 0.05
    weight_role: float = 0.05

    accept_threshold: float = 0.60
    unresolved_threshold: float = 0.40

    minimum_team_confidence: float = 0.50
    minimum_jersey_confidence: float = 0.50
    minimum_reid_similarity: float = 0.30

    max_player_speed_mps: float = 12.0
    position_tolerance_m: float = 5.0
    position_scale_m: float = 10.0
    time_scale_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not 0.0 < self.accept_threshold <= 1.0:
            raise ValueError("accept_threshold must be in (0, 1]")
        if not 0.0 <= self.unresolved_threshold <= self.accept_threshold:
            raise ValueError("unresolved_threshold must be in [0, accept_threshold]")


@dataclass(frozen=True)
class IdentitySnapshot:
    """Running summary of a global identity, as seen by the matcher."""

    global_id: int
    team_id: int | None
    team_confidence: float
    jersey_number: int | None
    jersey_confidence: float
    role: PlayerRole
    embedding: NDArray[np.float64] | None
    last_time_seconds: float | None
    last_pitch_xy_m: tuple[float, float] | None
    last_camera_id: str | None


@dataclass(frozen=True)
class CueScores:
    """Per-cue scores in [0, 1]; None when the cue is unavailable."""

    reid: float | None = None
    team: float | None = None
    jersey: float | None = None
    position: float | None = None
    time: float | None = None
    role: float | None = None


@dataclass(frozen=True)
class MatchScore:
    identity_global_id: int
    score: float
    cues: CueScores
    hard_reject: bool = False
    reject_reason: str | None = None
    supported_by_non_jersey: bool = False

    @property
    def acceptable(self) -> bool:
        return not self.hard_reject and self.supported_by_non_jersey


def score_candidate(
    observation: LocalObservation,
    identity: IdentitySnapshot,
    config: ReidMatchConfig | None = None,
) -> MatchScore:
    """Score how well ``observation`` matches ``identity``.

    Hard rejections (role conflict, confident team conflict, impossible
    position) zero the score regardless of other cues.
    """
    cfg = config or ReidMatchConfig()

    def rejected(reason: str, cues: CueScores) -> MatchScore:
        return MatchScore(
            identity_global_id=identity.global_id,
            score=0.0,
            cues=cues,
            hard_reject=True,
            reject_reason=reason,
        )

    # Role compatibility.
    role_score: float | None = None
    if observation.role != PlayerRole.UNKNOWN and identity.role != PlayerRole.UNKNOWN:
        if observation.role != identity.role:
            return rejected(
                f"role conflict: {observation.role.value} vs {identity.role.value}",
                CueScores(role=0.0),
            )
        role_score = 1.0

    # Team agreement.
    team_score: float | None = None
    if (
        observation.team_id is not None
        and identity.team_id is not None
        and observation.team_confidence >= cfg.minimum_team_confidence
        and identity.team_confidence >= cfg.minimum_team_confidence
    ):
        if observation.team_id != identity.team_id:
            return rejected(
                f"team conflict: {observation.team_id} vs {identity.team_id}",
                CueScores(role=role_score, team=0.0),
            )
        team_score = min(observation.team_confidence, identity.team_confidence)

    # Position feasibility on the pitch.
    position_score: float | None = None
    if (
        observation.pitch_xy_m is not None
        and identity.last_pitch_xy_m is not None
        and identity.last_time_seconds is not None
    ):
        dt = abs(observation.reference_time_seconds - identity.last_time_seconds)
        distance = float(
            np.hypot(
                observation.pitch_xy_m[0] - identity.last_pitch_xy_m[0],
                observation.pitch_xy_m[1] - identity.last_pitch_xy_m[1],
            )
        )
        feasible_radius = cfg.max_player_speed_mps * dt + cfg.position_tolerance_m
        if distance > feasible_radius:
            return rejected(
                f"impossible position: {distance:.1f} m apart within {dt:.2f} s",
                CueScores(role=role_score, team=team_score, position=0.0),
            )
        position_score = float(np.exp(-distance / cfg.position_scale_m))

    # Temporal proximity.
    time_score: float | None = None
    if identity.last_time_seconds is not None:
        dt = abs(observation.reference_time_seconds - identity.last_time_seconds)
        time_score = float(np.exp(-dt / cfg.time_scale_seconds))

    # Appearance embedding.
    reid_score: float | None = None
    obs_embedding = observation.embedding_array()
    if obs_embedding is not None and identity.embedding is not None:
        similarity = cosine_similarity(obs_embedding, identity.embedding)
        reid_score = float(np.clip(similarity, 0.0, 1.0))
        if similarity < cfg.minimum_reid_similarity:
            # When both sides have embeddings, a weak appearance match is a
            # hard reject — otherwise same-kit teammates over-merge.
            return rejected(
                f"reid mismatch: similarity {similarity:.3f} < {cfg.minimum_reid_similarity:.3f}",
                CueScores(
                    role=role_score,
                    team=team_score,
                    position=position_score,
                    time=time_score,
                    reid=0.0,
                ),
            )

    # Jersey number.
    jersey_score: float | None = None
    if (
        observation.jersey_number is not None
        and identity.jersey_number is not None
        and observation.jersey_confidence >= cfg.minimum_jersey_confidence
        and identity.jersey_confidence >= cfg.minimum_jersey_confidence
    ):
        jersey_score = (
            min(observation.jersey_confidence, identity.jersey_confidence)
            if observation.jersey_number == identity.jersey_number
            else 0.0
        )

    cues = CueScores(
        reid=reid_score,
        team=team_score,
        jersey=jersey_score,
        position=position_score,
        time=time_score,
        role=role_score,
    )

    weighted: list[tuple[float, float]] = []
    for weight, value in (
        (cfg.weight_reid, reid_score),
        (cfg.weight_team, team_score),
        (cfg.weight_jersey, jersey_score),
        (cfg.weight_position, position_score),
        (cfg.weight_time, time_score),
        (cfg.weight_role, role_score),
    ):
        if value is not None:
            weighted.append((weight, value))
    total_weight = sum(weight for weight, _ in weighted)
    score = (
        sum(weight * value for weight, value in weighted) / total_weight
        if total_weight > 0
        else 0.0
    )

    # Jersey agreement can only corroborate; some other positive cue must exist.
    supported = any(
        value is not None and value > 0.0
        for value in (reid_score, team_score, position_score)
    )

    return MatchScore(
        identity_global_id=identity.global_id,
        score=float(np.clip(score, 0.0, 1.0)),
        cues=cues,
        supported_by_non_jersey=supported,
    )
