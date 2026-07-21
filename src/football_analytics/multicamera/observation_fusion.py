"""Fuse simultaneous observations of one identity into a single record.

When several cameras see the same global identity at (nearly) the same
instant, their observations are combined by confidence-weighted selection:
positions are averaged with detection-confidence weights, while categorical
attributes (team, jersey, role) are taken from the most confident source.
Disagreements between confident sources are never silently resolved — they
are surfaced as explicit conflict flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .local_tracking import LocalObservation, PlayerRole

_EPSILON = 1e-12


@dataclass(frozen=True)
class FusionConfig:
    # Sources below these confidences do not participate in attribute voting.
    minimum_team_confidence: float = 0.5
    minimum_jersey_confidence: float = 0.5


@dataclass(frozen=True)
class FusedObservation:
    """One identity at one instant, merged across cameras."""

    global_id: int
    reference_time_seconds: float
    pitch_xy_m: tuple[float, float] | None
    team_id: int | None
    team_confidence: float
    jersey_number: int | None
    jersey_confidence: float
    role: PlayerRole
    source_cameras: tuple[str, ...]
    observation_count: int
    confidence: float
    conflicts: tuple[str, ...] = field(default=())

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_id": self.global_id,
            "reference_time_seconds": self.reference_time_seconds,
            "pitch_xy_m": list(self.pitch_xy_m) if self.pitch_xy_m else None,
            "team_id": self.team_id,
            "team_confidence": self.team_confidence,
            "jersey_number": self.jersey_number,
            "jersey_confidence": self.jersey_confidence,
            "role": self.role.value,
            "source_cameras": list(self.source_cameras),
            "observation_count": self.observation_count,
            "confidence": self.confidence,
            "conflicts": list(self.conflicts),
        }


def fuse_observations(
    global_id: int,
    observations: Sequence[LocalObservation],
    config: FusionConfig | None = None,
) -> FusedObservation:
    """Merge concurrent observations of one identity with conflict detection."""
    if not observations:
        raise ValueError("at least one observation is required")
    cfg = config or FusionConfig()
    conflicts: list[str] = []

    times = np.asarray([obs.reference_time_seconds for obs in observations], dtype=np.float64)
    weights_all = np.asarray(
        [max(obs.detection_confidence, _EPSILON) for obs in observations], dtype=np.float64
    )
    fused_time = float(np.average(times, weights=weights_all))

    # Position: confidence-weighted mean of observations with valid pitch coords.
    positioned = [obs for obs in observations if obs.pitch_xy_m is not None]
    pitch_xy: tuple[float, float] | None = None
    if positioned:
        points = np.asarray([obs.pitch_xy_m for obs in positioned], dtype=np.float64)
        weights = np.asarray(
            [max(obs.detection_confidence, _EPSILON) for obs in positioned], dtype=np.float64
        )
        fused = np.average(points, axis=0, weights=weights)
        pitch_xy = (float(fused[0]), float(fused[1]))

    # Team: most confident source wins; confident disagreement is a conflict.
    team_votes = [
        obs
        for obs in observations
        if obs.team_id is not None and obs.team_confidence >= cfg.minimum_team_confidence
    ]
    team_id: int | None = None
    team_confidence = 0.0
    if team_votes:
        best_team = max(team_votes, key=lambda obs: obs.team_confidence)
        team_id = best_team.team_id
        team_confidence = best_team.team_confidence
        if len({obs.team_id for obs in team_votes}) > 1:
            conflicts.append("team_conflict")

    # Jersey: same policy as team.
    jersey_votes = [
        obs
        for obs in observations
        if obs.jersey_number is not None
        and obs.jersey_confidence >= cfg.minimum_jersey_confidence
    ]
    jersey_number: int | None = None
    jersey_confidence = 0.0
    if jersey_votes:
        best_jersey = max(jersey_votes, key=lambda obs: obs.jersey_confidence)
        jersey_number = best_jersey.jersey_number
        jersey_confidence = best_jersey.jersey_confidence
        if len({obs.jersey_number for obs in jersey_votes}) > 1:
            conflicts.append("jersey_conflict")

    # Role: any known role wins over unknown; two different known roles conflict.
    known_roles = {obs.role for obs in observations if obs.role != PlayerRole.UNKNOWN}
    role = PlayerRole.UNKNOWN
    if known_roles:
        if len(known_roles) > 1:
            conflicts.append("role_conflict")
        role = next(iter(sorted(known_roles, key=lambda item: item.value)))

    return FusedObservation(
        global_id=global_id,
        reference_time_seconds=fused_time,
        pitch_xy_m=pitch_xy,
        team_id=team_id,
        team_confidence=team_confidence,
        jersey_number=jersey_number,
        jersey_confidence=jersey_confidence,
        role=role,
        source_cameras=tuple(sorted({obs.camera_id for obs in observations})),
        observation_count=len(observations),
        confidence=float(np.mean([obs.detection_confidence for obs in observations])),
        conflicts=tuple(conflicts),
    )


def fuse_timeline(
    global_id: int,
    observations: Sequence[LocalObservation],
    bin_seconds: float = 0.2,
    config: FusionConfig | None = None,
) -> list[FusedObservation]:
    """Bucket one identity's observations into time bins and fuse each bin."""
    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive")
    buckets: dict[int, list[LocalObservation]] = {}
    for observation in observations:
        buckets.setdefault(
            int(observation.reference_time_seconds // bin_seconds), []
        ).append(observation)
    return [
        fuse_observations(global_id, buckets[index], config)
        for index in sorted(buckets)
    ]
