"""Temporal player-role identity independent of shirt-team clustering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PlayerRole(str, Enum):
    OUTFIELD = "outfield"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RoleAssignment:
    role: PlayerRole
    confidence: float
    temporal_consistency: float

    @property
    def excluded_from_team_clustering(self) -> bool:
        return self.role is PlayerRole.REFEREE


class RoleIdentityTracker:
    """Smooth role evidence while making goalkeeper identity sticky."""

    def __init__(
        self,
        *,
        referee_threshold: float = 0.65,
        goalkeeper_threshold: float = 0.60,
        retention_threshold: float = 0.30,
        smoothing: float = 0.75,
    ) -> None:
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.referee_threshold = referee_threshold
        self.goalkeeper_threshold = goalkeeper_threshold
        self.retention_threshold = retention_threshold
        self.smoothing = smoothing
        self._scores: dict[int, tuple[float, float, int, PlayerRole]] = {}

    def update(
        self,
        track_id: int,
        *,
        referee_confidence: float = 0.0,
        goalkeeper_confidence: float = 0.0,
        is_player: bool = True,
    ) -> RoleAssignment:
        referee_confidence = _probability(referee_confidence)
        goalkeeper_confidence = _probability(goalkeeper_confidence)
        previous = self._scores.get(track_id)
        if previous is None:
            referee, goalkeeper, observations, previous_role = (
                referee_confidence,
                goalkeeper_confidence,
                1,
                PlayerRole.UNKNOWN,
            )
        else:
            old_ref, old_gk, observations, previous_role = previous
            alpha = self.smoothing
            referee = alpha * old_ref + (1.0 - alpha) * referee_confidence
            goalkeeper = alpha * old_gk + (1.0 - alpha) * goalkeeper_confidence
            observations += 1

        if not is_player:
            role, confidence = PlayerRole.UNKNOWN, 0.0
        elif referee >= self.referee_threshold:
            role, confidence = PlayerRole.REFEREE, referee
        elif goalkeeper >= self.goalkeeper_threshold or (
            previous_role is PlayerRole.GOALKEEPER and goalkeeper >= self.retention_threshold
        ):
            role, confidence = PlayerRole.GOALKEEPER, goalkeeper
        elif max(referee, goalkeeper) < min(self.retention_threshold, 0.5):
            role, confidence = PlayerRole.OUTFIELD, 1.0 - max(referee, goalkeeper)
        else:
            role, confidence = PlayerRole.UNKNOWN, 0.0

        self._scores[track_id] = (referee, goalkeeper, observations, role)
        consistency = 1.0 if previous_role in (PlayerRole.UNKNOWN, role) else 0.5
        return RoleAssignment(role, confidence, consistency)

    def get(self, track_id: int) -> RoleAssignment:
        state = self._scores.get(track_id)
        if state is None:
            return RoleAssignment(PlayerRole.UNKNOWN, 0.0, 0.0)
        referee, goalkeeper, _, role = state
        confidence = {
            PlayerRole.REFEREE: referee,
            PlayerRole.GOALKEEPER: goalkeeper,
            PlayerRole.OUTFIELD: 1.0 - max(referee, goalkeeper),
            PlayerRole.UNKNOWN: 0.0,
        }[role]
        return RoleAssignment(role, confidence, 1.0)

    def reset_scene(self) -> None:
        self._scores.clear()

    reset = reset_scene


def _probability(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return value
