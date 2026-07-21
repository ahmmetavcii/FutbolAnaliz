"""Temporal shirt-colour team identity using kit color-family descriptors."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.cluster import KMeans

from football_analytics.analytics.kit_descriptor import (
    FAMILY_ORDER,
    colored_score,
    is_dark_kit_fractions,
    kit_feature_from_frame,
    white_score,
)
from football_analytics.analytics.role_identity import PlayerRole
from football_analytics.geometry.bbox import BBox

FEATURE_DIM = len(FAMILY_ORDER)


@dataclass(frozen=True, slots=True)
class ColorSample:
    track_id: int
    feature: np.ndarray
    valid_pixel_fraction: float


@dataclass(frozen=True, slots=True)
class TeamAssignment:
    team_id: int | None
    confidence: float
    temporal_consistency: float
    role: PlayerRole = PlayerRole.OUTFIELD

    @property
    def is_unknown(self) -> bool:
        return self.team_id is None


def sample_upper_torso(
    frame_bgr: np.ndarray,
    bbox: BBox | Sequence[float],
    *,
    min_pixels: int = 24,
    top_ratio: float = 0.15,
    bottom_ratio: float = 0.65,
    side_inset: float = 0.20,
) -> tuple[np.ndarray | None, float]:
    """Extract kit color-family fractions from the center torso.

    ``top_ratio`` / ``bottom_ratio`` / ``side_inset`` map to the Stage 5B
    normalized torso ROI used by the reference kit descriptor.
    """
    x_min = float(side_inset)
    x_max = 1.0 - float(side_inset)
    feature, fraction = kit_feature_from_frame(
        frame_bgr,
        bbox,
        x_min=x_min,
        x_max=x_max,
        y_min=float(top_ratio),
        y_max=float(bottom_ratio),
    )
    if feature is None:
        return None, fraction
    # min_pixels approximated via useful fraction on a typical crop.
    if fraction <= 0.0 and min_pixels > 0:
        return None, fraction
    return feature, fraction


def robust_pool(features: Iterable[np.ndarray], *, z_threshold: float = 3.5) -> np.ndarray | None:
    """Pool temporal samples after multivariate MAD outlier rejection."""
    rows = [np.asarray(feature, dtype=np.float32).reshape(-1) for feature in features]
    if not rows:
        return None
    matrix = np.stack(rows)
    if matrix.shape[0] < 3:
        return np.median(matrix, axis=0).astype(np.float32)
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-3)
    score = np.sqrt(np.mean(np.square((matrix - median) / scale), axis=1))
    kept = matrix[score <= z_threshold]
    if kept.size == 0:
        kept = matrix
    return np.median(kept, axis=0).astype(np.float32)


class TeamIdentityAssigner:
    """Build two scene-local teams from temporal per-track kit features."""

    def __init__(
        self,
        *,
        min_samples: int = 8,
        min_tracks: int = 4,
        history_size: int = 20,
        min_valid_pixel_fraction: float = 0.05,
        unknown_confidence_threshold: float = 0.55,
        random_state: int = 0,
        exclude_dark_kits: bool = True,
    ) -> None:
        if min_samples < 2 or min_tracks < 2:
            raise ValueError("two-team clustering needs at least two samples and tracks")
        self.min_samples = min_samples
        self.min_tracks = min_tracks
        self.history_size = history_size
        self.min_valid_pixel_fraction = min_valid_pixel_fraction
        self.unknown_confidence_threshold = unknown_confidence_threshold
        self.random_state = random_state
        self.exclude_dark_kits = exclude_dark_kits
        self._samples: dict[int, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self._centres: np.ndarray | None = None
        self._scale: float = 1.0
        self._history: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=12))
        self._roles: dict[int, PlayerRole] = {}
        self._locked_team: dict[int, int] = {}
        self._switch_streak: dict[int, int] = defaultdict(int)
        self._dark_tracks: set[int] = set()

    @property
    def fitted(self) -> bool:
        return self._centres is not None

    @property
    def team_centres(self) -> np.ndarray | None:
        return None if self._centres is None else self._centres.copy()

    def add_sample(
        self,
        track_id: int,
        feature: np.ndarray,
        *,
        role: PlayerRole | str = PlayerRole.OUTFIELD,
        valid_pixel_fraction: float = 1.0,
    ) -> bool:
        parsed_role = PlayerRole(role)
        self._roles[track_id] = parsed_role
        if parsed_role is PlayerRole.REFEREE:
            return False
        if valid_pixel_fraction < self.min_valid_pixel_fraction:
            return False
        row = np.asarray(feature, dtype=np.float32).reshape(-1)
        if row.size != FEATURE_DIM or not np.all(np.isfinite(row)):
            raise ValueError(
                f"kit feature must contain {FEATURE_DIM} finite family fractions"
            )
        self._samples[track_id].append(row)
        return True

    def observe(
        self,
        track_id: int,
        frame_bgr: np.ndarray,
        bbox: BBox | Sequence[float],
        *,
        role: PlayerRole | str = PlayerRole.OUTFIELD,
    ) -> TeamAssignment:
        parsed_role = PlayerRole(role)
        self._roles[track_id] = parsed_role
        if parsed_role is PlayerRole.REFEREE:
            return TeamAssignment(None, 0.0, 0.0, parsed_role)
        feature, fraction = sample_upper_torso(frame_bgr, bbox)
        if feature is not None:
            self.add_sample(
                track_id, feature, role=parsed_role, valid_pixel_fraction=fraction
            )
        if not self.fitted:
            self.fit()
        return self.assign(track_id, role=parsed_role)

    update = observe

    def fit(self) -> bool:
        pooled: list[tuple[int, np.ndarray]] = []
        for track_id, samples in self._samples.items():
            if self._roles.get(track_id) is PlayerRole.REFEREE:
                continue
            row = robust_pool(samples)
            if row is None:
                continue
            pooled.append((track_id, row))

        sample_count = sum(len(samples) for samples in self._samples.values())
        if len(pooled) < self.min_tracks or sample_count < self.min_samples:
            return False

        self._dark_tracks = {
            track_id
            for track_id, row in pooled
            if is_dark_kit_fractions(row)
        }
        train = [
            (track_id, row)
            for track_id, row in pooled
            if not (self.exclude_dark_kits and track_id in self._dark_tracks)
        ]
        if len(train) < self.min_tracks:
            train = pooled
            self._dark_tracks = set()

        matrix = np.stack([row for _, row in train])
        self._centres, self._scale = self._fit_two_centres(matrix)
        return True

    def _fit_two_centres(self, matrix: np.ndarray) -> tuple[np.ndarray, float]:
        model = KMeans(n_clusters=2, n_init=20, random_state=self.random_state).fit(
            matrix
        )
        centres = model.cluster_centers_.astype(np.float32)
        # team_0 = whiter / greyer kits, team_1 = more coloured (yellow etc.).
        score0 = white_score(centres[0]) - colored_score(centres[0])
        score1 = white_score(centres[1]) - colored_score(centres[1])
        if score1 > score0:
            centres = centres[::-1]
        distances = np.linalg.norm(matrix[:, None, :] - centres[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        residuals = distances[np.arange(len(matrix)), labels]
        scale = max(float(np.median(residuals)) * 2.5, 0.08)
        return centres, scale

    def assign(
        self, track_id: int, *, role: PlayerRole | str | None = None
    ) -> TeamAssignment:
        parsed_role = PlayerRole(role) if role is not None else self._roles.get(
            track_id, PlayerRole.OUTFIELD
        )
        if parsed_role is PlayerRole.REFEREE or self._centres is None:
            return TeamAssignment(None, 0.0, 0.0, parsed_role)

        feature = robust_pool(self._samples.get(track_id, ()))
        if feature is None:
            return TeamAssignment(None, 0.0, 0.0, parsed_role)

        if self.exclude_dark_kits and (
            track_id in self._dark_tracks or is_dark_kit_fractions(feature)
        ):
            return TeamAssignment(None, 0.0, 0.0, PlayerRole.REFEREE)

        distances = np.linalg.norm(self._centres - feature, axis=1)
        winner = int(np.argmin(distances))
        nearest, other = float(distances[winner]), float(distances[1 - winner])
        separation = (other - nearest) / max(other, 1e-6)
        proximity = float(np.exp(-nearest / self._scale))
        # Soft prior from white vs coloured family scores.
        white = white_score(feature)
        color = colored_score(feature)
        prior = 0.0
        if abs(white - color) > 0.08:
            prior_team = 0 if white > color else 1
            prior = 0.15 if prior_team == winner else -0.10
        confidence = float(np.clip(0.5 * separation + 0.5 * proximity + prior, 0.0, 1.0))

        if confidence < self.unknown_confidence_threshold:
            locked = self._locked_team.get(track_id)
            if locked is not None:
                return TeamAssignment(
                    locked,
                    max(confidence, self.unknown_confidence_threshold),
                    1.0,
                    parsed_role,
                )
            return TeamAssignment(
                None, confidence, self._consistency(track_id, winner), parsed_role
            )

        locked = self._locked_team.get(track_id)
        if locked is not None and winner != locked:
            self._switch_streak[track_id] += 1
            if self._switch_streak[track_id] < 8 or confidence < max(
                0.75, self.unknown_confidence_threshold + 0.15
            ):
                winner = locked
                confidence = max(confidence, 0.65)
            else:
                self._locked_team[track_id] = winner
                self._switch_streak[track_id] = 0
        else:
            self._switch_streak[track_id] = 0
            if track_id not in self._locked_team and self._consistency(track_id, winner) >= 0.6:
                self._locked_team[track_id] = winner

        self._history[track_id].append(winner)
        if track_id not in self._locked_team and len(self._history[track_id]) >= 3:
            if self._consistency(track_id, winner) >= 0.66:
                self._locked_team[track_id] = winner
        return TeamAssignment(winner, confidence, self._consistency(track_id, winner), parsed_role)

    def _consistency(self, track_id: int, team_id: int) -> float:
        history = self._history.get(track_id)
        if not history:
            return 0.0
        return sum(value == team_id for value in history) / len(history)

    def reset_scene(self) -> None:
        self._samples.clear()
        self._history.clear()
        self._roles.clear()
        self._locked_team.clear()
        self._switch_streak.clear()
        self._dark_tracks.clear()
        self._centres = None
        self._scale = 1.0

    reset = reset_scene


TemporalTeamIdentity = TeamIdentityAssigner
