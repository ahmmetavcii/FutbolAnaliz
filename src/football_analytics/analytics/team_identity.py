"""Temporal, quality-masked shirt-colour team identity."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np
from sklearn.cluster import KMeans

from football_analytics.analytics.role_identity import PlayerRole
from football_analytics.geometry.bbox import BBox


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
) -> tuple[np.ndarray | None, float]:
    """Extract a robust LAB+HSV shirt feature from the central upper torso."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
        raise ValueError("frame_bgr must be a uint8 HxWx3 BGR image")
    box = bbox if isinstance(bbox, BBox) else BBox.from_sequence(bbox)
    height, width = frame_bgr.shape[:2]
    clipped = box.clip(width, height)
    if not clipped.is_valid(min_area=1.0):
        return None, 0.0

    # Avoid head, shorts, arms, and box-edge background.
    x1 = int(np.floor(clipped.x1 + 0.20 * clipped.width))
    x2 = int(np.ceil(clipped.x2 - 0.20 * clipped.width))
    y1 = int(np.floor(clipped.y1 + 0.18 * clipped.height))
    y2 = int(np.ceil(clipped.y1 + 0.58 * clipped.height))
    crop = frame_bgr[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
    if crop.size == 0:
        return None, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hue, saturation, value = cv2.split(hsv)
    green = (hue >= 30) & (hue <= 95) & (saturation >= 45)
    dark = value <= 35
    bright = value >= 235
    useful = ~(green | dark | bright)
    count = int(np.count_nonzero(useful))
    fraction = count / float(useful.size)
    if count < min_pixels:
        return None, fraction

    # Medians make logos, skin, and isolated background pixels low influence.
    lab_median = np.median(lab[useful], axis=0)
    hsv_median = np.median(hsv[useful], axis=0)
    feature = np.concatenate((lab_median, hsv_median)).astype(np.float32)
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
    scale = np.maximum(1.4826 * mad, 1.0)
    score = np.sqrt(np.mean(np.square((matrix - median) / scale), axis=1))
    kept = matrix[score <= z_threshold]
    if kept.size == 0:
        kept = matrix
    return np.median(kept, axis=0).astype(np.float32)


class TeamIdentityAssigner:
    """Build two scene-local teams from temporal per-track shirt features."""

    def __init__(
        self,
        *,
        min_samples: int = 8,
        min_tracks: int = 4,
        history_size: int = 20,
        min_valid_pixel_fraction: float = 0.15,
        unknown_confidence_threshold: float = 0.55,
        random_state: int = 0,
    ) -> None:
        if min_samples < 2 or min_tracks < 2:
            raise ValueError("two-team clustering needs at least two samples and tracks")
        self.min_samples = min_samples
        self.min_tracks = min_tracks
        self.history_size = history_size
        self.min_valid_pixel_fraction = min_valid_pixel_fraction
        self.unknown_confidence_threshold = unknown_confidence_threshold
        self.random_state = random_state
        self._samples: dict[int, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self._centres: np.ndarray | None = None
        self._scale: float = 1.0
        self._history: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=12))
        self._roles: dict[int, PlayerRole] = {}

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
        if row.size != 6 or not np.all(np.isfinite(row)):
            raise ValueError("colour feature must contain six finite LAB+HSV values")
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
        pooled = [
            (track_id, robust_pool(samples))
            for track_id, samples in self._samples.items()
            if self._roles.get(track_id) is not PlayerRole.REFEREE
        ]
        pooled = [(track_id, row) for track_id, row in pooled if row is not None]
        sample_count = sum(len(samples) for samples in self._samples.values())
        if len(pooled) < self.min_tracks or sample_count < self.min_samples:
            return False
        matrix = np.stack([row for _, row in pooled])
        model = KMeans(n_clusters=2, n_init=10, random_state=self.random_state).fit(matrix)
        centres = model.cluster_centers_.astype(np.float32)
        # Stable labels are based on LAB lightness, then chromatic channels.
        order = np.lexsort((centres[:, 2], centres[:, 1], centres[:, 0]))
        self._centres = centres[order]
        residuals = np.linalg.norm(matrix - model.cluster_centers_[model.labels_], axis=1)
        self._scale = max(float(np.median(residuals)) * 2.5, 8.0)
        return True

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
        distances = np.linalg.norm(self._centres - feature, axis=1)
        winner = int(np.argmin(distances))
        nearest, other = float(distances[winner]), float(distances[1 - winner])
        separation = (other - nearest) / max(other, 1e-6)
        proximity = float(np.exp(-nearest / self._scale))
        confidence = float(np.clip(0.5 * separation + 0.5 * proximity, 0.0, 1.0))
        if confidence < self.unknown_confidence_threshold:
            return TeamAssignment(None, confidence, self._consistency(track_id, winner), parsed_role)
        self._history[track_id].append(winner)
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
        self._centres = None
        self._scale = 1.0

    reset = reset_scene


TemporalTeamIdentity = TeamIdentityAssigner
