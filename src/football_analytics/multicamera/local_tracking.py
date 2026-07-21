"""Data model for per-camera (local) tracking output.

These dataclasses are the contract between single-camera trackers and the
cross-camera identity layer. They carry no inference logic: detections, team
labels, jersey readings, and re-identification embeddings are produced
upstream and only transported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


class PlayerRole(str, Enum):
    OUTFIELD = "outfield"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LocalObservation:
    """One tracked detection from one camera at one instant.

    ``reference_time_seconds`` must already be on the shared reference
    timeline (see :mod:`football_analytics.multicamera.synchronization`).
    ``pitch_xy_m`` is None whenever the camera calibration is invalid.
    """

    camera_id: str
    local_track_id: int
    frame_index: int
    reference_time_seconds: float
    bbox_xyxy: tuple[float, float, float, float]
    pitch_xy_m: tuple[float, float] | None = None
    team_id: int | None = None
    team_confidence: float = 0.0
    jersey_number: int | None = None
    jersey_confidence: float = 0.0
    role: PlayerRole = PlayerRole.UNKNOWN
    reid_embedding: tuple[float, ...] | None = None
    detection_confidence: float = 1.0
    shot_id: int | None = None

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        if not (x2 > x1 and y2 > y1):
            raise ValueError(f"bbox_xyxy must be a positive box, got {self.bbox_xyxy}")
        for name, value in (
            ("team_confidence", self.team_confidence),
            ("jersey_confidence", self.jersey_confidence),
            ("detection_confidence", self.detection_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def track_key(self) -> tuple[str, int]:
        """Globally unique key of the local track this observation belongs to."""
        return (self.camera_id, self.local_track_id)

    def embedding_array(self) -> NDArray[np.float64] | None:
        if self.reid_embedding is None:
            return None
        return np.asarray(self.reid_embedding, dtype=np.float64)


@dataclass
class LocalTrack:
    """All observations of one local track id within one camera."""

    camera_id: str
    local_track_id: int
    observations: list[LocalObservation] = field(default_factory=list)

    def add(self, observation: LocalObservation) -> None:
        if observation.track_key != (self.camera_id, self.local_track_id):
            raise ValueError(
                f"observation {observation.track_key} does not belong to track "
                f"{(self.camera_id, self.local_track_id)}"
            )
        self.observations.append(observation)
        self.observations.sort(key=lambda obs: obs.reference_time_seconds)

    @property
    def start_time_seconds(self) -> float | None:
        return self.observations[0].reference_time_seconds if self.observations else None

    @property
    def end_time_seconds(self) -> float | None:
        return self.observations[-1].reference_time_seconds if self.observations else None

    def mean_embedding(self) -> NDArray[np.float64] | None:
        embeddings = [
            obs.embedding_array() for obs in self.observations if obs.reid_embedding is not None
        ]
        arrays = [e for e in embeddings if e is not None]
        if not arrays:
            return None
        return np.mean(np.stack(arrays), axis=0)


def group_into_tracks(observations: Iterable[LocalObservation]) -> dict[tuple[str, int], LocalTrack]:
    """Bucket a stream of observations into per-camera local tracks."""
    tracks: dict[tuple[str, int], LocalTrack] = {}
    for observation in observations:
        key = observation.track_key
        track = tracks.get(key)
        if track is None:
            track = LocalTrack(camera_id=key[0], local_track_id=key[1])
            tracks[key] = track
        track.add(observation)
    return tracks


def cosine_similarity(
    left: Sequence[float] | NDArray[np.floating],
    right: Sequence[float] | NDArray[np.floating],
) -> float:
    """Cosine similarity in [-1, 1]; 0.0 when either vector is degenerate."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        return 0.0
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm < 1e-12:
        return 0.0
    return float(np.dot(a, b) / norm)
