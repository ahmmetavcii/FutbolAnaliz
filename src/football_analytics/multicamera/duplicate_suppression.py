"""Detect and suppress duplicate observations of the same physical player.

Two flavours of duplication are handled:

- *intra-camera*: two boxes from one camera at the same instant, practically
  on top of each other on the pitch — usually a detector double-fire; the
  lower-confidence observation is suppressed.
- *cross-identity sanity check*: two different global identities that share a
  confident jersey+team at the same instant while standing far apart are NOT
  duplicates (that is physically consistent — e.g. a misread number) and are
  reported for review rather than merged, mirroring the impossible-duplicate
  rejection in :mod:`.cross_camera_reid`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .local_tracking import LocalObservation


@dataclass(frozen=True)
class DuplicateConfig:
    same_instant_tolerance_seconds: float = 0.08
    same_spot_distance_m: float = 1.0
    far_apart_distance_m: float = 20.0
    minimum_jersey_confidence: float = 0.5


@dataclass(frozen=True)
class DuplicatePair:
    kept: LocalObservation
    suppressed: LocalObservation
    distance_m: float
    reason: str


def find_intra_camera_duplicates(
    observations: Iterable[LocalObservation],
    config: DuplicateConfig | None = None,
) -> list[DuplicatePair]:
    """Find same-camera, same-instant, same-spot observation pairs."""
    cfg = config or DuplicateConfig()
    rows = sorted(observations, key=lambda obs: obs.reference_time_seconds)
    pairs: list[DuplicatePair] = []
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            dt = right.reference_time_seconds - left.reference_time_seconds
            if dt > cfg.same_instant_tolerance_seconds:
                break
            if left.camera_id != right.camera_id:
                continue
            if left.local_track_id == right.local_track_id:
                continue
            if left.pitch_xy_m is None or right.pitch_xy_m is None:
                continue
            distance = math.hypot(
                left.pitch_xy_m[0] - right.pitch_xy_m[0],
                left.pitch_xy_m[1] - right.pitch_xy_m[1],
            )
            if distance > cfg.same_spot_distance_m:
                continue
            kept, suppressed = (
                (left, right)
                if left.detection_confidence >= right.detection_confidence
                else (right, left)
            )
            pairs.append(
                DuplicatePair(
                    kept=kept,
                    suppressed=suppressed,
                    distance_m=distance,
                    reason="same_camera_same_spot",
                )
            )
    return pairs


def suppress_duplicates(
    observations: Sequence[LocalObservation],
    config: DuplicateConfig | None = None,
) -> tuple[list[LocalObservation], list[DuplicatePair]]:
    """Return (kept observations, suppressed pairs) preserving input order."""
    pairs = find_intra_camera_duplicates(observations, config)
    suppressed_ids = {id(pair.suppressed) for pair in pairs}
    kept = [obs for obs in observations if id(obs) not in suppressed_ids]
    return kept, pairs


def find_impossible_shared_identities(
    observations: Iterable[LocalObservation],
    config: DuplicateConfig | None = None,
) -> list[dict]:
    """Flag same jersey+team seen far apart at the same instant.

    These pairs must never be merged into one identity; they are returned as
    review items with the evidence that makes the merge impossible.
    """
    cfg = config or DuplicateConfig()
    rows = sorted(
        (
            obs
            for obs in observations
            if obs.jersey_number is not None
            and obs.jersey_confidence >= cfg.minimum_jersey_confidence
            and obs.pitch_xy_m is not None
        ),
        key=lambda obs: obs.reference_time_seconds,
    )
    flags: list[dict] = []
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            dt = right.reference_time_seconds - left.reference_time_seconds
            if dt > cfg.same_instant_tolerance_seconds:
                break
            if left.track_key == right.track_key:
                continue
            if left.jersey_number != right.jersey_number:
                continue
            if (
                left.team_id is not None
                and right.team_id is not None
                and left.team_id != right.team_id
            ):
                continue
            distance = math.hypot(
                left.pitch_xy_m[0] - right.pitch_xy_m[0],
                left.pitch_xy_m[1] - right.pitch_xy_m[1],
            )
            if distance < cfg.far_apart_distance_m:
                continue
            flags.append(
                {
                    "reason": "impossible_simultaneous_duplicate",
                    "jersey_number": left.jersey_number,
                    "track_keys": [list(left.track_key), list(right.track_key)],
                    "distance_m": distance,
                    "time_delta_seconds": dt,
                }
            )
    return flags
