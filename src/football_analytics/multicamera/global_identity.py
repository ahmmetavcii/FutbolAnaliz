"""Global identity resolution across cameras with conservative, auditable merges.

Local tracks (one camera, one track id) are bound to numeric global identities
using the multi-cue scorer from :mod:`.cross_camera_reid`. The safety rules
required of this layer are:

- a jersey number alone is never sufficient to merge tracks,
- physically impossible matches (the identity seen almost simultaneously at an
  unreachable pitch position) are hard-rejected,
- ambiguous matches produce *unresolved* identities instead of risky merges,
- every merge is recorded with enough state to be reversed exactly, and every
  decision lands in the append-only audit log from :mod:`.identity_audit`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .cross_camera_reid import IdentitySnapshot, MatchScore, ReidMatchConfig, score_candidate
from .identity_audit import AuditEventKind, IdentityAuditLog
from .local_tracking import LocalObservation, PlayerRole

TrackKey = tuple[str, int]


class AssignmentStatus(str, Enum):
    MATCHED = "matched"
    NEW_IDENTITY = "new_identity"
    UNRESOLVED = "unresolved"


@dataclass
class GlobalIdentity:
    """Running summary of one player identity across all cameras."""

    global_id: int
    track_keys: set[TrackKey] = field(default_factory=set)
    team_id: int | None = None
    team_confidence: float = 0.0
    jersey_number: int | None = None
    jersey_confidence: float = 0.0
    role: PlayerRole = PlayerRole.UNKNOWN
    embedding: NDArray[np.float64] | None = None
    embedding_count: int = 0
    last_time_seconds: float | None = None
    last_pitch_xy_m: tuple[float, float] | None = None
    last_camera_id: str | None = None
    unresolved: bool = False

    def snapshot(self) -> IdentitySnapshot:
        return IdentitySnapshot(
            global_id=self.global_id,
            team_id=self.team_id,
            team_confidence=self.team_confidence,
            jersey_number=self.jersey_number,
            jersey_confidence=self.jersey_confidence,
            role=self.role,
            embedding=self.embedding,
            last_time_seconds=self.last_time_seconds,
            last_pitch_xy_m=self.last_pitch_xy_m,
            last_camera_id=self.last_camera_id,
        )

    def absorb(self, observation: LocalObservation) -> None:
        """Fold one observation into the running summary."""
        self.track_keys.add(observation.track_key)
        if (
            observation.team_id is not None
            and observation.team_confidence >= self.team_confidence
        ):
            self.team_id = observation.team_id
            self.team_confidence = observation.team_confidence
        if (
            observation.jersey_number is not None
            and observation.jersey_confidence >= self.jersey_confidence
        ):
            self.jersey_number = observation.jersey_number
            self.jersey_confidence = observation.jersey_confidence
        if self.role == PlayerRole.UNKNOWN and observation.role != PlayerRole.UNKNOWN:
            self.role = observation.role

        embedding = observation.embedding_array()
        if embedding is not None:
            if self.embedding is None or self.embedding.shape != embedding.shape:
                self.embedding = embedding.copy()
                self.embedding_count = 1
            else:
                total = self.embedding * self.embedding_count + embedding
                self.embedding_count += 1
                self.embedding = total / self.embedding_count

        # last_time and last_pitch must describe the same instant so the
        # position-feasibility gate in cross_camera_reid stays meaningful.
        if self.last_time_seconds is None or (
            observation.reference_time_seconds >= self.last_time_seconds
        ):
            self.last_time_seconds = observation.reference_time_seconds
            self.last_pitch_xy_m = observation.pitch_xy_m
            self.last_camera_id = observation.camera_id

    def copy(self) -> "GlobalIdentity":
        clone = GlobalIdentity(
            global_id=self.global_id,
            track_keys=set(self.track_keys),
            team_id=self.team_id,
            team_confidence=self.team_confidence,
            jersey_number=self.jersey_number,
            jersey_confidence=self.jersey_confidence,
            role=self.role,
            embedding=None if self.embedding is None else self.embedding.copy(),
            embedding_count=self.embedding_count,
            last_time_seconds=self.last_time_seconds,
            last_pitch_xy_m=self.last_pitch_xy_m,
            last_camera_id=self.last_camera_id,
            unresolved=self.unresolved,
        )
        return clone


@dataclass(frozen=True)
class Assignment:
    """Outcome of routing one observation into the identity registry."""

    status: AssignmentStatus
    global_id: int
    score: MatchScore | None = None
    candidate_global_id: int | None = None


@dataclass
class GlobalIdentityRegistry:
    """Bind local tracks to global identities under conservative rules."""

    config: ReidMatchConfig = field(default_factory=ReidMatchConfig)
    audit: IdentityAuditLog = field(default_factory=IdentityAuditLog)
    identities: dict[int, GlobalIdentity] = field(default_factory=dict)
    track_to_global: dict[TrackKey, int] = field(default_factory=dict)
    _next_global_id: int = 1
    #: Same-camera new local tracks are only eligible to merge after this gap,
    #: so two simultaneously visible teammates cannot collapse into one id.
    same_camera_reappear_gap_seconds: float = 1.5

    # ------------------------------------------------------------------ #
    # Assignment
    # ------------------------------------------------------------------ #
    def assign(self, observation: LocalObservation) -> Assignment:
        """Route an observation to an existing or new global identity.

        A previously bound local track always stays with its identity. New
        tracks are matched against every identity; a match is only accepted
        when the score clears the threshold *and* is supported by at least one
        non-jersey cue. Ambiguous scores yield an unresolved identity.
        """
        key = observation.track_key
        bound = self.track_to_global.get(key)
        if bound is not None:
            self.identities[bound].absorb(observation)
            return Assignment(status=AssignmentStatus.MATCHED, global_id=bound)

        best: MatchScore | None = None
        for identity in self.identities.values():
            if (
                identity.last_camera_id == observation.camera_id
                and identity.last_time_seconds is not None
                and abs(
                    observation.reference_time_seconds - identity.last_time_seconds
                )
                < self.same_camera_reappear_gap_seconds
            ):
                # Identity is still live on this camera: a different local
                # track at nearly the same time is a different person.
                continue
            candidate = score_candidate(observation, identity.snapshot(), self.config)
            if candidate.hard_reject:
                continue
            if best is None or candidate.score > best.score:
                best = candidate

        if (
            best is not None
            and best.score >= self.config.accept_threshold
            and best.acceptable
        ):
            identity = self.identities[best.identity_global_id]
            identity.absorb(observation)
            self.track_to_global[key] = identity.global_id
            self.audit.record(
                AuditEventKind.MATCH,
                identity.global_id,
                observation.reference_time_seconds,
                details={"track_key": list(key), "score": best.score},
            )
            return Assignment(
                status=AssignmentStatus.MATCHED,
                global_id=identity.global_id,
                score=best,
            )

        if best is not None and best.score >= self.config.unresolved_threshold:
            # Plausible but not confident enough (or jersey-only support):
            # keep the track separate and flag it for review.
            identity = self._create_identity(observation, unresolved=True)
            self.audit.record(
                AuditEventKind.UNRESOLVED,
                identity.global_id,
                observation.reference_time_seconds,
                details={
                    "track_key": list(key),
                    "candidate_global_id": best.identity_global_id,
                    "score": best.score,
                    "jersey_only": not best.supported_by_non_jersey,
                },
            )
            return Assignment(
                status=AssignmentStatus.UNRESOLVED,
                global_id=identity.global_id,
                score=best,
                candidate_global_id=best.identity_global_id,
            )

        identity = self._create_identity(observation, unresolved=False)
        return Assignment(status=AssignmentStatus.NEW_IDENTITY, global_id=identity.global_id)

    def assign_all(self, observations: Iterable[LocalObservation]) -> list[Assignment]:
        ordered = sorted(observations, key=lambda obs: obs.reference_time_seconds)
        return [self.assign(observation) for observation in ordered]

    def _create_identity(
        self, observation: LocalObservation, unresolved: bool
    ) -> GlobalIdentity:
        identity = GlobalIdentity(global_id=self._next_global_id, unresolved=unresolved)
        self._next_global_id += 1
        identity.absorb(observation)
        self.identities[identity.global_id] = identity
        self.track_to_global[observation.track_key] = identity.global_id
        self.audit.record(
            AuditEventKind.CREATE,
            identity.global_id,
            observation.reference_time_seconds,
            details={"track_key": list(observation.track_key), "unresolved": unresolved},
        )
        return identity

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def global_id_for(self, track_key: TrackKey) -> int | None:
        return self.track_to_global.get(track_key)

    def unresolved_identities(self) -> tuple[GlobalIdentity, ...]:
        return tuple(
            identity for identity in self.identities.values() if identity.unresolved
        )

    def history_for(self, global_id: int):
        return self.audit.history_for(global_id)

    # ------------------------------------------------------------------ #
    # Reversible merge / split
    # ------------------------------------------------------------------ #
    def merge_identities(self, source_global_id: int, target_global_id: int, reason: str) -> int:
        """Merge ``source`` into ``target``; returns the merge audit sequence.

        The pre-merge state of both identities is captured so
        :meth:`undo_merge` can restore them exactly.
        """
        if source_global_id == target_global_id:
            raise ValueError("cannot merge an identity into itself")
        source = self.identities[source_global_id]
        target = self.identities[target_global_id]

        reversal_payload = {
            "source_state": source.copy(),
            "target_state": target.copy(),
        }

        for key in source.track_keys:
            target.track_keys.add(key)
            self.track_to_global[key] = target_global_id
        if source.team_confidence > target.team_confidence:
            target.team_id = source.team_id
            target.team_confidence = source.team_confidence
        if source.jersey_confidence > target.jersey_confidence:
            target.jersey_number = source.jersey_number
            target.jersey_confidence = source.jersey_confidence
        if target.role == PlayerRole.UNKNOWN:
            target.role = source.role
        if source.embedding is not None:
            if target.embedding is None or target.embedding.shape != source.embedding.shape:
                target.embedding = source.embedding.copy()
                target.embedding_count = source.embedding_count
            else:
                total = (
                    target.embedding * target.embedding_count
                    + source.embedding * source.embedding_count
                )
                target.embedding_count += source.embedding_count
                target.embedding = total / target.embedding_count
        if source.last_time_seconds is not None and (
            target.last_time_seconds is None
            or source.last_time_seconds > target.last_time_seconds
        ):
            target.last_time_seconds = source.last_time_seconds
            target.last_pitch_xy_m = source.last_pitch_xy_m
            target.last_camera_id = source.last_camera_id
        target.unresolved = target.unresolved and source.unresolved

        del self.identities[source_global_id]
        event = self.audit.record(
            AuditEventKind.MERGE,
            target_global_id,
            details={
                "source_global_id": source_global_id,
                "target_global_id": target_global_id,
                "reason": reason,
            },
            reversal_payload=reversal_payload,
        )
        return event.sequence

    def undo_merge(self, merge_sequence: int) -> int:
        """Reverse a recorded merge exactly; returns the restored global id."""
        event = self.audit.merge_event(merge_sequence)
        if self.audit.is_merge_reversed(merge_sequence):
            raise ValueError(f"merge {merge_sequence} was already reversed")
        source_state: GlobalIdentity = event.reversal_payload["source_state"]
        target_state: GlobalIdentity = event.reversal_payload["target_state"]

        restored_source = source_state.copy()
        restored_target = target_state.copy()
        self.identities[restored_source.global_id] = restored_source
        self.identities[restored_target.global_id] = restored_target
        for key in restored_source.track_keys:
            self.track_to_global[key] = restored_source.global_id
        for key in restored_target.track_keys:
            self.track_to_global[key] = restored_target.global_id

        self.audit.mark_merge_reversed(merge_sequence)
        self.audit.record(
            AuditEventKind.SPLIT,
            restored_target.global_id,
            details={
                "restored_global_id": restored_source.global_id,
                "target_global_id": restored_target.global_id,
                "reason": "undo_merge",
            },
            reversal_payload={"merge_sequence": merge_sequence},
        )
        return restored_source.global_id

    def split_identity(
        self, global_id: int, track_keys: Sequence[TrackKey], reason: str
    ) -> int:
        """Move the given local tracks out into a fresh identity.

        Summary attributes of the new identity start blank on purpose: after a
        manual split the safe assumption is that the moved tracks' attributes
        must be re-derived from their own observations.
        """
        identity = self.identities[global_id]
        moving = [tuple(key) for key in track_keys if tuple(key) in identity.track_keys]
        if not moving:
            raise ValueError(f"none of the given tracks belong to identity {global_id}")

        new_identity = GlobalIdentity(global_id=self._next_global_id, unresolved=True)
        self._next_global_id += 1
        for key in moving:
            identity.track_keys.remove(key)
            new_identity.track_keys.add(key)
            self.track_to_global[key] = new_identity.global_id
        self.identities[new_identity.global_id] = new_identity

        self.audit.record(
            AuditEventKind.SPLIT,
            global_id,
            details={
                "restored_global_id": new_identity.global_id,
                "moved_track_keys": [list(key) for key in moving],
                "reason": reason,
            },
        )
        return new_identity.global_id


def apply_manual_corrections(
    registry: GlobalIdentityRegistry,
    corrections: Mapping[str, Any],
) -> GlobalIdentityRegistry:
    """Apply reviewed merge/split/undo decisions to a registry.

    Expected layout::

        {
          "merges": [{"source": 3, "target": 1}],
          "splits": [{"global_id": 2, "track_keys": [["cam2", 7]]}],
          "undo_merges": [12],
        }
    """
    for merge in corrections.get("merges", []):
        source = int(merge["source"])
        target = int(merge["target"])
        if source in registry.identities and target in registry.identities:
            registry.merge_identities(source, target, reason="manual_merge")
    for split in corrections.get("splits", []):
        global_id = int(split["global_id"])
        keys = [(str(cam), int(track)) for cam, track in split.get("track_keys", [])]
        if global_id in registry.identities and keys:
            registry.split_identity(global_id, keys, reason="manual_split")
    for sequence in corrections.get("undo_merges", []):
        registry.undo_merge(int(sequence))
    return registry
