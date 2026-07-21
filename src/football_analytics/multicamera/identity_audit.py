"""Append-only audit trail for global identity decisions.

Every identity creation, track attachment, merge, and split is recorded with a
monotonically increasing sequence number. Merge events capture enough state
(the absorbed identity's track bindings) to be reversed exactly by a later
split, and split events point back at the merge they undo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class AuditEventKind(str, Enum):
    CREATE = "create"
    ATTACH = "attach"
    MATCH = "match"
    UNRESOLVED = "unresolved"
    MERGE = "merge"
    SPLIT = "split"


@dataclass(frozen=True)
class AuditEvent:
    sequence: int
    kind: AuditEventKind
    global_id: int | None
    reference_time_seconds: float | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    # For MERGE: bindings moved from the absorbed identity, keyed for reversal.
    # For SPLIT: the sequence number of the merge event being reversed.
    reversal_payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class IdentityAuditLog:
    events: list[AuditEvent] = field(default_factory=list)
    _next_sequence: int = 0
    _reversed_merges: set[int] = field(default_factory=set, repr=False)

    def record(
        self,
        kind: AuditEventKind,
        global_id: int | None,
        reference_time_seconds: float | None = None,
        details: Mapping[str, Any] | None = None,
        reversal_payload: Mapping[str, Any] | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            sequence=self._next_sequence,
            kind=kind,
            global_id=global_id,
            reference_time_seconds=reference_time_seconds,
            details=dict(details or {}),
            reversal_payload=dict(reversal_payload or {}),
        )
        self._next_sequence += 1
        self.events.append(event)
        return event

    def merge_event(self, sequence: int) -> AuditEvent:
        for event in self.events:
            if event.sequence == sequence and event.kind == AuditEventKind.MERGE:
                return event
        raise KeyError(f"no merge event with sequence {sequence}")

    def is_merge_reversed(self, sequence: int) -> bool:
        return sequence in self._reversed_merges

    def mark_merge_reversed(self, sequence: int) -> None:
        self._reversed_merges.add(sequence)

    def history_for(self, global_id: int) -> tuple[AuditEvent, ...]:
        """All events touching a global id, in chronological order."""
        return tuple(
            event
            for event in self.events
            if event.global_id == global_id
            or event.details.get("source_global_id") == global_id
            or event.details.get("target_global_id") == global_id
            or event.details.get("restored_global_id") == global_id
        )
