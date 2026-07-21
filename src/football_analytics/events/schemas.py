"""Canonical event records, statuses, and lifecycle rules.

Status lifecycle:

- Detectors emit ``CANDIDATE_REVIEW_REQUIRED`` (enough evidence to surface) or
  ``UNRESOLVED`` (some evidence, but conflicting/insufficient). Automatic
  confirmation (``AUTO_CONFIRMED``) is reserved for overwhelming corroborated
  evidence and is deliberately rare.
- Human review moves events to ``MANUALLY_CONFIRMED`` or ``MANUALLY_REJECTED``.
- Only ``AUTO_CONFIRMED`` and ``MANUALLY_CONFIRMED`` events count in any
  summary. A candidate is never counted as confirmed.
- Attribution fields (scorer, assist) stay ``None`` until resolved; an
  unresolved scorer/assist is represented as ``None``, never guessed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

from football_analytics.events.event_evidence import EvidenceBundle, EvidenceItem


class EventType(str, Enum):
    GOAL = "goal"
    SHOT = "shot"
    ASSIST = "assist"
    SUBSTITUTION = "substitution"


class EventStatus(str, Enum):
    AUTO_CONFIRMED = "auto_confirmed"
    CANDIDATE_REVIEW_REQUIRED = "candidate_review_required"
    MANUALLY_CONFIRMED = "manually_confirmed"
    MANUALLY_REJECTED = "manually_rejected"
    UNRESOLVED = "unresolved"


CONFIRMED_STATUSES: frozenset[EventStatus] = frozenset(
    {EventStatus.AUTO_CONFIRMED, EventStatus.MANUALLY_CONFIRMED}
)


def is_confirmed(status: EventStatus) -> bool:
    """True only for auto/manually confirmed events; candidates never count."""
    return status in CONFIRMED_STATUSES


def is_countable(status: EventStatus) -> bool:
    return is_confirmed(status)


@dataclass(frozen=True)
class MatchEvent:
    """One detected or manually entered match event.

    ``scorer_track_id`` / ``assist_track_id`` / ``team_id`` are ``None`` when
    unresolved. ``attributes`` carries type-specific flags such as
    ``own_goal`` and ``penalty``.
    """

    event_id: str
    event_type: EventType
    status: EventStatus
    timestamp_ms: float
    team_id: int | None = None
    scorer_track_id: int | None = None
    assist_track_id: int | None = None
    confidence: float = 0.0
    evidence: EvidenceBundle = field(default_factory=EvidenceBundle)
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if not math.isfinite(self.timestamp_ms):
            raise ValueError("timestamp_ms must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "attributes", dict(self.attributes))
        if self.event_type is EventType.GOAL and self.assist_track_id is not None:
            if self.attributes.get("own_goal") or self.attributes.get("penalty"):
                raise ValueError("own goals and direct penalties cannot carry an assist")

    @property
    def counts_as_confirmed(self) -> bool:
        return is_confirmed(self.status)

    @property
    def own_goal(self) -> bool:
        return bool(self.attributes.get("own_goal", False))

    @property
    def penalty(self) -> bool:
        return bool(self.attributes.get("penalty", False))

    def with_status(self, status: EventStatus) -> "MatchEvent":
        return replace(self, status=status)


@dataclass(frozen=True)
class SubstitutionInterval:
    """Time interval over which a substitution takes place.

    Substitutions are not instants: the board goes up, the player walks off,
    the replacement enters. ``start_ms``/``end_ms`` bound that window; the
    nominal event timestamp is the interval midpoint.
    """

    start_ms: float
    end_ms: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start_ms) and math.isfinite(self.end_ms)):
            raise ValueError("interval bounds must be finite")
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")

    @property
    def midpoint_ms(self) -> float:
        return (self.start_ms + self.end_ms) / 2.0

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


__all__ = [
    "CONFIRMED_STATUSES",
    "EventStatus",
    "EventType",
    "EvidenceBundle",
    "EvidenceItem",
    "MatchEvent",
    "SubstitutionInterval",
    "is_confirmed",
    "is_countable",
]
