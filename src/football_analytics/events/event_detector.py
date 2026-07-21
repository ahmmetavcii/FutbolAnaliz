"""Shared machinery for conservative event detectors.

Detectors here are *evidence aggregators*: they collect named signals, score
them pessimistically, and emit candidates for review. They do not claim real
detection accuracy. Status policy:

- ``AUTO_CONFIRMED`` requires a high aggregate score AND corroboration from
  at least two distinct evidence sources.
- ``CANDIDATE_REVIEW_REQUIRED`` for anything above the candidate threshold.
- ``UNRESOLVED`` for weak-but-nonzero evidence worth keeping for audit.
- Below the floor, no event is emitted at all.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable

from football_analytics.events.event_evidence import EvidenceBundle, EvidenceItem
from football_analytics.events.schemas import EventStatus, EventType, MatchEvent


@dataclass(frozen=True)
class EventDetectorConfig:
    #: Aggregate score required for auto-confirmation (corroboration is also
    #: required; a single source can never reach this on its own).
    auto_confirm_score: float = 0.85
    #: Aggregate score to surface a candidate for human review.
    candidate_score: float = 0.45
    #: Aggregate score below which evidence is kept only as UNRESOLVED.
    unresolved_floor: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 <= self.unresolved_floor <= self.candidate_score <= self.auto_confirm_score <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= unresolved_floor <= candidate_score"
                " <= auto_confirm_score <= 1"
            )


class EventDetector:
    """Base class turning evidence bundles into status-tagged events."""

    event_type: EventType

    def __init__(self, config: EventDetectorConfig | None = None) -> None:
        self.config = config or EventDetectorConfig()
        self._id_counter = itertools.count(1)

    def _next_event_id(self) -> str:
        return f"{self.event_type.value}-{next(self._id_counter):04d}"

    def resolve_status(self, evidence: EvidenceBundle) -> EventStatus | None:
        """Map an evidence bundle to a status, or None to emit nothing."""
        score = evidence.aggregate_score()
        if score >= self.config.auto_confirm_score and evidence.corroborated:
            return EventStatus.AUTO_CONFIRMED
        if score >= self.config.candidate_score:
            return EventStatus.CANDIDATE_REVIEW_REQUIRED
        if score >= self.config.unresolved_floor:
            return EventStatus.UNRESOLVED
        return None

    def build_event(
        self,
        *,
        timestamp_ms: float,
        evidence: Iterable[EvidenceItem] | EvidenceBundle,
        team_id: int | None = None,
        scorer_track_id: int | None = None,
        assist_track_id: int | None = None,
        attributes: dict | None = None,
        attribution_is_subject: bool = False,
    ) -> MatchEvent | None:
        bundle = (
            evidence
            if isinstance(evidence, EvidenceBundle)
            else EvidenceBundle(tuple(evidence))
        )
        status = self.resolve_status(bundle)
        if status is None:
            return None
        if status is not EventStatus.AUTO_CONFIRMED and not attribution_is_subject:
            # Attribution is only kept on confirmed events; candidates and
            # unresolved events carry it inside attributes as a *suggestion*.
            attributes = dict(attributes or {})
            if scorer_track_id is not None:
                attributes.setdefault("suggested_scorer_track_id", scorer_track_id)
            if assist_track_id is not None:
                attributes.setdefault("suggested_assist_track_id", assist_track_id)
            scorer_track_id = None
            assist_track_id = None
        return MatchEvent(
            event_id=self._next_event_id(),
            event_type=self.event_type,
            status=status,
            timestamp_ms=timestamp_ms,
            team_id=team_id,
            scorer_track_id=scorer_track_id,
            assist_track_id=assist_track_id,
            confidence=bundle.aggregate_score(),
            evidence=bundle,
            attributes=attributes or {},
        )
