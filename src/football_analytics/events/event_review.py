"""Manual review: corrections applied as an ordered, replayable log.

Corrections never mutate detector output in place. They live in a
:class:`ReviewLog` and are re-applied on top of freshly detected events, so
re-running detection with improved models keeps every human decision
(recompute-friendly). Rules enforced on application:

- Confirm / reject moves an event to MANUALLY_CONFIRMED / MANUALLY_REJECTED.
- Attribution edits (scorer/assist) may set a value or explicitly null it.
- Own-goal / direct-penalty goals cannot be given an assist.
- Corrections referencing unknown events are reported, not silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Sequence

from football_analytics.events.schemas import EventStatus, EventType, MatchEvent

_UNSET = object()


class CorrectionKind(str, Enum):
    CONFIRM = "confirm"
    REJECT = "reject"
    SET_SCORER = "set_scorer"
    SET_ASSIST = "set_assist"
    SET_TEAM = "set_team"
    SET_ATTRIBUTE = "set_attribute"


@dataclass(frozen=True)
class Correction:
    event_id: str
    kind: CorrectionKind
    #: New value for SET_* corrections; None explicitly nulls the field.
    value: object = None
    attribute: str | None = None
    reviewer: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind is CorrectionKind.SET_ATTRIBUTE and not self.attribute:
            raise ValueError("SET_ATTRIBUTE corrections require an attribute name")


@dataclass
class ReviewLog:
    """Append-only ordered list of corrections."""

    corrections: list[Correction] = field(default_factory=list)

    def add(self, correction: Correction) -> None:
        self.corrections.append(correction)

    def confirm(self, event_id: str, *, reviewer: str = "", note: str = "") -> None:
        self.add(Correction(event_id, CorrectionKind.CONFIRM, reviewer=reviewer, note=note))

    def reject(self, event_id: str, *, reviewer: str = "", note: str = "") -> None:
        self.add(Correction(event_id, CorrectionKind.REJECT, reviewer=reviewer, note=note))

    def set_scorer(self, event_id: str, track_id: int | None, *, reviewer: str = "") -> None:
        self.add(Correction(event_id, CorrectionKind.SET_SCORER, track_id, reviewer=reviewer))

    def set_assist(self, event_id: str, track_id: int | None, *, reviewer: str = "") -> None:
        self.add(Correction(event_id, CorrectionKind.SET_ASSIST, track_id, reviewer=reviewer))


@dataclass(frozen=True)
class ReviewResult:
    events: tuple[MatchEvent, ...]
    unmatched_event_ids: tuple[str, ...]
    rejected_corrections: tuple[tuple[Correction, str], ...]


def apply_review(
    events: Sequence[MatchEvent],
    log: ReviewLog | Iterable[Correction],
) -> ReviewResult:
    """Apply a correction log on top of detected events (pure function)."""
    corrections = list(log.corrections if isinstance(log, ReviewLog) else log)
    by_id: dict[str, MatchEvent] = {}
    for event in events:
        if event.event_id in by_id:
            raise ValueError(f"duplicate event_id {event.event_id}")
        by_id[event.event_id] = event

    unmatched: list[str] = []
    rejected: list[tuple[Correction, str]] = []
    for correction in corrections:
        event = by_id.get(correction.event_id)
        if event is None:
            unmatched.append(correction.event_id)
            continue
        try:
            by_id[correction.event_id] = _apply_one(event, correction)
        except ValueError as exc:
            rejected.append((correction, str(exc)))

    ordered = tuple(by_id[event.event_id] for event in events)
    return ReviewResult(ordered, tuple(unmatched), tuple(rejected))


def _apply_one(event: MatchEvent, correction: Correction) -> MatchEvent:
    kind = correction.kind
    if kind is CorrectionKind.CONFIRM:
        return replace(event, status=EventStatus.MANUALLY_CONFIRMED)
    if kind is CorrectionKind.REJECT:
        return replace(event, status=EventStatus.MANUALLY_REJECTED)
    if kind is CorrectionKind.SET_SCORER:
        return replace(event, scorer_track_id=_as_optional_int(correction.value))
    if kind is CorrectionKind.SET_ASSIST:
        assist = _as_optional_int(correction.value)
        if assist is not None and event.event_type is EventType.GOAL:
            if event.own_goal or event.penalty:
                raise ValueError("own goals and direct penalties cannot carry an assist")
        return replace(event, assist_track_id=assist)
    if kind is CorrectionKind.SET_TEAM:
        return replace(event, team_id=_as_optional_int(correction.value))
    if kind is CorrectionKind.SET_ATTRIBUTE:
        attributes = dict(event.attributes)
        attributes[correction.attribute] = correction.value  # type: ignore[index]
        return replace(event, attributes=attributes)
    raise ValueError(f"unsupported correction kind {kind}")


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("track/team corrections require an int or None")
    return value
