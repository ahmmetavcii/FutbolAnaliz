"""Substitution candidates modelled as time intervals.

A substitution is never an instant: a bench player becomes active around the
same window in which an active player leaves the pitch permanently. The
detector pairs exit/entry observations into :class:`SubstitutionInterval`
windows and emits a candidate per pairing. Unpaired exits (injury, red card,
tracking loss) surface as UNRESOLVED evidence rather than being forced into a
substitution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from football_analytics.events.event_detector import EventDetector, EventDetectorConfig
from football_analytics.events.event_evidence import EvidenceItem
from football_analytics.events.schemas import EventType, MatchEvent, SubstitutionInterval


@dataclass(frozen=True)
class PitchExit:
    """An active player leaving the pitch without returning."""

    track_id: int
    timestamp_ms: float
    team_id: int | None = None
    confidence: float = 0.0
    from_replay: bool = False


@dataclass(frozen=True)
class PitchEntry:
    """A previously-benched player becoming active on the pitch."""

    track_id: int
    timestamp_ms: float
    team_id: int | None = None
    confidence: float = 0.0
    came_from_bench: bool = False
    from_replay: bool = False


@dataclass(frozen=True)
class SubstitutionDetectorConfig:
    detector: EventDetectorConfig = EventDetectorConfig()
    #: Maximum |entry - exit| gap for a pairing to form one substitution.
    max_pairing_gap_ms: float = 120_000.0


class SubstitutionDetector(EventDetector):
    event_type = EventType.SUBSTITUTION

    def __init__(self, config: SubstitutionDetectorConfig | None = None) -> None:
        self.sub_config = config or SubstitutionDetectorConfig()
        super().__init__(self.sub_config.detector)

    def detect(
        self, exits: Sequence[PitchExit], entries: Sequence[PitchEntry]
    ) -> list[MatchEvent]:
        usable_exits = sorted(
            (e for e in exits if not e.from_replay), key=lambda e: e.timestamp_ms
        )
        usable_entries = sorted(
            (e for e in entries if not e.from_replay), key=lambda e: e.timestamp_ms
        )
        events: list[MatchEvent] = []
        remaining_entries = list(usable_entries)
        for exit_obs in usable_exits:
            entry = self._match_entry(exit_obs, remaining_entries)
            if entry is not None:
                remaining_entries.remove(entry)
                event = self._paired_event(exit_obs, entry)
            else:
                event = self._unpaired_exit_event(exit_obs)
            if event is not None:
                events.append(event)
        return events

    def _match_entry(
        self, exit_obs: PitchExit, entries: list[PitchEntry]
    ) -> PitchEntry | None:
        candidates = [
            entry
            for entry in entries
            if (exit_obs.team_id is None or entry.team_id is None or entry.team_id == exit_obs.team_id)
            and abs(entry.timestamp_ms - exit_obs.timestamp_ms) <= self.sub_config.max_pairing_gap_ms
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda e: abs(e.timestamp_ms - exit_obs.timestamp_ms))

    def _paired_event(self, exit_obs: PitchExit, entry: PitchEntry) -> MatchEvent | None:
        interval = SubstitutionInterval(
            start_ms=min(exit_obs.timestamp_ms, entry.timestamp_ms),
            end_ms=max(exit_obs.timestamp_ms, entry.timestamp_ms),
        )
        items = [
            EvidenceItem(
                source="pitch_exit",
                score=exit_obs.confidence,
                timestamp_ms=exit_obs.timestamp_ms,
                description=f"track {exit_obs.track_id} left the pitch",
            ),
            EvidenceItem(
                source="pitch_entry",
                score=entry.confidence,
                timestamp_ms=entry.timestamp_ms,
                description=f"track {entry.track_id} entered the pitch",
            ),
        ]
        if entry.came_from_bench:
            items.append(
                EvidenceItem(
                    source="bench_origin",
                    score=min(entry.confidence, 0.8),
                    timestamp_ms=entry.timestamp_ms,
                )
            )
        return self.build_event(
            timestamp_ms=interval.midpoint_ms,
            evidence=items,
            team_id=exit_obs.team_id if exit_obs.team_id is not None else entry.team_id,
            attributes={
                "player_off_track_id": exit_obs.track_id,
                "player_on_track_id": entry.track_id,
                "interval_start_ms": interval.start_ms,
                "interval_end_ms": interval.end_ms,
            },
        )

    def _unpaired_exit_event(self, exit_obs: PitchExit) -> MatchEvent | None:
        # A lone exit is weak evidence: keep it as UNRESOLVED for audit but
        # never surface it as a substitution candidate on its own.
        items = [
            EvidenceItem(
                source="pitch_exit",
                score=min(exit_obs.confidence, 0.4),
                timestamp_ms=exit_obs.timestamp_ms,
                description=f"track {exit_obs.track_id} left the pitch (no paired entry)",
            )
        ]
        return self.build_event(
            timestamp_ms=exit_obs.timestamp_ms,
            evidence=items,
            team_id=exit_obs.team_id,
            attributes={
                "player_off_track_id": exit_obs.track_id,
                "unpaired_exit": True,
            },
        )


def interval_of(event: MatchEvent) -> SubstitutionInterval | None:
    """Recover the substitution interval stored on an event, if any."""
    start = event.attributes.get("interval_start_ms")
    end = event.attributes.get("interval_end_ms")
    if start is None or end is None:
        return None
    return SubstitutionInterval(float(start), float(end))
