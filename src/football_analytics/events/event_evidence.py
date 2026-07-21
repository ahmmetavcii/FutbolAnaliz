"""Evidence items and conservative aggregation for event detection.

Evidence is a set of independent, named signals (ball crossed line estimate,
scoreboard change, restart pattern, possession change, ...). Aggregation is
deliberately pessimistic:

- Replay-sourced evidence is excluded from scoring entirely.
- A single strong signal is capped; corroboration from *distinct* sources is
  required for high aggregate scores.
- The aggregate score is an internal evidence measure, not a calibrated
  probability of the event being real.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator


@dataclass(frozen=True)
class EvidenceItem:
    """One named signal supporting (or contradicting) an event."""

    source: str
    score: float
    timestamp_ms: float | None = None
    description: str = ""
    from_replay: bool = False

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source must be non-empty")
        if not -1.0 <= self.score <= 1.0:
            raise ValueError("score must be in [-1, 1] (negative = contradicting)")
        if self.timestamp_ms is not None and not math.isfinite(self.timestamp_ms):
            raise ValueError("timestamp_ms must be finite")


@dataclass(frozen=True)
class EvidenceBundle:
    """Immutable collection of evidence with conservative scoring."""

    items: tuple[EvidenceItem, ...] = field(default_factory=tuple)
    #: Score contribution cap for the strongest evidence source.
    single_source_cap: float = 0.5
    #: Cap for each additional distinct source (diminishing returns).
    secondary_source_cap: float = 0.3

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        if not 0.0 < self.single_source_cap <= 1.0:
            raise ValueError("single_source_cap must be in (0, 1]")
        if not 0.0 < self.secondary_source_cap <= self.single_source_cap:
            raise ValueError("secondary_source_cap must be in (0, single_source_cap]")

    def __iter__(self) -> Iterator[EvidenceItem]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def with_items(self, items: Iterable[EvidenceItem]) -> "EvidenceBundle":
        return EvidenceBundle(self.items + tuple(items), self.single_source_cap)

    @property
    def usable_items(self) -> tuple[EvidenceItem, ...]:
        """Evidence eligible for scoring: replay-sourced items are excluded."""
        return tuple(item for item in self.items if not item.from_replay)

    @property
    def sources(self) -> frozenset[str]:
        return frozenset(item.source for item in self.usable_items)

    def aggregate_score(self) -> float:
        """Conservative evidence score in ``[0, 1]``.

        The strongest supporting source contributes at most
        ``single_source_cap``; each further distinct source at most
        ``secondary_source_cap`` (repeated observations from one source never
        stack). The strongest contradicting item is subtracted. One source
        alone can therefore never exceed the cap, so auto-confirmation
        thresholds above it structurally require corroboration.
        """
        per_source_support: dict[str, float] = {}
        contradiction = 0.0
        for item in self.usable_items:
            if item.score >= 0.0:
                per_source_support[item.source] = max(
                    per_source_support.get(item.source, 0.0), item.score
                )
            else:
                contradiction = max(contradiction, -item.score)
        ranked = sorted(per_source_support.values(), reverse=True)
        support = 0.0
        for index, score in enumerate(ranked):
            cap = self.single_source_cap if index == 0 else self.secondary_source_cap
            support += min(score, cap)
        return max(0.0, min(1.0, support - contradiction))

    @property
    def corroborated(self) -> bool:
        """True when at least two distinct sources support the event."""
        supporting = {
            item.source for item in self.usable_items if item.score > 0.0
        }
        return len(supporting) >= 2


def bundle(items: Iterable[EvidenceItem]) -> EvidenceBundle:
    return EvidenceBundle(tuple(items))
