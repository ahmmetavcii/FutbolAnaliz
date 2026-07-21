"""Scorer attribution from touch chains near a goal moment.

Never guesses: unresolved scorers stay ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ScorerDetectorConfig:
    lookback_ms: float = 4_000.0
    min_attribution: float = 0.70
    require_controlled_touch: bool = True


@dataclass(frozen=True)
class ScorerResult:
    scorer_global_player_id: int | None
    scorer_track_id: int | None
    scorer_team_id: Any
    scorer_jersey_number: int | None
    scorer_confidence: float
    scorer_evidence: dict[str, Any]
    scorer_status: str


def detect_scorer(
    *,
    goal_timestamp_ms: float,
    touches: pd.DataFrame,
    jersey_by_track: dict[int, int] | None = None,
    config: ScorerDetectorConfig | None = None,
) -> ScorerResult:
    cfg = config or ScorerDetectorConfig()
    empty = ScorerResult(
        scorer_global_player_id=None,
        scorer_track_id=None,
        scorer_team_id=None,
        scorer_jersey_number=None,
        scorer_confidence=0.0,
        scorer_evidence={"reason": "no_touch_in_lookback"},
        scorer_status="unresolved",
    )
    if touches is None or touches.empty:
        return empty

    window = touches[
        (touches["timestamp_ms"] <= goal_timestamp_ms)
        & (touches["timestamp_ms"] >= goal_timestamp_ms - cfg.lookback_ms)
    ].sort_values("timestamp_ms")
    if window.empty:
        return empty

    candidates = window
    if cfg.require_controlled_touch:
        controlled = window[window["controlled_touch"].fillna(False)]
        if not controlled.empty:
            candidates = controlled
    last = candidates.iloc[-1]
    confidence = float(last["confidence"])
    track_id = int(last["track_id"])
    jersey = (jersey_by_track or {}).get(track_id)
    evidence = {
        "touch_id": str(last["touch_id"]),
        "distance_px": float(last.get("distance_px") or 0.0),
        "controlled_touch": bool(last.get("controlled_touch", False)),
        "lookback_ms": cfg.lookback_ms,
    }
    if confidence < cfg.min_attribution:
        return ScorerResult(
            scorer_global_player_id=None,
            scorer_track_id=None,
            scorer_team_id=last.get("team_id"),
            scorer_jersey_number=jersey,
            scorer_confidence=confidence,
            scorer_evidence={**evidence, "reason": "attribution_below_threshold"},
            scorer_status="unresolved",
        )
    global_id = last.get("global_player_id")
    return ScorerResult(
        scorer_global_player_id=int(global_id) if global_id is not None and str(global_id) != "nan" else None,
        scorer_track_id=track_id,
        scorer_team_id=last.get("team_id"),
        scorer_jersey_number=jersey,
        scorer_confidence=confidence,
        scorer_evidence=evidence,
        scorer_status="resolved",
    )
