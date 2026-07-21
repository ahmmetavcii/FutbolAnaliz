"""Possession chain builder from touch events (pass candidates for assists)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from football_analytics.events.assist_detector import PassObservation


POSSESSION_CHAIN_COLUMNS = [
    "chain_id",
    "segment_index",
    "start_ms",
    "end_ms",
    "team_id",
    "from_track_id",
    "to_track_id",
    "from_global_player_id",
    "to_global_player_id",
    "confidence",
    "pass_like",
    "deflection",
]


@dataclass(frozen=True)
class PossessionChainConfig:
    max_pass_gap_ms: float = 4_000.0
    min_pass_confidence: float = 0.40


def build_possession_chain(
    touches: pd.DataFrame,
    *,
    config: PossessionChainConfig | None = None,
) -> pd.DataFrame:
    cfg = config or PossessionChainConfig()
    if touches is None or touches.empty:
        return pd.DataFrame(columns=POSSESSION_CHAIN_COLUMNS)

    ordered = touches.sort_values("timestamp_ms").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    chain_id = 0
    segment = 0
    for index in range(1, len(ordered)):
        prev = ordered.iloc[index - 1]
        curr = ordered.iloc[index]
        gap = float(curr["timestamp_ms"] - prev["timestamp_ms"])
        if gap <= 0 or gap > cfg.max_pass_gap_ms:
            chain_id += 1
            segment = 0
            continue
        if int(prev["track_id"]) == int(curr["track_id"]):
            continue
        same_team = (
            prev["team_id"] is not None
            and curr["team_id"] is not None
            and str(prev["team_id"]) == str(curr["team_id"])
        )
        conf = float(min(prev["confidence"], curr["confidence"]))
        if not same_team:
            chain_id += 1
            segment = 0
            continue
        if conf < cfg.min_pass_confidence:
            continue
        segment += 1
        rows.append(
            {
                "chain_id": f"chain-{chain_id:04d}",
                "segment_index": segment,
                "start_ms": float(prev["timestamp_ms"]),
                "end_ms": float(curr["timestamp_ms"]),
                "team_id": prev["team_id"],
                "from_track_id": int(prev["track_id"]),
                "to_track_id": int(curr["track_id"]),
                "from_global_player_id": prev.get("global_player_id"),
                "to_global_player_id": curr.get("global_player_id"),
                "confidence": conf,
                "pass_like": bool(prev.get("controlled_touch", False)),
                "deflection": bool(prev.get("deflection", False) or curr.get("deflection", False)),
            }
        )
    return pd.DataFrame(rows, columns=POSSESSION_CHAIN_COLUMNS)


def passes_from_chain(chain: pd.DataFrame) -> list[PassObservation]:
    if chain is None or chain.empty:
        return []
    passes: list[PassObservation] = []
    for row in chain.itertuples(index=False):
        if not bool(row.pass_like):
            continue
        team = row.team_id
        team_id = None
        if team is not None and str(team) not in {"", "unknown", "None"}:
            text = str(team)
            if text.endswith("0"):
                team_id = 0
            elif text.endswith("1"):
                team_id = 1
        passes.append(
            PassObservation(
                timestamp_ms=float(row.end_ms),
                passer_track_id=int(row.from_track_id),
                receiver_track_id=int(row.to_track_id),
                team_id=team_id,
                confidence=float(row.confidence),
                from_replay=False,
            )
        )
    return passes
