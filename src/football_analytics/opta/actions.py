"""Pass / dribble / duel / defensive action inference from touches + possession."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from football_analytics.opta.pitch_zones import PitchZones


PASS_COLUMNS = [
    "pass_id",
    "passer_global_id",
    "receiver_global_id",
    "passer_track_id",
    "receiver_track_id",
    "team_id",
    "start_time_ms",
    "end_time_ms",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "start_zone",
    "end_zone",
    "distance_m",
    "forward_progress_m",
    "successful",
    "long_pass",
    "progressive_pass",
    "confidence",
    "status",
]

DRIBBLE_COLUMNS = [
    "dribble_id",
    "attacker_global_id",
    "defender_global_id",
    "attacker_track_id",
    "timestamp_ms",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "distance_m",
    "successful",
    "defender_beaten",
    "confidence",
    "status",
]

DUEL_COLUMNS = [
    "duel_id",
    "player_a",
    "player_b",
    "team_a",
    "team_b",
    "timestamp_ms",
    "duel_type",
    "winner_global_id",
    "loser_global_id",
    "confidence",
    "status",
]

DEFENSIVE_COLUMNS = [
    "action_id",
    "action_type",
    "global_player_id",
    "track_id",
    "team_id",
    "timestamp_ms",
    "start_zone",
    "end_zone",
    "distance_m",
    "under_pressure",
    "confidence",
    "status",
]


@dataclass(frozen=True)
class ActionInferenceConfig:
    long_pass_m: float = 25.0
    min_pass_confidence: float = 0.45
    max_pass_gap_ms: float = 4000.0
    dribble_min_m: float = 3.0
    dribble_opponent_m: float = 3.0
    duel_proximity_m: float = 2.5
    confirmed_confidence: float = 0.65
    candidate_confidence: float = 0.40


def _team_key(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value)
    if text in {"", "unknown", "None"}:
        return None
    return text


def _status(conf: float, cfg: ActionInferenceConfig) -> str:
    if conf >= cfg.confirmed_confidence:
        return "confirmed"
    if conf >= cfg.candidate_confidence:
        return "candidate"
    return "unresolved"


def infer_passes(
    touches: pd.DataFrame,
    *,
    zones: PitchZones | None = None,
    config: ActionInferenceConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ActionInferenceConfig()
    zones = zones or PitchZones()
    if touches is None or touches.empty:
        return pd.DataFrame(columns=PASS_COLUMNS)

    ordered = touches.sort_values("timestamp_ms").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    pid = 0
    for i in range(1, len(ordered)):
        a = ordered.iloc[i - 1]
        b = ordered.iloc[i]
        gap = float(b["timestamp_ms"] - a["timestamp_ms"])
        if gap <= 0 or gap > cfg.max_pass_gap_ms:
            continue
        if int(a["track_id"]) == int(b["track_id"]):
            continue
        team_a = _team_key(a.get("team_id"))
        team_b = _team_key(b.get("team_id"))
        if team_a is None or team_b is None:
            continue
        # Require controlled touch evidence when available
        if "controlled_touch" in a and not bool(a.get("controlled_touch", False)):
            continue
        conf = float(min(a.get("confidence", 0.0), b.get("confidence", 0.0)))
        if conf < cfg.min_pass_confidence:
            continue
        same_team = team_a == team_b
        sx = float(a.get("distance_m")) if False else np.nan  # placeholder
        # Prefer pitch coords if present on touches; else NaN
        start_x = float(a["pitch_x"]) if "pitch_x" in a and pd.notna(a.get("pitch_x")) else np.nan
        start_y = float(a["pitch_y"]) if "pitch_y" in a and pd.notna(a.get("pitch_y")) else np.nan
        end_x = float(b["pitch_x"]) if "pitch_x" in b and pd.notna(b.get("pitch_x")) else np.nan
        end_y = float(b["pitch_y"]) if "pitch_y" in b and pd.notna(b.get("pitch_y")) else np.nan
        # Fallback: use distance_m column only for length estimate
        if np.isfinite(start_x) and np.isfinite(end_x):
            dist = float(np.hypot(end_x - start_x, end_y - start_y))
            team_idx = 0 if str(team_a).endswith("0") else 1
            sign = zones.team0_attack_sign if team_idx == 0 else -zones.team0_attack_sign
            forward = float((end_x - start_x) * sign)
            start_zone = zones.zone_third(start_x, team_id=team_idx)
            end_zone = zones.zone_third(end_x, team_id=team_idx)
        else:
            dist = float("nan")
            forward = float("nan")
            start_zone = end_zone = None
        successful = bool(same_team)
        if bool(a.get("deflection", False)) or bool(b.get("deflection", False)):
            conf = min(conf, 0.5)
        pid += 1
        rows.append(
            {
                "pass_id": f"pass-{pid:05d}",
                "passer_global_id": a.get("global_player_id"),
                "receiver_global_id": b.get("global_player_id"),
                "passer_track_id": int(a["track_id"]),
                "receiver_track_id": int(b["track_id"]),
                "team_id": team_a,
                "start_time_ms": float(a["timestamp_ms"]),
                "end_time_ms": float(b["timestamp_ms"]),
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "start_zone": start_zone,
                "end_zone": end_zone,
                "distance_m": dist,
                "forward_progress_m": forward,
                "successful": successful,
                "long_pass": bool(np.isfinite(dist) and dist >= cfg.long_pass_m),
                "progressive_pass": bool(np.isfinite(forward) and forward >= 10.0 and successful),
                "confidence": conf,
                "status": _status(conf, cfg),
            }
        )
    return pd.DataFrame(rows, columns=PASS_COLUMNS)


def infer_clearances(
    touches: pd.DataFrame,
    *,
    zones: PitchZones | None = None,
    config: ActionInferenceConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ActionInferenceConfig()
    zones = zones or PitchZones()
    if touches is None or touches.empty or "pitch_x" not in touches.columns:
        return pd.DataFrame(columns=DEFENSIVE_COLUMNS)
    rows: list[dict[str, Any]] = []
    aid = 0
    ordered = touches.sort_values("timestamp_ms").reset_index(drop=True)
    for i in range(len(ordered) - 1):
        a = ordered.iloc[i]
        b = ordered.iloc[i + 1]
        if not bool(a.get("controlled_touch", False)):
            continue
        sx, sy = a.get("pitch_x"), a.get("pitch_y")
        ex, ey = b.get("pitch_x"), b.get("pitch_y")
        if not all(pd.notna(v) for v in (sx, sy, ex, ey)):
            continue
        team = _team_key(a.get("team_id"))
        team_idx = 0 if team and str(team).endswith("0") else 1
        if zones.zone_third(float(sx), team_id=team_idx) != "zone_1":
            continue
        dist = float(np.hypot(float(ex) - float(sx), float(ey) - float(sy)))
        if dist < 8.0:
            continue
        # Same player clearance kick away
        if int(a["track_id"]) != int(b["track_id"]) and _team_key(b.get("team_id")) == team:
            continue
        conf = float(a.get("confidence", 0.0))
        aid += 1
        rows.append(
            {
                "action_id": f"clr-{aid:05d}",
                "action_type": "clearance",
                "global_player_id": a.get("global_player_id"),
                "track_id": int(a["track_id"]),
                "team_id": team,
                "timestamp_ms": float(a["timestamp_ms"]),
                "start_zone": "zone_1",
                "end_zone": zones.zone_third(float(ex), team_id=team_idx),
                "distance_m": dist,
                "under_pressure": False,
                "confidence": conf,
                "status": _status(conf, cfg),
            }
        )
    return pd.DataFrame(rows, columns=DEFENSIVE_COLUMNS)


def infer_turnovers_and_interceptions(
    passes: pd.DataFrame,
    *,
    config: ActionInferenceConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ActionInferenceConfig()
    if passes is None or passes.empty:
        return pd.DataFrame(columns=DEFENSIVE_COLUMNS)
    rows: list[dict[str, Any]] = []
    aid = 0
    for row in passes.itertuples(index=False):
        if bool(row.successful):
            continue
        conf = float(row.confidence)
        intercepted = row.receiver_track_id is not None
        aid += 1
        rows.append(
            {
                "action_id": f"def-{aid:05d}",
                "action_type": "interception" if intercepted else "turnover",
                "global_player_id": row.receiver_global_id if intercepted else None,
                "track_id": row.receiver_track_id if intercepted else None,
                "team_id": None,
                "timestamp_ms": float(row.end_time_ms),
                "start_zone": row.start_zone,
                "end_zone": row.end_zone,
                "distance_m": row.distance_m,
                "under_pressure": False,
                "confidence": conf,
                "status": _status(conf, cfg),
            }
        )
        aid += 1
        rows.append(
            {
                "action_id": f"def-{aid:05d}",
                "action_type": "dispossession" if intercepted else "turnover",
                "global_player_id": row.passer_global_id,
                "track_id": row.passer_track_id,
                "team_id": row.team_id,
                "timestamp_ms": float(row.start_time_ms),
                "start_zone": row.start_zone,
                "end_zone": row.end_zone,
                "distance_m": row.distance_m,
                "under_pressure": False,
                "confidence": conf,
                "status": _status(conf, cfg),
            }
        )
    return pd.DataFrame(rows, columns=DEFENSIVE_COLUMNS)


def infer_duels_from_proximity(
    tracks: pd.DataFrame,
    touches: pd.DataFrame,
    *,
    identities: pd.DataFrame | None = None,
    config: ActionInferenceConfig | None = None,
) -> pd.DataFrame:
    """Conservative duel candidates: opposing players near a touch within a short window."""
    cfg = config or ActionInferenceConfig()
    if tracks is None or tracks.empty or touches is None or touches.empty:
        return pd.DataFrame(columns=DUEL_COLUMNS)
    if "x_field" not in tracks.columns and "foot_x_pixel" in tracks.columns:
        # Without field coords, do not invent duels.
        return pd.DataFrame(columns=DUEL_COLUMNS)

    team_by_track: dict[int, Any] = {}
    if identities is not None and not identities.empty:
        for tid, g in identities.groupby("track_id"):
            assigned = g[g["team_id"].notna()]
            if not assigned.empty:
                team_by_track[int(tid)] = assigned.iloc[-1]["team_id"]

    rows: list[dict[str, Any]] = []
    did = 0
    # Only use person tracks with field coords
    if "x_field" not in tracks.columns:
        return pd.DataFrame(columns=DUEL_COLUMNS)
    if "timestamp_ms" not in tracks.columns:
        return pd.DataFrame(columns=DUEL_COLUMNS)
    for touch in touches.itertuples(index=False):
        ts = float(touch.timestamp_ms)
        window = tracks[
            (tracks["timestamp_ms"] >= ts - 400)
            & (tracks["timestamp_ms"] <= ts + 400)
            & tracks["x_field"].notna()
        ]
        if window.empty:
            continue
        # Find two closest opposing tracks to touch pitch if available
        # Without pitch on touch, skip
        if not hasattr(touch, "pitch_x") or pd.isna(getattr(touch, "pitch_x", np.nan)):
            continue
        px, py = float(touch.pitch_x), float(touch.pitch_y)
        near = []
        for tr in window.itertuples(index=False):
            d = float(np.hypot(float(tr.x_field) - px, float(tr.y_field) - py))
            if d <= cfg.duel_proximity_m:
                near.append((d, tr))
        near.sort(key=lambda item: item[0])
        if len(near) < 2:
            continue
        a = near[0][1]
        b = near[1][1]
        ta = team_by_track.get(int(a.track_id))
        tb = team_by_track.get(int(b.track_id))
        if ta is None or tb is None or str(ta) == str(tb):
            continue
        conf = 0.45
        did += 1
        winner = int(touch.track_id) if int(touch.track_id) in {int(a.track_id), int(b.track_id)} else None
        rows.append(
            {
                "duel_id": f"duel-{did:05d}",
                "player_a": int(a.track_id),
                "player_b": int(b.track_id),
                "team_a": ta,
                "team_b": tb,
                "timestamp_ms": ts,
                "duel_type": "ground_duel",
                "winner_global_id": None,
                "loser_global_id": None,
                "confidence": conf,
                "status": _status(conf, cfg) if winner is not None else "unresolved",
            }
        )
    return pd.DataFrame(rows, columns=DUEL_COLUMNS)


def infer_dribbles(
    touches: pd.DataFrame,
    *,
    zones: PitchZones | None = None,
    config: ActionInferenceConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ActionInferenceConfig()
    if touches is None or touches.empty or "pitch_x" not in touches.columns:
        return pd.DataFrame(columns=DRIBBLE_COLUMNS)
    ordered = touches.sort_values("timestamp_ms")
    rows: list[dict[str, Any]] = []
    did = 0
    by_track = {int(k): v.sort_values("timestamp_ms") for k, v in ordered.groupby("track_id")}
    for track_id, group in by_track.items():
        controlled = group[group.get("controlled_touch", True) == True] if "controlled_touch" in group else group
        if len(controlled) < 2:
            continue
        first = controlled.iloc[0]
        last = controlled.iloc[-1]
        if not all(pd.notna(v) for v in (first.get("pitch_x"), first.get("pitch_y"), last.get("pitch_x"), last.get("pitch_y"))):
            continue
        dist = float(
            np.hypot(
                float(last["pitch_x"]) - float(first["pitch_x"]),
                float(last["pitch_y"]) - float(first["pitch_y"]),
            )
        )
        if dist < cfg.dribble_min_m:
            continue
        conf = float(min(first.get("confidence", 0.0), last.get("confidence", 0.0)))
        did += 1
        rows.append(
            {
                "dribble_id": f"drb-{did:05d}",
                "attacker_global_id": first.get("global_player_id"),
                "defender_global_id": None,
                "attacker_track_id": track_id,
                "timestamp_ms": float(first["timestamp_ms"]),
                "start_x": float(first["pitch_x"]),
                "start_y": float(first["pitch_y"]),
                "end_x": float(last["pitch_x"]),
                "end_y": float(last["pitch_y"]),
                "distance_m": dist,
                "successful": True,  # only same-player sustained control observed
                "defender_beaten": False,
                "confidence": min(conf, 0.55),
                "status": _status(min(conf, 0.55), cfg),
            }
        )
    return pd.DataFrame(rows, columns=DRIBBLE_COLUMNS)
