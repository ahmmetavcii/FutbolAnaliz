"""Opta-like player/team aggregates, activity index, heatmaps (honest coverage)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_analytics.analytics.heatmaps import HeatmapConfig, HeatmapSample, compute_heatmap
from football_analytics.opta.pitch_zones import PitchZones


PLAYER_SUMMARY_COLUMNS = [
    "global_player_id",
    "team_id",
    "jersey_number",
    "role",
    "visible_seconds",
    "identity_quality",
    "metric_quality",
    "distance_m",
    "max_speed_kmh",
    "sprint_count",
    "pass_attempts",
    "passes_completed",
    "pass_completion_pct",
    "long_pass_attempts",
    "long_passes_completed",
    "long_pass_completion_pct",
    "zone_1_to_2_passes",
    "zone_2_to_3_passes",
    "dribble_attempts",
    "dribbles_completed",
    "dribble_success_pct",
    "duels",
    "duels_won",
    "duel_win_pct",
    "aerial_duels",
    "aerial_duels_won",
    "tackles_won",
    "interceptions",
    "clearances",
    "turnovers",
    "dispossessions",
    "penalty_area_touches",
    "activity_index",
    "quality_flags",
]


@dataclass(frozen=True)
class ActivityIndexConfig:
    min_visible_seconds: float = 8.0
    weight_distance: float = 0.20
    weight_touches: float = 0.20
    weight_passes: float = 0.20
    weight_duels: float = 0.15
    weight_defensive: float = 0.15
    weight_penalty: float = 0.10


def _pct(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return round(100.0 * num / den, 1)


def _confirmed(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or "status" not in frame.columns:
        return frame.iloc[0:0] if frame is not None else pd.DataFrame()
    return frame[frame["status"].astype(str).eq("confirmed")]


def enrich_touches_with_ball_pitch(touches: pd.DataFrame, ball: pd.DataFrame) -> pd.DataFrame:
    if touches is None or touches.empty or ball is None or ball.empty:
        return touches
    if "pitch_x" in touches.columns and touches["pitch_x"].notna().any():
        return touches
    ball_idx = ball.set_index("frame_id") if "frame_id" in ball.columns else None
    if ball_idx is None:
        return touches
    rows = []
    for row in touches.to_dict("records"):
        fid = int(row["frame_id"])
        if fid in ball_idx.index:
            b = ball_idx.loc[fid]
            if isinstance(b, pd.DataFrame):
                b = b.iloc[0]
            row["pitch_x"] = b.get("pitch_x", b.get("ball_x_field", np.nan))
            row["pitch_y"] = b.get("pitch_y", b.get("ball_y_field", np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def build_player_identity_table(
    tracks: pd.DataFrame,
    identities: pd.DataFrame | None,
    global_map: pd.DataFrame | None,
    *,
    roles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map local tracks → global players with quality flags (no invented merges)."""
    if tracks is None or tracks.empty:
        return pd.DataFrame()
    person = tracks[tracks["object_type"].eq("person")] if "object_type" in tracks.columns else tracks
    team_by_track: dict[int, Any] = {}
    if identities is not None and not identities.empty:
        for tid, g in identities.groupby("track_id"):
            assigned = g[g["team_id"].notna()]
            if not assigned.empty:
                team_by_track[int(tid)] = str(assigned.iloc[-1]["team_id"])
    global_by_track: dict[int, Any] = {}
    unresolved: set[int] = set()
    if global_map is not None and not global_map.empty:
        for row in global_map.itertuples(index=False):
            global_by_track[int(row.local_track_id)] = row.global_id
            if bool(getattr(row, "unresolved", False)):
                unresolved.add(int(row.local_track_id))

    rows = []
    for tid, g in person.groupby("track_id"):
        tid = int(tid)
        ts = g["timestamp_ms"].astype(float)
        visible = float((ts.max() - ts.min()) / 1000.0) if len(ts) else 0.0
        gid = global_by_track.get(tid, tid)  # fallback: local id as provisional
        team = team_by_track.get(tid)
        role = "unknown_person"
        rows.append(
            {
                "global_player_id": gid,
                "local_track_id": tid,
                "team_id": team,
                "role": role,
                "visible_seconds": round(visible, 2),
                "first_seen": float(ts.min()) if len(ts) else None,
                "last_seen": float(ts.max()) if len(ts) else None,
                "track_fragment_count": 1,
                "reid_confidence": None if tid in unresolved else 0.5,
                "team_confidence": 0.5 if team else 0.0,
                "identity_quality": "low" if tid in unresolved or team is None else "medium",
                "unresolved": tid in unresolved,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Collapse fragments sharing global id
    agg = (
        frame.groupby("global_player_id", as_index=False)
        .agg(
            team_id=("team_id", "first"),
            role=("role", "first"),
            visible_seconds=("visible_seconds", "sum"),
            first_seen=("first_seen", "min"),
            last_seen=("last_seen", "max"),
            track_fragment_count=("local_track_id", "count"),
            identity_quality=("identity_quality", "first"),
            unresolved=("unresolved", "max"),
        )
    )
    # Exclude non-players by role keyword
    agg = agg[~agg["role"].astype(str).str.contains("referee|official|staff", case=False, na=False)]
    # Flag invalid count per team
    flags = []
    for team, g in agg.groupby("team_id"):
        if team is None or str(team) in {"", "unknown", "None"}:
            continue
        validated = g[~g["unresolved"].astype(bool)]
        if len(validated) > 11:
            flags.append(f"INVALID_PLAYER_IDENTITY_COUNT:{team}:{len(validated)}")
    agg["quality_flags"] = ""
    if flags:
        agg.attrs["identity_flags"] = flags
    return agg


def compute_activity_index(
    *,
    visible_seconds: float,
    distance_m: float | None,
    touches: int,
    passes: int,
    duels: int,
    defensive: int,
    penalty_touches: int,
    config: ActivityIndexConfig | None = None,
) -> float | None:
    cfg = config or ActivityIndexConfig()
    if visible_seconds < cfg.min_visible_seconds:
        return None
    minutes = max(visible_seconds / 60.0, 1e-6)
    dist_score = min(1.0, (distance_m or 0.0) / minutes / 120.0)
    touch_score = min(1.0, touches / minutes / 8.0)
    pass_score = min(1.0, passes / minutes / 6.0)
    duel_score = min(1.0, duels / minutes / 3.0)
    def_score = min(1.0, defensive / minutes / 2.0)
    pen_score = min(1.0, penalty_touches / max(minutes, 1.0))
    score = 100.0 * (
        cfg.weight_distance * dist_score
        + cfg.weight_touches * touch_score
        + cfg.weight_passes * pass_score
        + cfg.weight_duels * duel_score
        + cfg.weight_defensive * def_score
        + cfg.weight_penalty * pen_score
    )
    return round(float(score), 1)


def build_opta_player_summary(
    identities: pd.DataFrame,
    *,
    passes: pd.DataFrame,
    dribbles: pd.DataFrame,
    duels: pd.DataFrame,
    defensive: pd.DataFrame,
    touches: pd.DataFrame,
    player_metrics: pd.DataFrame | None,
    zones: PitchZones | None = None,
    track_to_global: dict[int, Any] | None = None,
) -> pd.DataFrame:
    zones = zones or PitchZones()
    track_to_global = track_to_global or {}
    c_passes = _confirmed(passes)
    c_dribbles = _confirmed(dribbles)
    c_duels = _confirmed(duels)
    c_def = _confirmed(defensive)

    metrics_by_track: dict[int, dict[str, Any]] = {}
    if player_metrics is not None and not player_metrics.empty and "track_id" in player_metrics.columns:
        for tid, g in player_metrics.groupby("track_id"):
            valid = g[g["valid"] == True] if "valid" in g.columns else g
            if valid.empty:
                continue
            metrics_by_track[int(tid)] = {
                "distance_m": float(valid["cumulative_distance_m"].max())
                if "cumulative_distance_m" in valid
                else None,
                "max_speed_kmh": float(valid["smoothed_speed_kmh"].max())
                if "smoothed_speed_kmh" in valid
                else None,
                "sprint_count": int((valid.get("sprint_state") == "sprint").sum())
                if "sprint_state" in valid
                else None,
                "metric_quality": float(valid["valid"].mean()) if "valid" in valid else None,
            }

    rows = []
    for player in identities.itertuples(index=False):
        gid = player.global_player_id
        # Map metrics via any local fragment — approximate: use local if gid==local
        local_ids = [gid] if isinstance(gid, (int, np.integer)) else []
        dist = max_speed = sprints = None
        metric_q = None
        for lid in local_ids:
            if int(lid) in metrics_by_track:
                m = metrics_by_track[int(lid)]
                dist = m["distance_m"]
                max_speed = m["max_speed_kmh"]
                sprints = m["sprint_count"]
                metric_q = m["metric_quality"]
                break

        p_att = c_passes[c_passes["passer_global_id"] == gid] if not c_passes.empty else c_passes
        if c_passes.empty and not passes.empty and "passer_track_id" in passes.columns:
            # fallback track match when global missing
            p_att = _confirmed(passes)
            p_att = p_att[p_att["passer_track_id"] == gid] if not p_att.empty else p_att
        pass_attempts = len(p_att)
        passes_ok = int(p_att["successful"].sum()) if pass_attempts and "successful" in p_att else 0
        long_att = int(p_att["long_pass"].sum()) if pass_attempts and "long_pass" in p_att else 0
        long_ok = int(p_att[p_att["long_pass"] == True]["successful"].sum()) if long_att else 0
        z12 = 0
        z23 = 0
        if pass_attempts and "start_zone" in p_att.columns:
            z12 = int(((p_att["start_zone"] == "zone_1") & (p_att["end_zone"] == "zone_2") & p_att["successful"]).sum())
            z23 = int(((p_att["start_zone"] == "zone_2") & (p_att["end_zone"] == "zone_3") & p_att["successful"]).sum())

        d_att = c_dribbles[c_dribbles["attacker_global_id"] == gid] if not c_dribbles.empty else c_dribbles
        if d_att.empty and not dribbles.empty:
            d_att = _confirmed(dribbles)
            d_att = d_att[d_att["attacker_track_id"] == gid] if not d_att.empty else d_att
        dribble_attempts = len(d_att)
        dribbles_ok = int(d_att["successful"].sum()) if dribble_attempts else 0

        duel_n = 0
        duel_w = 0
        aerial_n = aerial_w = 0
        # Duels often unresolved → count only confirmed with winner
        if not c_duels.empty:
            mine = c_duels[
                (c_duels["player_a"] == gid) | (c_duels["player_b"] == gid)
            ]
            duel_n = len(mine)
            duel_w = int((mine["winner_global_id"] == gid).sum()) if "winner_global_id" in mine else 0
            aerial = mine[mine["duel_type"].astype(str).eq("aerial_duel")]
            aerial_n = len(aerial)
            aerial_w = int((aerial["winner_global_id"] == gid).sum()) if len(aerial) else 0

        tackles = interceptions = clearances = turnovers = disposals = 0
        if not c_def.empty:
            mine = c_def[c_def["global_player_id"] == gid]
            tackles = int((mine["action_type"] == "tackle_won").sum())
            interceptions = int((mine["action_type"] == "interception").sum())
            clearances = int((mine["action_type"] == "clearance").sum())
            turnovers = int((mine["action_type"] == "turnover").sum())
            disposals = int((mine["action_type"] == "dispossession").sum())

        pen_touches = 0
        if touches is not None and not touches.empty and "pitch_x" in touches.columns:
            mine_t = touches[touches["global_player_id"] == gid]
            team_idx = 0 if str(getattr(player, "team_id", "")).endswith("0") else 1
            for t in mine_t.itertuples(index=False):
                if pd.notna(getattr(t, "pitch_x", np.nan)) and zones.in_opponent_penalty(
                    float(t.pitch_x), float(t.pitch_y), team_id=team_idx
                ):
                    if bool(getattr(t, "controlled_touch", True)):
                        pen_touches += 1

        touch_n = 0
        if touches is not None and not touches.empty:
            touch_n = int((touches["global_player_id"] == gid).sum())

        activity = compute_activity_index(
            visible_seconds=float(player.visible_seconds),
            distance_m=dist,
            touches=touch_n,
            passes=pass_attempts,
            duels=duel_n,
            defensive=tackles + interceptions + clearances,
            penalty_touches=pen_touches,
        )
        flags = []
        if getattr(player, "unresolved", False):
            flags.append("unresolved_identity")
        if metric_q is not None and metric_q < 0.3:
            flags.append("low_metric_coverage")
        if activity is None:
            flags.append("activity_insufficient_visibility")

        rows.append(
            {
                "global_player_id": gid,
                "team_id": getattr(player, "team_id", None),
                "jersey_number": None,
                "role": getattr(player, "role", "unknown_person"),
                "visible_seconds": float(player.visible_seconds),
                "identity_quality": getattr(player, "identity_quality", "low"),
                "metric_quality": metric_q,
                "distance_m": None if dist is None else round(dist, 1),
                "max_speed_kmh": None if max_speed is None else round(max_speed, 1),
                "sprint_count": sprints,
                "pass_attempts": pass_attempts,
                "passes_completed": passes_ok,
                "pass_completion_pct": _pct(passes_ok, pass_attempts),
                "long_pass_attempts": long_att,
                "long_passes_completed": long_ok,
                "long_pass_completion_pct": _pct(long_ok, long_att),
                "zone_1_to_2_passes": z12,
                "zone_2_to_3_passes": z23,
                "dribble_attempts": dribble_attempts,
                "dribbles_completed": dribbles_ok,
                "dribble_success_pct": _pct(dribbles_ok, dribble_attempts),
                "duels": duel_n,
                "duels_won": duel_w,
                "duel_win_pct": _pct(duel_w, duel_n),
                "aerial_duels": aerial_n,
                "aerial_duels_won": aerial_w,
                "tackles_won": tackles,
                "interceptions": interceptions,
                "clearances": clearances,
                "turnovers": turnovers,
                "dispossessions": disposals,
                "penalty_area_touches": pen_touches,
                "activity_index": activity,
                "quality_flags": ",".join(flags),
            }
        )
    return pd.DataFrame(rows, columns=PLAYER_SUMMARY_COLUMNS)


def build_opta_team_summary(players: pd.DataFrame) -> pd.DataFrame:
    if players is None or players.empty:
        return pd.DataFrame()
    rows = []
    for team, g in players.groupby("team_id"):
        if team is None or str(team) in {"", "unknown", "None"}:
            continue
        validated = g[g["identity_quality"].astype(str) != "low"] if "identity_quality" in g else g
        if len(validated) > 11:
            # Still report but flag
            pass
        def _sum(col: str) -> float:
            if col not in g.columns:
                return 0.0
            return float(pd.to_numeric(g[col], errors="coerce").fillna(0).sum())

        pa = _sum("pass_attempts")
        pc = _sum("passes_completed")
        la = _sum("long_pass_attempts")
        lc = _sum("long_passes_completed")
        da = _sum("dribble_attempts")
        dc = _sum("dribbles_completed")
        du = _sum("duels")
        dw = _sum("duels_won")
        rows.append(
            {
                "team_id": team,
                "validated_player_count": int(len(validated)),
                "metric_coverage": round(float(pd.to_numeric(g.get("metric_quality"), errors="coerce").mean() or 0), 2),
                "total_distance_m": round(_sum("distance_m"), 1),
                "sprint_count": int(_sum("sprint_count")),
                "pass_attempts": int(pa),
                "passes_completed": int(pc),
                "pass_completion_pct": _pct(pc, pa),
                "long_pass_attempts": int(la),
                "long_passes_completed": int(lc),
                "long_pass_completion_pct": _pct(lc, la),
                "zone_1_to_2_passes": int(_sum("zone_1_to_2_passes")),
                "zone_2_to_3_passes": int(_sum("zone_2_to_3_passes")),
                "dribble_attempts": int(da),
                "dribbles_completed": int(dc),
                "dribble_success_pct": _pct(dc, da),
                "duels": int(du),
                "duels_won": int(dw),
                "duel_win_pct": _pct(dw, du),
                "aerial_duels": int(_sum("aerial_duels")),
                "aerial_duels_won": int(_sum("aerial_duels_won")),
                "tackles_won": int(_sum("tackles_won")),
                "interceptions": int(_sum("interceptions")),
                "clearances": int(_sum("clearances")),
                "turnovers": int(_sum("turnovers")),
                "penalty_area_touches": int(_sum("penalty_area_touches")),
            }
        )
    return pd.DataFrame(rows)


def export_heatmaps(
    player_metrics: pd.DataFrame,
    out_dir: Path,
    *,
    identities: pd.DataFrame | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if player_metrics is None or player_metrics.empty:
        return {"players": 0, "path": str(out_dir)}
    if "x_field" not in player_metrics.columns:
        return {"players": 0, "path": str(out_dir), "reason": "no_field_coords"}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = 0
    grid_rows = []
    for tid, g in player_metrics.groupby("track_id"):
        samples = [
            HeatmapSample(
                timestamp_ms=float(r.timestamp_ms),
                x_field=None if pd.isna(r.x_field) else float(r.x_field),
                y_field=None if pd.isna(r.y_field) else float(r.y_field),
                calibration_valid=bool(getattr(r, "valid", True)),
            )
            for r in g.itertuples(index=False)
        ]
        heat = compute_heatmap(samples, HeatmapConfig())
        if heat.used_samples == 0:
            continue
        arr = np.asarray(heat.normalized, dtype=float)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(arr, origin="lower", aspect="auto", cmap="YlOrRd")
        ax.set_title(f"player_{tid}")
        ax.axis("off")
        path = out_dir / f"player_{tid}_position.png"
        fig.savefig(path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        written += 1
        for iy, row in enumerate(arr):
            for ix, val in enumerate(row):
                if val > 0:
                    grid_rows.append(
                        {
                            "global_player_id": tid,
                            "bin_x": ix,
                            "bin_y": iy,
                            "dwell_share": float(val),
                        }
                    )
    if grid_rows:
        pd.DataFrame(grid_rows).to_parquet(out_dir.parent / "player_heatmap.parquet", index=False)
    return {"players": written, "path": str(out_dir)}
