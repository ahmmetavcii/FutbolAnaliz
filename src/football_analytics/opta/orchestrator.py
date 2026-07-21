"""Opta-like analytics orchestrator (no invented confirmed events)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from football_analytics.events.ball_trajectory import (
    BallTrajectoryConfig,
    load_ball_trajectory_from_ball_state,
    write_ball_trajectory,
)
from football_analytics.events.possession_chain import (
    PossessionChainConfig,
    build_possession_chain,
)
from football_analytics.events.touch_inference import TouchInferenceConfig, infer_touches
from football_analytics.opta.actions import (
    ActionInferenceConfig,
    infer_clearances,
    infer_dribbles,
    infer_duels_from_proximity,
    infer_passes,
    infer_turnovers_and_interceptions,
)
from football_analytics.opta.aggregate import (
    build_opta_player_summary,
    build_opta_team_summary,
    build_player_identity_table,
    enrich_touches_with_ball_pitch,
    export_heatmaps,
)
from football_analytics.opta.pitch_zones import PitchZones, infer_attack_sign_from_positions
from football_analytics.utils.io import write_json


def _read(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def run_opta_analytics(run_dir: Path, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = config or {}
    started = dt.datetime.now(dt.timezone.utc)
    stage_dir = run_dir / "stages" / "opta_analytics"
    stage_dir.mkdir(parents=True, exist_ok=True)

    tracks = _read(run_dir, "tracks.parquet")
    identities = _read(run_dir, "track_identities.parquet")
    ball_state = _read(run_dir, "ball_state.parquet")
    player_metrics = _read(run_dir, "player_metrics.parquet")
    global_map = _read(run_dir, "global_identity_map.parquet")
    game_state = _read(run_dir, "game_state.parquet")

    # Pitch zones / attack direction
    team0_xs: list[float] = []
    team1_xs: list[float] = []
    if not game_state.empty and {"team_id", "x_field"} <= set(game_state.columns):
        for row in game_state.itertuples(index=False):
            if pd.isna(row.x_field):
                continue
            tid = str(getattr(row, "team_id", ""))
            if tid.endswith("0"):
                team0_xs.append(float(row.x_field))
            elif tid.endswith("1"):
                team1_xs.append(float(row.x_field))
    zones = PitchZones(team0_attack_sign=infer_attack_sign_from_positions(team0_xs, team1_xs))
    write_json(run_dir / "pitch_zones.json", zones.to_dict())

    # Ball trajectory
    ball = load_ball_trajectory_from_ball_state(ball_state, config=BallTrajectoryConfig())
    # enrich velocity
    if not ball.empty:
        ball = ball.sort_values("frame_id").reset_index(drop=True)
        vx = [None]
        for i in range(1, len(ball)):
            dt_s = (float(ball.loc[i, "timestamp_ms"]) - float(ball.loc[i - 1, "timestamp_ms"])) / 1000.0
            if dt_s <= 0 or not ball.loc[i, "visible"] or not ball.loc[i - 1, "visible"]:
                vx.append(None)
                continue
            if pd.isna(ball.loc[i, "pitch_x"]) or pd.isna(ball.loc[i - 1, "pitch_x"]):
                vx.append(None)
                continue
            dist = (
                (float(ball.loc[i, "pitch_x"]) - float(ball.loc[i - 1, "pitch_x"])) ** 2
                + (float(ball.loc[i, "pitch_y"]) - float(ball.loc[i - 1, "pitch_y"])) ** 2
            ) ** 0.5
            vx.append(dist / dt_s)
        ball["velocity_mps"] = vx
        ball["direction"] = None
        ball["interpolation_length"] = 0
        ball["camera_id"] = "camera_1"
    write_ball_trajectory(run_dir / "ball_trajectory.parquet", ball)

    # Touches + possession
    touches = infer_touches(
        ball,
        tracks,
        identities=identities,
        global_map=global_map if not global_map.empty else None,
        config=TouchInferenceConfig(),
    )
    touches = enrich_touches_with_ball_pitch(touches, ball)
    touches.to_parquet(run_dir / "touch_events.parquet", index=False)
    chain = build_possession_chain(touches, config=PossessionChainConfig())
    chain.to_parquet(run_dir / "possession_sequences.parquet", index=False)
    # Keep timeline alias from existing possession stage if present
    if (run_dir / "possession_timeline.parquet").is_file():
        pass
    else:
        pd.DataFrame().to_parquet(run_dir / "possession_timeline.parquet", index=False)

    action_cfg = ActionInferenceConfig(**{
        k: v for k, v in (cfg.get("actions") or {}).items()
        if k in ActionInferenceConfig.__dataclass_fields__
    })
    passes = infer_passes(touches, zones=zones, config=action_cfg)
    dribbles = infer_dribbles(touches, zones=zones, config=action_cfg)
    duels = infer_duels_from_proximity(game_state if not game_state.empty else tracks, touches, identities=identities, config=action_cfg)
    clearances = infer_clearances(touches, zones=zones, config=action_cfg)
    defensive = infer_turnovers_and_interceptions(passes, config=action_cfg)
    if not clearances.empty:
        defensive = pd.concat([defensive, clearances], ignore_index=True)

    passes.to_parquet(run_dir / "pass_events.parquet", index=False)
    dribbles.to_parquet(run_dir / "dribble_events.parquet", index=False)
    duels.to_parquet(run_dir / "duel_events.parquet", index=False)
    defensive.to_parquet(run_dir / "defensive_actions.parquet", index=False)

    identity_table = build_player_identity_table(tracks, identities, global_map if not global_map.empty else None)
    identity_flags = list(getattr(identity_table, "attrs", {}).get("identity_flags", []))
    identity_table.to_parquet(run_dir / "global_player_identity.parquet", index=False)

    player_summary = build_opta_player_summary(
        identity_table,
        passes=passes,
        dribbles=dribbles,
        duels=duels,
        defensive=defensive,
        touches=touches,
        player_metrics=player_metrics,
        zones=zones,
    )
    team_summary = build_opta_team_summary(player_summary)
    player_summary.to_csv(run_dir / "player_opta_summary.csv", index=False)
    team_summary.to_csv(run_dir / "team_opta_summary.csv", index=False)

    heat = export_heatmaps(player_metrics, run_dir / "heatmaps")

    def _count(frame: pd.DataFrame, status: str = "confirmed") -> int:
        if frame is None or frame.empty or "status" not in frame.columns:
            return 0
        return int((frame["status"] == status).sum())

    finished = dt.datetime.now(dt.timezone.utc)
    manifest = {
        "status": "PASS",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "product": "opta_like_automatic_video_analytics",
        "identity_flags": identity_flags,
        "ball_frames": int(len(ball)),
        "touch_count": int(len(touches)),
        "pass_confirmed": _count(passes),
        "pass_candidate": _count(passes, "candidate"),
        "dribble_confirmed": _count(dribbles),
        "duel_confirmed": _count(duels),
        "duel_unresolved": _count(duels, "unresolved"),
        "defensive_confirmed": _count(defensive),
        "heatmap_players": heat.get("players", 0),
        "validated_players": int(len(identity_table)),
        "artifact_paths": {
            "pitch_zones": str(run_dir / "pitch_zones.json"),
            "ball_trajectory": str(run_dir / "ball_trajectory.parquet"),
            "touch_events": str(run_dir / "touch_events.parquet"),
            "pass_events": str(run_dir / "pass_events.parquet"),
            "dribble_events": str(run_dir / "dribble_events.parquet"),
            "duel_events": str(run_dir / "duel_events.parquet"),
            "defensive_actions": str(run_dir / "defensive_actions.parquet"),
            "player_opta_summary": str(run_dir / "player_opta_summary.csv"),
            "team_opta_summary": str(run_dir / "team_opta_summary.csv"),
            "heatmaps": str(run_dir / "heatmaps"),
        },
        "error": None,
    }
    write_json(stage_dir / "stage_manifest.json", {**manifest, "stage": "opta_analytics"})
    write_json(stage_dir / "metrics.json", manifest)
    (run_dir / "stage_manifests").mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "stage_manifests" / "opta_analytics.json", manifest)
    return manifest
