"""Conservative player–ball touch inference for event attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


TOUCH_COLUMNS = [
    "touch_id",
    "frame_id",
    "timestamp_ms",
    "global_player_id",
    "track_id",
    "team_id",
    "confidence",
    "controlled_touch",
    "deflection",
    "candidate_only",
    "distance_px",
    "distance_m",
]


@dataclass(frozen=True)
class TouchInferenceConfig:
    max_foot_distance_px: float = 55.0
    max_pitch_distance_m: float = 2.5
    min_ball_confidence: float = 0.15
    min_touch_confidence: float = 0.35
    speed_change_bonus: float = 0.15


def infer_touches(
    ball: pd.DataFrame,
    tracks: pd.DataFrame,
    *,
    identities: pd.DataFrame | None = None,
    global_map: pd.DataFrame | None = None,
    config: TouchInferenceConfig | None = None,
) -> pd.DataFrame:
    """Emit touch candidates; never invent contacts when the ball is invisible."""
    cfg = config or TouchInferenceConfig()
    if ball is None or ball.empty or tracks is None or tracks.empty:
        return pd.DataFrame(columns=TOUCH_COLUMNS)

    team_by_track: dict[int, Any] = {}
    if identities is not None and not identities.empty:
        for track_id, group in identities.groupby("track_id"):
            assigned = group[group["team_id"].notna()]
            if not assigned.empty:
                team_by_track[int(track_id)] = assigned.iloc[-1]["team_id"]

    global_by_track: dict[int, Any] = {}
    if global_map is not None and not global_map.empty:
        for row in global_map.itertuples(index=False):
            global_by_track[int(row.local_track_id)] = row.global_id

    person = tracks[tracks.get("object_type", "person") == "person"].copy() if "object_type" in tracks.columns else tracks
    by_frame = {int(k): v for k, v in person.groupby("frame_id")}
    ball_sorted = ball.sort_values("frame_id")
    prev_speed: float | None = None
    prev_xy: tuple[float, float] | None = None
    rows: list[dict[str, Any]] = []
    touch_id = 0

    for row in ball_sorted.itertuples(index=False):
        if not bool(getattr(row, "visible", False)):
            prev_speed = None
            prev_xy = None
            continue
        # Prefer explicit confidence; allow visible tracked ball with weak conf
        # when coordinates are finite (short-gap predictions).
        ball_conf = float(getattr(row, "ball_confidence", 0.0) or 0.0)
        bx = float(row.ball_x_pixel)
        by = float(row.ball_y_pixel)
        if not np.isfinite(bx) or not np.isfinite(by):
            continue
        if ball_conf < cfg.min_ball_confidence and ball_conf > 0.0:
            continue
        if ball_conf <= 0.0 and not bool(getattr(row, "interpolated", False)):
            # Zero-conf non-interpolated visible: keep if marked visible by tracker
            ball_conf = 0.15
        elif ball_conf <= 0.0:
            ball_conf = 0.10
        if ball_conf < cfg.min_ball_confidence * 0.5:
            continue
        speed = None
        if prev_xy is not None:
            speed = float(np.hypot(bx - prev_xy[0], by - prev_xy[1]))
        speed_delta = 0.0
        if speed is not None and prev_speed is not None:
            speed_delta = abs(speed - prev_speed)
        prev_xy = (bx, by)
        prev_speed = speed if speed is not None else prev_speed

        frame_tracks = by_frame.get(int(row.frame_id))
        if frame_tracks is None or frame_tracks.empty:
            continue
        best = None
        for track in frame_tracks.itertuples(index=False):
            foot_x = float(getattr(track, "foot_x_pixel", (track.bbox_x1 + track.bbox_x2) / 2.0))
            foot_y = float(getattr(track, "foot_y_pixel", track.bbox_y2))
            dist_px = float(np.hypot(foot_x - bx, foot_y - by))
            if dist_px > cfg.max_foot_distance_px:
                continue
            pitch_dist = np.nan
            px = float(getattr(row, "pitch_x", np.nan))
            py = float(getattr(row, "pitch_y", np.nan))
            # Field distance only if both sides have pitch coords (optional).
            if np.isfinite(px) and np.isfinite(py) and hasattr(track, "x_field"):
                tx = float(track.x_field)
                ty = float(track.y_field)
                if np.isfinite(tx) and np.isfinite(ty):
                    pitch_dist = float(np.hypot(tx - px, ty - py))
                    if pitch_dist > cfg.max_pitch_distance_m:
                        continue
            proximity = max(0.0, 1.0 - dist_px / cfg.max_foot_distance_px)
            conf = 0.55 * proximity + 0.25 * float(ball_conf)
            if speed_delta > 8.0:
                conf += cfg.speed_change_bonus
            conf = float(min(1.0, conf))
            candidate = (conf, dist_px, pitch_dist, track)
            if best is None or conf > best[0]:
                best = candidate
        if best is None or best[0] < cfg.min_touch_confidence:
            continue
        conf, dist_px, pitch_dist, track = best
        track_id = int(track.track_id)
        touch_id += 1
        rows.append(
            {
                "touch_id": f"touch-{touch_id:05d}",
                "frame_id": int(row.frame_id),
                "timestamp_ms": float(row.timestamp_ms),
                "global_player_id": global_by_track.get(track_id),
                "track_id": track_id,
                "team_id": team_by_track.get(track_id),
                "confidence": conf,
                "controlled_touch": conf >= 0.55 and speed_delta >= 5.0,
                "deflection": conf < 0.55 and speed_delta >= 12.0,
                "candidate_only": conf < 0.65,
                "distance_px": dist_px,
                "distance_m": pitch_dist if np.isfinite(pitch_dist) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=TOUCH_COLUMNS)
