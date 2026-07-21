"""Resolution-independent streaming renderer for canonical analytics artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd


TEAM_COLORS = {
    "team_0": (255, 90, 40),
    "team_1": (40, 90, 255),
    "unknown": (170, 170, 170),
}


def _rows_by_frame(frame: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if frame.empty or "frame_id" not in frame:
        return rows
    for item in frame.to_dict("records"):
        rows[int(item["frame_id"])].append(item)
    return rows


def _nullable_text(value: Any, fallback: str = "?") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    return str(value)


def render_analytics_video(
    input_video: Path,
    output_video: Path,
    tracks: pd.DataFrame,
    identities: pd.DataFrame,
    ball_state: pd.DataFrame,
    possession: pd.DataFrame,
    player_metrics: pd.DataFrame,
    camera_motion: pd.DataFrame,
    calibration: pd.DataFrame,
    layers: Iterable[str],
) -> dict[str, Any]:
    """Render frame-by-frame; video frames are never accumulated in memory."""
    layer_set = set(layers)
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open output writer: {output_video}")

    track_rows = _rows_by_frame(tracks)
    identity_rows = _rows_by_frame(identities)
    ball_rows = _rows_by_frame(ball_state)
    possession_rows = _rows_by_frame(possession)
    metric_rows = _rows_by_frame(player_metrics)
    motion_rows = _rows_by_frame(camera_motion)
    calibration_rows = _rows_by_frame(calibration)
    identity_lookup = {
        (int(row["frame_id"]), int(row["track_id"])): row
        for frame_rows in identity_rows.values()
        for row in frame_rows
        if row.get("track_id") is not None
    }
    metric_lookup = {
        (int(row["frame_id"]), int(row["track_id"])): row
        for frame_rows in metric_rows.values()
        for row in frame_rows
        if row.get("track_id") is not None
    }

    frame_id = 0
    written = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        font_scale = max(0.42, min(width, height) / 1400.0)
        thickness = max(1, int(round(min(width, height) / 720.0)))
        for track in track_rows.get(frame_id, []):
            tid = int(track["track_id"])
            ident = identity_lookup.get((frame_id, tid), {})
            team = _nullable_text(ident.get("team_id"), "unknown")
            color = TEAM_COLORS.get(team, TEAM_COLORS["unknown"])
            x1, y1, x2, y2 = (
                int(track["bbox_x1"]),
                int(track["bbox_y1"]),
                int(track["bbox_x2"]),
                int(track["bbox_y2"]),
            )
            foot = (int((x1 + x2) / 2), y2)
            axes = (max(8, int((x2 - x1) * 0.55)), max(4, int((x2 - x1) * 0.18)))
            cv2.ellipse(image, foot, axes, 0, 200, 340, color, thickness)
            labels = []
            if "track_id" in layer_set:
                labels.append(f"ID {tid}")
            if "role" in layer_set:
                labels.append(_nullable_text(ident.get("role"), "unknown"))
            if "team" in layer_set:
                labels.append(team)
            if "identity_confidence" in layer_set and ident:
                labels.append(f"{float(ident.get('confidence') or 0):.2f}")
            metric = metric_lookup.get((frame_id, tid), {})
            if metric.get("valid"):
                if "speed" in layer_set:
                    labels.append(f"{float(metric.get('smoothed_speed_kmh') or 0):.1f}km/h")
                if "distance" in layer_set:
                    labels.append(f"{float(metric.get('cumulative_distance_m') or 0):.1f}m")
            if labels:
                cv2.putText(
                    image,
                    " ".join(labels),
                    (max(0, x1), max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

        if "ball" in layer_set:
            for ball in ball_rows.get(frame_id, []):
                x, y = ball.get("ball_x_pixel"), ball.get("ball_y_pixel")
                if x is None or y is None or pd.isna(x) or pd.isna(y):
                    continue
                p = (int(x), int(y))
                radius = max(6, int(min(width, height) * 0.008))
                points = np.array(
                    [[p[0], p[1]], [p[0] - radius, p[1] - 2 * radius], [p[0] + radius, p[1] - 2 * radius]]
                )
                cv2.drawContours(image, [points], 0, (0, 255, 255), -1)

        panel_h = max(52, int(height * 0.12))
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
        lines: list[str] = []
        if "possession" in layer_set and possession_rows.get(frame_id):
            row = possession_rows[frame_id][0]
            lines.append(
                f"Possession: {_nullable_text(row.get('possession_state'), 'unknown')} "
                f"({float(row.get('confidence') or 0):.2f})"
            )
        if "camera_motion" in layer_set and motion_rows.get(frame_id):
            row = motion_rows[frame_id][0]
            lines.append(
                f"Camera dx={float(row.get('dx_pixel') or 0):.1f} "
                f"dy={float(row.get('dy_pixel') or 0):.1f} valid={bool(row.get('valid'))}"
            )
        if "calibration" in layer_set:
            row = calibration_rows.get(frame_id, [{}])[0]
            lines.append(
                "Calibration: valid"
                if row.get("valid")
                else f"Calibration: unavailable ({_nullable_text(row.get('invalid_reason'))})"
            )
        if "warnings" in layer_set and not calibration_rows.get(frame_id, [{}])[0].get("valid"):
            lines.append("Spatial speed/distance/team shape withheld")
        for index, line in enumerate(lines[:3]):
            cv2.putText(
                image,
                line,
                (int(width * 0.015), int((index + 1) * panel_h / 3.6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
        writer.write(image)
        written += 1
        frame_id += 1

    writer.release()
    capture.release()
    if written == 0:
        raise RuntimeError("Analytics renderer wrote zero frames")
    return {"frames": written, "fps": fps, "width": width, "height": height}


def render_tactical_preview(
    output_video: Path,
    input_fps: float,
    frame_count: int,
    game_state: pd.DataFrame,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> dict[str, Any]:
    """Render a tactical pitch; invalid frames clearly state no calibration."""
    width, height = 840, 544
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), input_fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create tactical preview: {output_video}")
    grouped = _rows_by_frame(game_state)
    valid_frames = 0
    for frame_id in range(frame_count):
        image = np.full((height, width, 3), (35, 110, 35), dtype=np.uint8)
        margin = 32
        cv2.rectangle(image, (margin, margin), (width - margin, height - margin), (255, 255, 255), 2)
        cv2.line(image, (width // 2, margin), (width // 2, height - margin), (255, 255, 255), 2)
        cv2.circle(image, (width // 2, height // 2), 52, (255, 255, 255), 2)
        rows = [row for row in grouped.get(frame_id, []) if row.get("valid")]
        if rows:
            valid_frames += 1
            for row in rows:
                x = margin + int(float(row["x_field"]) / pitch_length_m * (width - 2 * margin))
                y = margin + int(float(row["y_field"]) / pitch_width_m * (height - 2 * margin))
                color = TEAM_COLORS.get(_nullable_text(row.get("team_id"), "unknown"), (180, 180, 180))
                cv2.circle(image, (x, y), 7, color, -1)
        else:
            cv2.putText(
                image,
                "NO VALID CALIBRATION - POSITIONS WITHHELD",
                (90, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        writer.write(image)
    writer.release()
    return {"frames": frame_count, "valid_frames": valid_frames}
