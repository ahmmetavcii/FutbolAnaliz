"""Tactical pitch-map video exporter (streaming, one synthesized frame at a time)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

TEAM_COLORS = {
    "team_0": (255, 90, 40),
    "team_1": (40, 90, 255),
    "referee": (0, 220, 255),
    "unknown": (170, 170, 170),
}

_PITCH_GREEN = (35, 110, 35)
_LINE_COLOR = (255, 255, 255)


def _draw_pitch(width: int, height: int, margin: int) -> np.ndarray:
    image = np.full((height, width, 3), _PITCH_GREEN, dtype=np.uint8)
    left, top = margin, margin
    right, bottom = width - margin, height - margin
    pitch_w = right - left
    pitch_h = bottom - top
    cv2.rectangle(image, (left, top), (right, bottom), _LINE_COLOR, 2)
    cv2.line(image, (width // 2, top), (width // 2, bottom), _LINE_COLOR, 2)
    cv2.circle(image, (width // 2, height // 2), int(pitch_h * 0.17), _LINE_COLOR, 2)
    # Penalty areas (16.5m deep, 40.3m wide on a 105x68 pitch).
    box_depth = int(pitch_w * 16.5 / 105.0)
    box_half = int(pitch_h * 20.15 / 68.0)
    for x_edge, direction in ((left, 1), (right, -1)):
        cv2.rectangle(
            image,
            (x_edge, height // 2 - box_half),
            (x_edge + direction * box_depth, height // 2 + box_half),
            _LINE_COLOR,
            2,
        )
    return image


def _format_timestamp(timestamp_ms: float) -> str:
    total_seconds = max(0.0, timestamp_ms) / 1000.0
    minutes, seconds = divmod(int(total_seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


def export_tactical_map_video(
    output_video: Path,
    positions: pd.DataFrame,
    *,
    fps: float,
    frame_count: int,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
    size: tuple[int, int] = (840, 544),
) -> dict[str, Any]:
    """Render a top-down tactical map video from field-coordinate positions.

    ``positions`` needs columns frame_id, x_field, y_field, team_id, valid and
    optionally ``label`` (e.g. jersey number). Frames are synthesized and
    written one at a time; frames without valid calibration state that clearly.
    """
    output_video = Path(output_video)
    if fps <= 0:
        raise ValueError("fps must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    width, height = size
    margin = 32

    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not positions.empty and "frame_id" in positions:
        for row in positions.to_dict("records"):
            rows_by_frame[int(row["frame_id"])].append(row)

    pitch_template = _draw_pitch(width, height, margin)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open tactical map writer: {output_video}")

    valid_frames = 0
    try:
        for frame_id in range(frame_count):
            image = pitch_template.copy()
            rows = [row for row in rows_by_frame.get(frame_id, []) if row.get("valid")]
            if rows:
                valid_frames += 1
                for row in rows:
                    x_field = float(row["x_field"])
                    y_field = float(row["y_field"])
                    x = margin + int(
                        np.clip(x_field / pitch_length_m, 0.0, 1.0) * (width - 2 * margin)
                    )
                    y = margin + int(
                        np.clip(y_field / pitch_width_m, 0.0, 1.0) * (height - 2 * margin)
                    )
                    team = str(row.get("team_id") or "unknown")
                    color = TEAM_COLORS.get(team, TEAM_COLORS["unknown"])
                    cv2.circle(image, (x, y), 8, color, -1)
                    cv2.circle(image, (x, y), 8, (0, 0, 0), 1)
                    label = row.get("label")
                    if label is not None and not pd.isna(label):
                        cv2.putText(
                            image,
                            str(label),
                            (x - 6, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.38,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
            else:
                cv2.putText(
                    image,
                    "NO VALID CALIBRATION - POSITIONS WITHHELD",
                    (int(width * 0.1), height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    _LINE_COLOR,
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                image,
                _format_timestamp(frame_id * 1000.0 / fps),
                (margin, margin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                _LINE_COLOR,
                2,
                cv2.LINE_AA,
            )
            writer.write(image)
    finally:
        writer.release()

    from football_analytics.export.video_exporter import validate_video_export

    validation = validate_video_export(
        output_video, expected_frames=frame_count, expected_size=(width, height)
    )
    return {
        "path": str(output_video),
        "frames": frame_count,
        "valid_frames": valid_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "validation": validation,
    }
