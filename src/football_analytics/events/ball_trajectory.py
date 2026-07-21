"""Ball trajectory export for event detection (from existing ball_state artifacts)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BALL_TRAJECTORY_COLUMNS = [
    "frame_id",
    "timestamp_ms",
    "ball_x_pixel",
    "ball_y_pixel",
    "pitch_x",
    "pitch_y",
    "ball_confidence",
    "visible",
    "interpolated",
    "source_camera",
]


@dataclass(frozen=True)
class BallTrajectoryConfig:
    max_interp_gap_frames: int = 5
    source_camera: str = "camera_1"


def load_ball_trajectory_from_ball_state(
    ball_state: pd.DataFrame,
    *,
    config: BallTrajectoryConfig | None = None,
) -> pd.DataFrame:
    """Map MVP-2 ``ball_state.parquet`` into the event ball-trajectory schema."""
    cfg = config or BallTrajectoryConfig()
    if ball_state is None or ball_state.empty:
        return pd.DataFrame(columns=BALL_TRAJECTORY_COLUMNS)

    rows: list[dict[str, Any]] = []
    last_visible: dict[str, Any] | None = None
    gap = 0
    for row in ball_state.sort_values("frame_id").itertuples(index=False):
        visibility = str(getattr(row, "visibility_state", "") or "")
        visible = visibility in {"detected", "predicted", "occluded_short", "airborne"}
        interpolated = visibility in {"predicted", "occluded_short"}
        if visible:
            gap = 0
        else:
            gap += 1
        if not visible and gap > cfg.max_interp_gap_frames:
            interpolated = False
            bx = by = px = py = np.nan
            conf = 0.0
        else:
            bx = float(getattr(row, "ball_x_pixel", np.nan))
            by = float(getattr(row, "ball_y_pixel", np.nan))
            px = float(getattr(row, "ball_x_field", np.nan))
            py = float(getattr(row, "ball_y_field", np.nan))
            conf = float(getattr(row, "detection_confidence", 0.0) or 0.0)
            if not visible and last_visible is not None and gap <= cfg.max_interp_gap_frames:
                interpolated = True
                bx = last_visible["ball_x_pixel"]
                by = last_visible["ball_y_pixel"]
                px = last_visible["pitch_x"]
                py = last_visible["pitch_y"]
                conf = min(conf, 0.25)
        record = {
            "frame_id": int(row.frame_id),
            "timestamp_ms": float(getattr(row, "timestamp_ms", 0.0) or 0.0),
            "ball_x_pixel": bx,
            "ball_y_pixel": by,
            "pitch_x": px,
            "pitch_y": py,
            "ball_confidence": conf,
            "visible": bool(visible or interpolated),
            "interpolated": bool(interpolated and not visibility == "detected"),
            "source_camera": cfg.source_camera,
        }
        if visibility == "detected":
            last_visible = record
        rows.append(record)
    return pd.DataFrame(rows, columns=BALL_TRAJECTORY_COLUMNS)


def write_ball_trajectory(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
