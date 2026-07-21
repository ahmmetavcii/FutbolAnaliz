"""Touch inference helpers: ankle proxy + visual contact sheets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def ankle_point_from_bbox(
    bbox_x1: float,
    bbox_y1: float,
    bbox_x2: float,
    bbox_y2: float,
    *,
    side: str = "center",
) -> tuple[float, float]:
    """Approximate ankle/keypoint when pose is unavailable.

    Uses lower bbox edge with a slight inward bias (not the absolute corner).
    """
    cx = (bbox_x1 + bbox_x2) / 2.0
    width = max(1.0, bbox_x2 - bbox_x1)
    ankle_y = bbox_y1 + 0.92 * (bbox_y2 - bbox_y1)
    if side == "left":
        ankle_x = cx - 0.15 * width
    elif side == "right":
        ankle_x = cx + 0.15 * width
    else:
        ankle_x = cx
    return float(ankle_x), float(ankle_y)


def enrich_tracks_with_ankle(tracks: pd.DataFrame) -> pd.DataFrame:
    if tracks is None or tracks.empty:
        return tracks
    out = tracks.copy()
    if {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"} <= set(out.columns):
        xs, ys = [], []
        for row in out.itertuples(index=False):
            ax, ay = ankle_point_from_bbox(
                float(row.bbox_x1), float(row.bbox_y1), float(row.bbox_x2), float(row.bbox_y2)
            )
            xs.append(ax)
            ys.append(ay)
        out["ankle_x_pixel"] = xs
        out["ankle_y_pixel"] = ys
        # Prefer ankle over foot_x when computing touches
        out["foot_x_pixel"] = out["ankle_x_pixel"]
        out["foot_y_pixel"] = out["ankle_y_pixel"]
    return out


def export_touch_contact_sheets(
    touches: pd.DataFrame,
    video_path: Path,
    out_dir: Path,
    *,
    tracks: pd.DataFrame | None = None,
    ball: pd.DataFrame | None = None,
    max_touches: int = 50,
) -> dict[str, Any]:
    """Write per-touch JPG contact sheets for manual review."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if touches is None or touches.empty or not Path(video_path).is_file():
        return {"written": 0, "path": str(out_dir)}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"written": 0, "path": str(out_dir), "error": "cannot_open_video"}

    track_by_frame: dict[tuple[int, int], Any] = {}
    if tracks is not None and not tracks.empty:
        for row in tracks.itertuples(index=False):
            track_by_frame[(int(row.frame_id), int(row.track_id))] = row

    ball_by_frame: dict[int, Any] = {}
    if ball is not None and not ball.empty:
        for row in ball.itertuples(index=False):
            ball_by_frame[int(row.frame_id)] = row

    written = 0
    index_rows = []
    for touch in touches.head(max_touches).itertuples(index=False):
        fid = int(touch.frame_id)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, bgr = cap.read()
        if not ok:
            continue
        canvas = bgr.copy()
        # Ball
        brow = ball_by_frame.get(fid)
        if brow is not None and np.isfinite(getattr(brow, "ball_x_pixel", np.nan)):
            bx, by = int(brow.ball_x_pixel), int(brow.ball_y_pixel)
            cv2.circle(canvas, (bx, by), 8, (0, 255, 255), 2)
        # Player bbox / ankle
        trow = track_by_frame.get((fid, int(touch.track_id)))
        if trow is not None:
            x1, y1 = int(trow.bbox_x1), int(trow.bbox_y1)
            x2, y2 = int(trow.bbox_x2), int(trow.bbox_y2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if hasattr(trow, "ankle_x_pixel") and np.isfinite(trow.ankle_x_pixel):
                cv2.circle(
                    canvas,
                    (int(trow.ankle_x_pixel), int(trow.ankle_y_pixel)),
                    5,
                    (255, 0, 0),
                    -1,
                )
        label = (
            f"{getattr(touch, 'touch_id', written)} "
            f"t={getattr(touch, 'track_id', '?')} "
            f"c={float(getattr(touch, 'confidence', 0)):.2f}"
        )
        cv2.putText(canvas, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        touch_id = str(getattr(touch, "touch_id", f"touch_{written}"))
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in touch_id)
        path = out_dir / f"{safe}_f{fid}.jpg"
        cv2.imwrite(str(path), canvas)
        written += 1
        index_rows.append(
            {
                "touch_id": touch_id,
                "frame_id": fid,
                "track_id": int(touch.track_id),
                "confidence": float(getattr(touch, "confidence", 0.0)),
                "image_path": str(path),
            }
        )
    cap.release()
    if index_rows:
        pd.DataFrame(index_rows).to_csv(out_dir / "touch_review_index.csv", index=False)
    return {"written": written, "path": str(out_dir)}
