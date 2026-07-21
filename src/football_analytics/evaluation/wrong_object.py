"""Reject wrong-object ball trajectory locks (shoe/line/board)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WrongObjectConfig:
    max_pixel_speed: float = 2800.0
    max_size_ratio_jump: float = 4.0
    min_confidence_keep: float = 0.12
    max_prediction_frames: int = 8
    board_edge_margin: float = 0.02


def filter_ball_trajectory_candidates(
    detections: pd.DataFrame,
    *,
    frame_width: int = 1920,
    frame_height: int = 1080,
    config: WrongObjectConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Greedy temporal selection; terminate track on impossible jumps."""
    cfg = config or WrongObjectConfig()
    if detections is None or detections.empty:
        return detections, {"wrong_object_switches": 0, "trajectory_jumps": 0, "segments": 0}

    df = detections.sort_values(["frame_id"]).reset_index(drop=True).copy()
    chosen_rows: list[dict[str, Any]] = []
    switches = 0
    jumps = 0
    segments = 1
    prev: dict[str, Any] | None = None
    pred_run = 0

    for fid, group in df.groupby("frame_id", sort=True):
        candidates = []
        for _, row in group.iterrows():
            cx = float(row.get("ball_x_pixel", np.nan))
            cy = float(row.get("ball_y_pixel", np.nan))
            if not np.isfinite(cx) or not np.isfinite(cy):
                continue
            w = float(row.get("bbox_w", 20) or 20)
            h = float(row.get("bbox_h", 20) or 20)
            conf = float(row.get("detection_confidence", 0) or 0)
            # Reject extreme edge board-like boxes with huge width
            mx = cfg.board_edge_margin
            near_edge = (
                cx < frame_width * mx
                or cx > frame_width * (1 - mx)
                or cy < frame_height * mx
                or cy > frame_height * (1 - mx)
            )
            if near_edge and max(w, h) > 80 and conf < 0.4:
                continue
            candidates.append(
                {
                    "frame_id": int(fid),
                    "ball_x_pixel": cx,
                    "ball_y_pixel": cy,
                    "bbox_w": w,
                    "bbox_h": h,
                    "detection_confidence": conf,
                    "detector_source": row.get("detector_source"),
                    "detector_backend": row.get("detector_backend"),
                }
            )
        if not candidates:
            pred_run += 1
            if prev is not None and pred_run > cfg.max_prediction_frames:
                prev = None
                segments += 1
            continue

        if prev is None:
            best = max(candidates, key=lambda c: c["detection_confidence"])
            chosen_rows.append(best)
            prev = best
            pred_run = 0
            continue

        scored = []
        for c in candidates:
            dist = float(
                np.hypot(c["ball_x_pixel"] - prev["ball_x_pixel"], c["ball_y_pixel"] - prev["ball_y_pixel"])
            )
            # Assume ~25fps if timestamps absent
            speed = dist * 25.0
            size_prev = max(prev["bbox_w"] * prev["bbox_h"], 1.0)
            size_now = max(c["bbox_w"] * c["bbox_h"], 1.0)
            size_ratio = max(size_now / size_prev, size_prev / size_now)
            ok = speed <= cfg.max_pixel_speed and size_ratio <= cfg.max_size_ratio_jump
            score = c["detection_confidence"] + 0.35 * max(0.0, 1.0 - dist / 200.0)
            scored.append((ok, score, speed, size_ratio, c))

        feasible = [s for s in scored if s[0]]
        if not feasible:
            # Terminate segment — do not lock onto impossible object
            switches += 1
            jumps += 1
            prev = None
            segments += 1
            pred_run = 0
            # Accept only high-conf restart
            restart = [s for s in scored if s[4]["detection_confidence"] >= cfg.min_confidence_keep]
            if restart:
                best = max(restart, key=lambda s: s[1])[4]
                chosen_rows.append(best)
                prev = best
            continue

        best = max(feasible, key=lambda s: s[1])
        if best[2] > cfg.max_pixel_speed * 0.85:
            jumps += 1
        chosen_rows.append(best[4])
        prev = best[4]
        pred_run = 0

    out = pd.DataFrame(chosen_rows)
    metrics = {
        "wrong_object_switches": int(switches),
        "trajectory_jumps": int(jumps),
        "segments": int(segments),
        "selected_detections": int(len(out)),
        "input_detections": int(len(df)),
    }
    return out, metrics
