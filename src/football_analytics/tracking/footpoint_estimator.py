"""Footpoint estimation: prefer reliable bottom-center with quality scoring.

Pose/segmentation optional hooks — default to smoothed bbox bottom-center when
pose confidence unavailable (no new heavy deps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FootpointResult:
    method: str
    confidence: float
    raw_foot_x: float
    raw_foot_y: float
    stabilized_foot_x: float
    stabilized_foot_y: float


class FootpointEstimator:
    def __init__(self) -> None:
        self._ema: dict[int, tuple[float, float]] = {}

    def reset(self) -> None:
        self._ema.clear()

    def estimate(
        self,
        tid: int,
        box: tuple[float, float, float, float],
        *,
        smoothed_box: tuple[float, float, float, float] | None = None,
        ankle_mid: tuple[float, float, float] | None = None,  # x,y,conf
        seg_ground: tuple[float, float, float] | None = None,
        alpha: float = 0.22,
    ) -> FootpointResult:
        x1, y1, x2, y2 = box
        raw_x, raw_y = 0.5 * (x1 + x2), float(y2)
        method = "bbox_bottom_center"
        conf = 0.55
        fx, fy = raw_x, raw_y

        if ankle_mid is not None and ankle_mid[2] >= 0.5:
            fx, fy, conf = float(ankle_mid[0]), float(ankle_mid[1]), float(ankle_mid[2])
            method = "ankle_midpoint"
        elif seg_ground is not None and seg_ground[2] >= 0.4:
            fx, fy, conf = float(seg_ground[0]), float(seg_ground[1]), float(seg_ground[2])
            method = "segmentation_ground"
        elif smoothed_box is not None:
            sx1, sy1, sx2, sy2 = smoothed_box
            fx, fy = 0.5 * (sx1 + sx2), float(sy2)
            method = "smoothed_bbox_bottom_center"
            conf = 0.65

        if tid >= 0:
            if tid in self._ema:
                px, py = self._ema[tid]
                fx = alpha * fx + (1 - alpha) * px
                fy = alpha * fy + (1 - alpha) * py
            self._ema[tid] = (fx, fy)

        return FootpointResult(method, conf, raw_x, raw_y, fx, fy)
