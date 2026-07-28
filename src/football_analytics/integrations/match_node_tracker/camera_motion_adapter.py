"""Camera-motion helper adapted from Match-node-tracker SpeedTracker._cam.

Upstream source: third_party/authorized/match-node-tracker/speed_tracker.py
Upstream commit: 2777aa3f1e9cc563eba07a675cebdf4bfd9306bf
Changes vs upstream:
  - separated from bbox-height speed / max_kmh hard-cap (those are rejected)
  - returns per-frame affine dx/dy instead of mutating cumulative speed state
  - optional only; production CameraMotionEstimator remains default
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

UPSTREAM_COMMIT = "2777aa3f1e9cc563eba07a675cebdf4bfd9306bf"
BBox = tuple[float, float, float, float]


@dataclass
class MatchNodeCameraMotionConfig:
    enabled: bool = False
    max_corners: int = 300
    quality_level: float = 0.01
    min_distance: float = 10.0
    block_size: int = 7
    bbox_pad_px: int = 15
    min_points: int = 10


@dataclass(frozen=True)
class AffineMotion:
    dx: float = 0.0
    dy: float = 0.0
    valid: bool = False
    inliers: int = 0


class MatchNodeCameraMotionAdapter:
    """Player-masked LK + RANSAC affine (upstream _cam logic)."""

    def __init__(self, config: MatchNodeCameraMotionConfig | None = None) -> None:
        self.config = config or MatchNodeCameraMotionConfig()
        self._prev_gray: NDArray[np.uint8] | None = None

    def reset(self) -> None:
        self._prev_gray = None

    def update(self, frame_bgr: np.ndarray, boxes: list[BBox]) -> AffineMotion:
        if not self.config.enabled:
            return AffineMotion()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        motion = AffineMotion()
        if self._prev_gray is not None:
            h, w = gray.shape
            mask = np.full((h, w), 255, np.uint8)
            pad = self.config.bbox_pad_px
            for x1, y1, x2, y2 in boxes:
                mask[
                    max(0, int(y1) - pad) : min(h, int(y2) + pad),
                    max(0, int(x1) - pad) : min(w, int(x2) + pad),
                ] = 0
            pts = cv2.goodFeaturesToTrack(
                self._prev_gray,
                self.config.max_corners,
                self.config.quality_level,
                self.config.min_distance,
                mask=mask,
                blockSize=self.config.block_size,
            )
            if pts is not None and len(pts) >= self.config.min_points:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, pts, None)
                ok = st.ravel().astype(bool)
                if ok.sum() >= self.config.min_points:
                    # Upstream estimates transform mapping nxt->prev (camera compensation)
                    M, inliers = cv2.estimateAffinePartial2D(
                        nxt[ok], pts[ok], method=cv2.RANSAC
                    )
                    if M is not None:
                        nin = int(inliers.sum()) if inliers is not None else 0
                        motion = AffineMotion(
                            dx=float(M[0, 2]),
                            dy=float(M[1, 2]),
                            valid=True,
                            inliers=nin,
                        )
        self._prev_gray = gray
        return motion
