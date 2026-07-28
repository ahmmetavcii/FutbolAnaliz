"""Track-level team color adapter inspired by Match-node-tracker jersey_color.

Upstream source: third_party/authorized/match-node-tracker/custom_markers.py
  (jersey_color, TeamTracker)
Upstream commit: 2777aa3f1e9cc563eba07a675cebdf4bfd9306bf
Changes vs upstream:
  - frame-level assignment rejected; votes accumulate per track_id
  - blur / min-bbox / unresolved team handling added
  - no hard-cap of 11 players per team
  - does not replace SigLIP production team identity by default
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

UPSTREAM_COMMIT = "2777aa3f1e9cc563eba07a675cebdf4bfd9306bf"


@dataclass
class MatchNodeTeamColorConfig:
    enabled: bool = False
    ema_alpha: float = 0.08
    min_bbox_h: int = 32
    min_non_green_pixels: int = 20
    max_laplacian_var_for_blur_reject: float = 15.0
    hsv_green_low: tuple[int, int, int] = (35, 40, 40)
    hsv_green_high: tuple[int, int, int] = (85, 255, 255)


@dataclass
class _TrackColorState:
    votes: list[np.ndarray] = field(default_factory=list)
    team_id: int | None = None


class MatchNodeTeamColorAdapter:
    """Optional HSV jersey-color team hint at track level."""

    def __init__(self, config: MatchNodeTeamColorConfig | None = None) -> None:
        self.config = config or MatchNodeTeamColorConfig()
        self._centers: np.ndarray | None = None
        self._tracks: dict[int, _TrackColorState] = defaultdict(_TrackColorState)
        self.stats = {
            "valid_crops": 0,
            "rejected_blur_crops": 0,
            "rejected_small_crops": 0,
            "rejected_green_or_empty": 0,
        }

    def jersey_bgr(
        self, frame_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> tuple[float, float, float] | None:
        """Upstream jersey_color with blur/size gates.

        Upstream: custom_markers.jersey_color — upper-half crop, HSV green mask,
        mean BGR of non-green pixels.
        """
        if not self.config.enabled:
            return None
        if (y2 - y1) < self.config.min_bbox_h or (x2 - x1) < 8:
            self.stats["rejected_small_crops"] += 1
            return None
        crop = frame_bgr[y1 : (y1 + y2) // 2, x1:x2]
        if crop.size == 0:
            self.stats["rejected_small_crops"] += 1
            return None
        if cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < (
            self.config.max_laplacian_var_for_blur_reject
        ):
            self.stats["rejected_blur_crops"] += 1
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(
            hsv,
            np.array(self.config.hsv_green_low, dtype=np.uint8),
            np.array(self.config.hsv_green_high, dtype=np.uint8),
        )
        mask = green.ravel() == 0
        ng = crop.reshape(-1, 3)[mask]
        if len(ng) < self.config.min_non_green_pixels:
            self.stats["rejected_green_or_empty"] += 1
            return None
        mean = ng.mean(0)
        self.stats["valid_crops"] += 1
        return float(mean[0]), float(mean[1]), float(mean[2])

    @property
    def centers_bgr(self) -> np.ndarray | None:
        return None if self._centers is None else self._centers.copy()

    def update_track(
        self, track_id: int, color_bgr: tuple[float, float, float] | None
    ) -> int | None:
        """Accumulate color votes; assign via 2-means with EMA centers."""
        if not self.config.enabled or color_bgr is None:
            return None
        st = self._tracks[track_id]
        st.votes.append(np.asarray(color_bgr, dtype=np.float32))
        # Refit centers from all track medians when enough tracks have votes
        medians = []
        tids = []
        for tid, s in self._tracks.items():
            if s.votes:
                medians.append(np.median(np.stack(s.votes), axis=0))
                tids.append(tid)
        if len(medians) < 2:
            return None
        pts = np.stack(medians).astype(np.float32)
        if self._centers is None:
            _, _, self._centers = cv2.kmeans(
                pts,
                2,
                None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
                5,
                cv2.KMEANS_PP_CENTERS,
            )
        labels = np.linalg.norm(pts[:, None] - self._centers, axis=2).argmin(axis=1)
        for t in range(2):
            m = labels == t
            if m.any():
                self._centers[t] = (1 - self.config.ema_alpha) * self._centers[
                    t
                ] + self.config.ema_alpha * pts[m].mean(0)
        for tid, lab in zip(tids, labels):
            self._tracks[tid].team_id = int(lab)
        return self._tracks[track_id].team_id

    def team_of(self, track_id: int) -> int | None:
        if not self.config.enabled:
            return None
        return self._tracks[track_id].team_id
