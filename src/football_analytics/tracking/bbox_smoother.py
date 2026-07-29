"""Confidence-weighted EMA bbox smoothing with scene-cut reset.

Keeps raw and smoothed coordinates separate. Association should use raw;
renderer / foot point may use smoothed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BBoxSmoothConfig:
    alpha: float = 0.35  # higher = more responsive
    alpha_size: float = 0.25
    max_center_jump_px: float = 80.0  # if larger, treat as cut/switch → reset


@dataclass
class _State:
    x1: float
    y1: float
    x2: float
    y2: float
    initialized: bool = True


class BBoxSmoother:
    def __init__(self, config: BBoxSmoothConfig | None = None) -> None:
        self.config = config or BBoxSmoothConfig()
        self._st: dict[int, _State] = {}
        self.audit: list[dict[str, Any]] = []

    def reset_all(self) -> None:
        self._st.clear()

    def reset_track(self, track_id: int) -> None:
        self._st.pop(track_id, None)

    def smooth(
        self,
        track_id: int,
        box: tuple[float, float, float, float],
        *,
        frame_idx: int,
        scene_cut: bool = False,
        confidence: float = 1.0,
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = map(float, box)
        if scene_cut:
            self.reset_all()
        if track_id < 0:
            return x1, y1, x2, y2
        if track_id not in self._st:
            self._st[track_id] = _State(x1, y1, x2, y2)
            self.audit.append(
                {
                    "frame_idx": frame_idx,
                    "local_track_id": track_id,
                    "event": "init",
                    "raw_cx": 0.5 * (x1 + x2),
                    "sm_cx": 0.5 * (x1 + x2),
                }
            )
            return x1, y1, x2, y2

        st = self._st[track_id]
        rcx, rcy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        scx, scy = 0.5 * (st.x1 + st.x2), 0.5 * (st.y1 + st.y2)
        jump = ((rcx - scx) ** 2 + (rcy - scy) ** 2) ** 0.5
        if jump > self.config.max_center_jump_px:
            self._st[track_id] = _State(x1, y1, x2, y2)
            self.audit.append(
                {
                    "frame_idx": frame_idx,
                    "local_track_id": track_id,
                    "event": "reset_jump",
                    "jump": jump,
                }
            )
            return x1, y1, x2, y2

        a = min(0.95, max(0.05, self.config.alpha * float(confidence)))
        asz = self.config.alpha_size
        st.x1 = a * x1 + (1 - a) * st.x1
        st.y1 = a * y1 + (1 - a) * st.y1
        st.x2 = a * x2 + (1 - a) * st.x2
        st.y2 = a * y2 + (1 - a) * st.y2
        # mild size blend toward raw size around smoothed center
        rw, rh = x2 - x1, y2 - y1
        cx, cy = 0.5 * (st.x1 + st.x2), 0.5 * (st.y1 + st.y2)
        sw, sh = st.x2 - st.x1, st.y2 - st.y1
        nw = asz * rw + (1 - asz) * sw
        nh = asz * rh + (1 - asz) * sh
        st.x1, st.x2 = cx - nw / 2, cx + nw / 2
        st.y1, st.y2 = cy - nh / 2, cy + nh / 2
        self.audit.append(
            {
                "frame_idx": frame_idx,
                "local_track_id": track_id,
                "event": "smooth",
                "raw_cx": rcx,
                "sm_cx": 0.5 * (st.x1 + st.x2),
                "jump": jump,
            }
        )
        return st.x1, st.y1, st.x2, st.y2
