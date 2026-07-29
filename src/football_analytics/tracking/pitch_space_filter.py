"""Pitch-space Kalman filter with stationary dead-zone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PitchFilterConfig:
    process_var: float = 0.35
    meas_var_base: float = 0.8
    deadzone_speed_mps: float = 0.55
    deadzone_image_px: float = 14.0
    deadzone_meas_m: float = 1.6
    max_speed_mps: float = 12.0
    miss_hold_frames: int = 8
    stationary_hold_alpha: float = 0.01  # nearly freeze when stationary


@dataclass
class PitchState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    P: np.ndarray | None = None
    age: int = 0
    miss: int = 0


class PitchSpaceFilter:
    """Constant-velocity Kalman in pitch meters."""

    def __init__(self, fps: float = 25.0, config: PitchFilterConfig | None = None) -> None:
        self.fps = fps
        self.dt = 1.0 / max(fps, 1e-6)
        self.config = config or PitchFilterConfig()
        self._tracks: dict[int, PitchState] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def reset_track(self, tid: int) -> None:
        self._tracks.pop(tid, None)

    def update(
        self,
        tid: int,
        meas_xy: tuple[float, float] | None,
        *,
        footpoint_conf: float,
        calib_conf: float,
        det_conf: float,
        image_disp_px: float,
        valid: bool,
    ) -> dict[str, Any]:
        if tid < 0:
            return {
                "raw_pitch_x": None,
                "raw_pitch_y": None,
                "filtered_pitch_x": None,
                "filtered_pitch_y": None,
                "coordinate_confidence": 0.0,
                "used_measurement": False,
            }

        raw_x = raw_y = None
        if meas_xy is not None:
            raw_x, raw_y = float(meas_xy[0]), float(meas_xy[1])

        conf = float(footpoint_conf) * float(calib_conf) * float(max(det_conf, 0.05))
        use_meas = bool(valid and meas_xy is not None and conf >= 0.15 and calib_conf >= 0.35)

        if tid not in self._tracks:
            if not use_meas:
                return {
                    "raw_pitch_x": raw_x,
                    "raw_pitch_y": raw_y,
                    "filtered_pitch_x": None,
                    "filtered_pitch_y": None,
                    "coordinate_confidence": 0.0,
                    "used_measurement": False,
                }
            P = np.eye(4) * 2.0
            self._tracks[tid] = PitchState(raw_x, raw_y, 0.0, 0.0, P, 0, 0)
            return {
                "raw_pitch_x": raw_x,
                "raw_pitch_y": raw_y,
                "filtered_pitch_x": raw_x,
                "filtered_pitch_y": raw_y,
                "coordinate_confidence": conf,
                "used_measurement": True,
            }

        st = self._tracks[tid]
        dt = self.dt
        prev_x, prev_y = float(st.x), float(st.y)
        prev_vx, prev_vy = float(st.vx), float(st.vy)

        # Stationary dead-zone: skip predictive drift; hold last filtered pose.
        stationary = bool(
            use_meas
            and image_disp_px <= self.config.deadzone_image_px
            and float(np.hypot(prev_vx, prev_vy)) <= self.config.deadzone_speed_mps
            and float(np.hypot(raw_x - prev_x, raw_y - prev_y)) <= self.config.deadzone_meas_m
        )

        F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        q = self.config.process_var
        Q = np.diag([q * dt, q * dt, q, q])
        x = np.array([st.x, st.y, st.vx, st.vy], dtype=np.float64)
        P = st.P if st.P is not None else np.eye(4)

        used = False
        if stationary:
            a = self.config.stationary_hold_alpha
            x[0] = (1.0 - a) * prev_x + a * float(raw_x)
            x[1] = (1.0 - a) * prev_y + a * float(raw_y)
            x[2] = 0.0
            x[3] = 0.0
            P = P * 0.95 + np.eye(4) * 0.01
            used = True
            st.miss = 0
        else:
            # predict
            x = F @ x
            P = F @ P @ F.T + Q
            if use_meas:
                R = np.eye(2) * (self.config.meas_var_base / max(conf, 0.05))
                H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
                z = np.array([raw_x, raw_y], dtype=np.float64)
                y = z - H @ x
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (np.eye(4) - K @ H) @ P
                used = True
                # clamp speed
                sp = float(np.hypot(x[2], x[3]))
                if sp > self.config.max_speed_mps:
                    x[2] *= self.config.max_speed_mps / sp
                    x[3] *= self.config.max_speed_mps / sp
                st.miss = 0
            else:
                st.miss += 1
                if st.miss > self.config.miss_hold_frames:
                    self._tracks.pop(tid, None)
                    return {
                        "raw_pitch_x": raw_x,
                        "raw_pitch_y": raw_y,
                        "filtered_pitch_x": None,
                        "filtered_pitch_y": None,
                        "coordinate_confidence": 0.0,
                        "used_measurement": False,
                    }

        st.x, st.y, st.vx, st.vy = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        st.P = P
        st.age += 1
        return {
            "raw_pitch_x": raw_x,
            "raw_pitch_y": raw_y,
            "filtered_pitch_x": st.x,
            "filtered_pitch_y": st.y,
            "coordinate_confidence": conf if used else conf * 0.3,
            "used_measurement": used,
            "vx_mps": st.vx,
            "vy_mps": st.vy,
        }
