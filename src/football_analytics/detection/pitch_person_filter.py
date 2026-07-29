"""Pitch / stands / bench zone classification for person detections.

Uses foot bottom-center (not bbox center), homography pitch bounds, local green
ratio, and broadcast region priors. Close-up / invalid pitch → UNKNOWN (no
blind reject).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np


class PersonZone(str, Enum):
    ON_PITCH = "ON_PITCH"
    PITCH_MARGIN = "PITCH_MARGIN"
    BENCH_TECHNICAL_AREA = "BENCH_TECHNICAL_AREA"
    STANDS = "STANDS"
    OFF_FIELD = "OFF_FIELD"
    UNKNOWN = "UNKNOWN"


class FilterDecision(str, Enum):
    ACCEPT = "ACCEPT"
    ACCEPT_MARGIN = "ACCEPT_MARGIN"
    EXCLUDE_FROM_PLAYER_TRACKING = "EXCLUDE_FROM_PLAYER_TRACKING"
    HOLD_UNKNOWN = "HOLD_UNKNOWN"


@dataclass
class PitchPersonFilterConfig:
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    margin_m: float = 3.0
    green_hue_low: int = 35
    green_hue_high: int = 85
    green_sat_min: int = 40
    green_val_min: int = 40
    foot_patch: int = 24
    stands_top_frac: float = 0.18
    bench_side_frac: float = 0.12
    bench_bottom_frac: float = 0.22
    min_green_on_pitch: float = 0.12
    wide_green_min: float = 0.28
    closeup_green_max: float = 0.15
    min_bbox_h_for_stands_reject: int = 28


@dataclass
class ZoneResult:
    zone: PersonZone
    zone_confidence: float
    pitch_mask_valid: bool
    filter_decision: FilterDecision
    decision_reason: str
    pitch_x: float | None = None
    pitch_y: float | None = None
    local_green: float = 0.0
    shot_hint: str = "unknown"


class PitchPersonFilter:
    """Per-frame person zone assignment."""

    def __init__(
        self,
        homography: np.ndarray | None,
        frame_wh: tuple[int, int],
        config: PitchPersonFilterConfig | None = None,
    ) -> None:
        self.config = config or PitchPersonFilterConfig()
        self.w, self.h = int(frame_wh[0]), int(frame_wh[1])
        self.H = None if homography is None else np.asarray(homography, dtype=np.float64)
        self.H_inv = None
        self.pitch_poly_img: np.ndarray | None = None
        if self.H is not None:
            try:
                self.H_inv = np.linalg.inv(self.H)
                corners = np.array(
                    [
                        [0, 0],
                        [self.config.pitch_length_m, 0],
                        [self.config.pitch_length_m, self.config.pitch_width_m],
                        [0, self.config.pitch_width_m],
                    ],
                    dtype=np.float64,
                )
                ones = np.ones((4, 1))
                pix = (self.H_inv @ np.hstack([corners, ones]).T).T
                pix = pix[:, :2] / np.maximum(pix[:, 2:3], 1e-9)
                self.pitch_poly_img = pix.astype(np.float32)
            except Exception:
                self.H_inv = None
                self.pitch_poly_img = None

        # Static region masks (image space priors for broadcast)
        self.stands_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        top = int(self.h * self.config.stands_top_frac)
        self.stands_mask[:top, :] = 255
        self.bench_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        side = int(self.w * self.config.bench_side_frac)
        bot = int(self.h * (1.0 - self.config.bench_bottom_frac))
        self.bench_mask[bot:, :side] = 255
        self.bench_mask[bot:, self.w - side :] = 255

    def frame_green_ratio(self, frame_bgr: np.ndarray) -> float:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            (self.config.green_hue_low, self.config.green_sat_min, self.config.green_val_min),
            (self.config.green_hue_high, 255, 255),
        )
        return float(np.count_nonzero(mask)) / float(mask.size)

    def shot_hint(self, frame_green: float) -> str:
        if frame_green >= self.config.wide_green_min:
            return "wide"
        if frame_green <= self.config.closeup_green_max:
            return "closeup_or_replay"
        return "medium"

    def _local_green(self, frame_bgr: np.ndarray, fx: float, fy: float) -> float:
        p = self.config.foot_patch
        x0, y0 = max(0, int(fx) - p), max(0, int(fy) - p)
        x1, y1 = min(self.w, int(fx) + p), min(self.h, int(fy) + p)
        patch = frame_bgr[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0
        return self.frame_green_ratio(patch)

    def _to_pitch(self, fx: float, fy: float) -> tuple[float, float] | None:
        if self.H is None:
            return None
        pt = np.array([[[fx, fy]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, self.H)[0, 0]
        if not np.isfinite(out).all():
            return None
        return float(out[0]), float(out[1])

    def classify(
        self,
        frame_bgr: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        frame_green: float | None = None,
    ) -> ZoneResult:
        fx = 0.5 * (x1 + x2)
        fy = float(y2)  # bottom-center foot
        bh = max(1.0, y2 - y1)
        bw = max(1.0, x2 - x1)
        fg = self.frame_green_ratio(frame_bgr) if frame_green is None else float(frame_green)
        shot = self.shot_hint(fg)
        local_g = self._local_green(frame_bgr, fx, fy)
        pitch_xy = self._to_pitch(fx, fy)
        pitch_valid = pitch_xy is not None and self.pitch_poly_img is not None

        # Close-up / invalid: do not blind-filter
        if shot == "closeup_or_replay" or not pitch_valid:
            return ZoneResult(
                PersonZone.UNKNOWN,
                0.35,
                pitch_valid,
                FilterDecision.HOLD_UNKNOWN,
                "closeup_or_invalid_pitch_mask",
                None if pitch_xy is None else pitch_xy[0],
                None if pitch_xy is None else pitch_xy[1],
                local_g,
                shot,
            )

        assert pitch_xy is not None
        px, py = pitch_xy
        L, W, m = self.config.pitch_length_m, self.config.pitch_width_m, self.config.margin_m
        in_pitch = 0 <= px <= L and 0 <= py <= W
        in_margin = (-m <= px <= L + m and -m <= py <= W + m) and not in_pitch

        ix, iy = int(np.clip(fx, 0, self.w - 1)), int(np.clip(fy, 0, self.h - 1))
        in_stands_prior = self.stands_mask[iy, ix] > 0
        in_bench_prior = self.bench_mask[iy, ix] > 0

        # Strong stands: outside pitch + stands prior + low green + small box
        if (not in_pitch) and in_stands_prior and local_g < 0.08 and bh < self.h * 0.12:
            return ZoneResult(
                PersonZone.STANDS,
                0.85,
                True,
                FilterDecision.EXCLUDE_FROM_PLAYER_TRACKING,
                "stands_prior+off_pitch+low_green+small",
                px,
                py,
                local_g,
                shot,
            )

        if in_pitch and local_g >= self.config.min_green_on_pitch:
            return ZoneResult(
                PersonZone.ON_PITCH,
                0.8,
                True,
                FilterDecision.ACCEPT,
                "homography_inside+green",
                px,
                py,
                local_g,
                shot,
            )

        if in_pitch and local_g < self.config.min_green_on_pitch:
            # inside H but weak green — margin-like caution
            return ZoneResult(
                PersonZone.PITCH_MARGIN,
                0.55,
                True,
                FilterDecision.ACCEPT_MARGIN,
                "inside_H_but_low_local_green",
                px,
                py,
                local_g,
                shot,
            )

        if in_margin:
            return ZoneResult(
                PersonZone.PITCH_MARGIN,
                0.65,
                True,
                FilterDecision.ACCEPT_MARGIN,
                "pitch_margin_band",
                px,
                py,
                local_g,
                shot,
            )

        if in_bench_prior or (fy > self.h * 0.75 and (fx < self.w * 0.15 or fx > self.w * 0.85)):
            return ZoneResult(
                PersonZone.BENCH_TECHNICAL_AREA,
                0.7,
                True,
                FilterDecision.EXCLUDE_FROM_PLAYER_TRACKING,
                "bench_technical_prior",
                px,
                py,
                local_g,
                shot,
            )

        if not in_pitch and (in_stands_prior or py < -5 or py > W + 5 or px < -8 or px > L + 8):
            return ZoneResult(
                PersonZone.STANDS if in_stands_prior else PersonZone.OFF_FIELD,
                0.75,
                True,
                FilterDecision.EXCLUDE_FROM_PLAYER_TRACKING,
                "off_pitch_far",
                px,
                py,
                local_g,
                shot,
            )

        return ZoneResult(
            PersonZone.OFF_FIELD,
            0.55,
            True,
            FilterDecision.EXCLUDE_FROM_PLAYER_TRACKING,
            "off_pitch_default",
            px,
            py,
            local_g,
            shot,
        )

    def as_row(
        self,
        *,
        frame_idx: int,
        detection_id: str,
        foot_x: float,
        foot_y: float,
        result: ZoneResult,
    ) -> dict[str, Any]:
        return {
            "frame_idx": int(frame_idx),
            "detection_id": detection_id,
            "foot_x": float(foot_x),
            "foot_y": float(foot_y),
            "zone": result.zone.value,
            "zone_confidence": float(result.zone_confidence),
            "pitch_mask_valid": bool(result.pitch_mask_valid),
            "filter_decision": result.filter_decision.value,
            "decision_reason": result.decision_reason,
            "pitch_x": result.pitch_x,
            "pitch_y": result.pitch_y,
            "local_green": result.local_green,
            "shot_hint": result.shot_hint,
        }
