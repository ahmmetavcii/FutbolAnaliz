"""Temporally stable camera/homography state for broadcast football.

Avoids per-frame noisy H updates. Supports MEASURED / TEMPORALLY_FILTERED /
PROPAGATED / FROZEN_LAST_GOOD / INVALID. Scene cuts reset state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np


class CalibSource(str, Enum):
    MEASURED = "MEASURED"
    TEMPORALLY_FILTERED = "TEMPORALLY_FILTERED"
    PROPAGATED = "PROPAGATED"
    FROZEN_LAST_GOOD = "FROZEN_LAST_GOOD"
    INVALID = "INVALID"


@dataclass
class CalibrationStateConfig:
    max_frozen_frames: int = 45  # ~1.8s @25fps
    max_corner_jump_px: float = 40.0
    min_confidence: float = 0.35
    ema_alpha: float = 0.15  # keypoint EMA in image space before H rebuild
    require_wide_shot: bool = True


@dataclass
class CalibrationFrameState:
    source: CalibSource
    homography: np.ndarray | None
    confidence: float
    accepted_update: bool
    rejection_reason: str = ""
    corner_jitter_px: float = 0.0
    frozen_age: int = 0


class CalibrationStateMachine:
    """Maintain one H per shot; freeze on bad/missing measurements."""

    def __init__(
        self,
        base_homography: np.ndarray,
        image_size: tuple[int, int],
        pitch_size: tuple[float, float] = (105.0, 68.0),
        config: CalibrationStateConfig | None = None,
    ) -> None:
        self.config = config or CalibrationStateConfig()
        self.w, self.h = image_size
        self.L, self.W = pitch_size
        self.base_H = np.asarray(base_homography, dtype=np.float64)
        self.H = self.base_H.copy()
        self.confidence = 0.7
        self.source = CalibSource.MEASURED
        self.frozen_age = 0
        self._prev_corners: np.ndarray | None = None
        self.audit: list[dict[str, Any]] = []
        self._pitch_corners = np.array(
            [[0, 0], [self.L, 0], [self.L, self.W], [0, self.W]], dtype=np.float64
        )

    def reset(self, reason: str = "scene_cut") -> None:
        self.H = self.base_H.copy()
        self.confidence = 0.4
        self.source = CalibSource.INVALID
        self.frozen_age = 0
        self._prev_corners = None
        self.audit.append({"event": "reset", "reason": reason})

    def _project_corners(self, H: np.ndarray) -> np.ndarray | None:
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        ones = np.ones((4, 1))
        pix = (Hinv @ np.hstack([self._pitch_corners, ones]).T).T
        pix = pix[:, :2] / np.maximum(pix[:, 2:3], 1e-9)
        if not np.all(np.isfinite(pix)):
            return None
        return pix

    def update(
        self,
        *,
        frame_idx: int,
        shot_type: str,
        scene_cut: bool,
        measured_H: np.ndarray | None = None,
        measured_confidence: float = 0.0,
        line_alignment_score: float = 0.0,
    ) -> CalibrationFrameState:
        if scene_cut:
            self.reset("scene_cut")

        shot = shot_type.lower()
        allow_measure = shot in {"main_wide", "main_medium"} and (
            not self.config.require_wide_shot or shot == "main_wide" or measured_confidence >= 0.6
        )

        # Closeup / bench / crowd / replay / unknown → freeze or invalidate
        if shot in {"close_up", "closeup", "bench", "crowd", "replay", "unknown", "graphic"}:
            if self.H is not None and self.frozen_age < self.config.max_frozen_frames and self.confidence >= self.config.min_confidence:
                self.frozen_age += 1
                self.source = CalibSource.FROZEN_LAST_GOOD
                st = CalibrationFrameState(
                    self.source, self.H.copy(), self.confidence * 0.95, False, "shot_gate_freeze", frozen_age=self.frozen_age
                )
            else:
                self.source = CalibSource.INVALID
                self.H = None
                st = CalibrationFrameState(CalibSource.INVALID, None, 0.0, False, "shot_gate_invalid")
            self._log(frame_idx, st, line_alignment_score)
            return st

        accepted = False
        reason = ""
        corner_jitter = 0.0

        if allow_measure and measured_H is not None and measured_confidence >= self.config.min_confidence:
            corners = self._project_corners(measured_H)
            if corners is None:
                reason = "singular_measured_H"
            else:
                if self._prev_corners is not None:
                    corner_jitter = float(np.mean(np.linalg.norm(corners - self._prev_corners, axis=1)))
                    if corner_jitter > self.config.max_corner_jump_px:
                        reason = "corner_jump"
                    else:
                        # EMA blend in image-corner space then rebuild approx via current H trust
                        a = self.config.ema_alpha * measured_confidence
                        # Blend homographies carefully via convex combination of matrices is unstable;
                        # instead keep previous H if jump moderate and only accept strong measurements.
                        if measured_confidence >= 0.55 and corner_jitter < self.config.max_corner_jump_px * 0.6:
                            self.H = measured_H.astype(np.float64)
                            self.confidence = float(measured_confidence)
                            self.source = CalibSource.MEASURED
                            self.frozen_age = 0
                            self._prev_corners = corners
                            accepted = True
                        else:
                            # temporally filtered: keep old H, bump confidence lightly
                            self.source = CalibSource.TEMPORALLY_FILTERED
                            self.confidence = 0.7 * self.confidence + 0.3 * float(measured_confidence)
                            self.frozen_age = 0
                            accepted = False
                            reason = "weak_measure_keep_filtered"
                else:
                    self.H = measured_H.astype(np.float64)
                    self.confidence = float(measured_confidence)
                    self.source = CalibSource.MEASURED
                    self._prev_corners = corners
                    self.frozen_age = 0
                    accepted = True
        else:
            reason = "no_valid_measurement"

        if not accepted:
            if self.H is not None and self.frozen_age < self.config.max_frozen_frames:
                self.frozen_age += 1
                self.source = CalibSource.FROZEN_LAST_GOOD if reason else CalibSource.PROPAGATED
                # decay confidence
                self.confidence *= 0.98
            else:
                self.source = CalibSource.INVALID
                self.H = None
                self.confidence = 0.0

        # For this task we primarily use base/manual H as MEASURED once, then freeze across wide shots
        # with camera-motion compensated footpoints feeding pitch filter.
        if self.H is None and shot == "main_wide":
            self.H = self.base_H.copy()
            self.confidence = 0.55
            self.source = CalibSource.FROZEN_LAST_GOOD
            self.frozen_age = 0
            self._prev_corners = self._project_corners(self.H)

        st = CalibrationFrameState(
            self.source,
            None if self.H is None else self.H.copy(),
            float(self.confidence),
            accepted,
            reason,
            corner_jitter,
            self.frozen_age,
        )
        self._log(frame_idx, st, line_alignment_score)
        return st

    def _log(self, frame_idx: int, st: CalibrationFrameState, line_alignment: float) -> None:
        self.audit.append(
            {
                "frame_idx": frame_idx,
                "calibration_source": st.source.value,
                "confidence": st.confidence,
                "accepted_update": st.accepted_update,
                "rejection_reason": st.rejection_reason,
                "homography_corner_jitter_px": st.corner_jitter_px,
                "frozen_age": st.frozen_age,
                "line_alignment_score": line_alignment,
                "valid": st.homography is not None and st.source != CalibSource.INVALID,
            }
        )

    def pixel_to_pitch(self, x: float, y: float) -> tuple[float, float] | None:
        if self.H is None or self.source == CalibSource.INVALID:
            return None
        pt = np.array([[[x, y]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, self.H)[0, 0]
        if not np.isfinite(out).all():
            return None
        return float(out[0]), float(out[1])
