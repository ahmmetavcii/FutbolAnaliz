"""Validated geometry primitives used by analytics stages."""

from .bbox import (
    BBox,
    CropQuality,
    canonical_foot_point,
    clip_bbox,
    foot_point_confidence,
    player_crop_quality,
)
from .calibration_state import CalibSource, CalibrationFrameState, CalibrationStateMachine

__all__ = [
    "BBox",
    "CropQuality",
    "canonical_foot_point",
    "clip_bbox",
    "foot_point_confidence",
    "player_crop_quality",
    "CalibSource",
    "CalibrationFrameState",
    "CalibrationStateMachine",
]
