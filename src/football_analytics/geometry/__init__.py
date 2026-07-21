"""Validated geometry primitives used by analytics stages."""

from .bbox import (
    BBox,
    CropQuality,
    canonical_foot_point,
    clip_bbox,
    foot_point_confidence,
    player_crop_quality,
)

__all__ = [
    "BBox",
    "CropQuality",
    "canonical_foot_point",
    "clip_bbox",
    "foot_point_confidence",
    "player_crop_quality",
]
