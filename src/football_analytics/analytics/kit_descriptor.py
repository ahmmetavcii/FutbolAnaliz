"""Coarse jersey/kit color descriptors adapted from Stage 5B kit measurement.

Ported from the sibling football-analytics kit pipeline (enesturkoglu2):
normalized center-torso ROI + HSV coarse color families. Used here as the
primary signal for two-team assignment (that sibling repo measured kits but
did not assign teams).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import cv2
import numpy as np

from football_analytics.geometry.bbox import BBox

FAMILY_ORDER: tuple[str, ...] = (
    "black",
    "gray",
    "white",
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "magenta",
)
CHROMATIC_FAMILIES: tuple[str, ...] = (
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "magenta",
)

# OpenCV hue 0..179 coverage (non-overlapping), same bins as kit_descriptor_stage5b.
_DEFAULT_HUE_RANGES: dict[str, list[tuple[int, int]]] = {
    "red": [(0, 10), (170, 179)],
    "orange": [(11, 24)],
    "yellow": [(25, 35)],
    "green": [(36, 85)],
    "cyan": [(86, 100)],
    "blue": [(101, 130)],
    "purple": [(131, 150)],
    "magenta": [(151, 169)],
}


def build_hue_family_table(
    hue_ranges: Mapping[str, Sequence[Sequence[int]]] | None = None,
) -> np.ndarray:
    table = np.empty(180, dtype=object)
    ranges = hue_ranges or _DEFAULT_HUE_RANGES
    for family, spans in ranges.items():
        for lo, hi in spans:
            table[int(lo) : int(hi) + 1] = family
    return table


_HUE_FAMILY_TABLE = build_hue_family_table()


def extract_torso_bgr(
    frame_bgr: np.ndarray,
    bbox: BBox | Sequence[float],
    *,
    x_min: float = 0.20,
    x_max: float = 0.80,
    y_min: float = 0.15,
    y_max: float = 0.65,
) -> np.ndarray | None:
    """Crop the normalized center-torso region from a full-frame person box."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
        raise ValueError("frame_bgr must be a uint8 HxWx3 BGR image")
    box = bbox if isinstance(bbox, BBox) else BBox.from_sequence(bbox)
    height, width = frame_bgr.shape[:2]
    clipped = box.clip(width, height)
    if not clipped.is_valid(min_area=1.0):
        return None
    x1 = int(np.floor(clipped.x1 + x_min * clipped.width))
    x2 = int(np.ceil(clipped.x1 + x_max * clipped.width))
    y1 = int(np.floor(clipped.y1 + y_min * clipped.height))
    y2 = int(np.ceil(clipped.y1 + y_max * clipped.height))
    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))
    torso = frame_bgr[y1:y2, x1:x2]
    return torso if torso.size else None


def pitch_mask_bgr(torso_bgr: np.ndarray) -> np.ndarray:
    """G-dominant pitch mask — keeps yellow kits (R≈G) unlike HSV green masks."""
    blue, green, red = cv2.split(torso_bgr)
    g16 = green.astype(np.int16)
    return (g16 > red.astype(np.int16) + 18) & (g16 > blue.astype(np.int16) + 18) & (
        green > 40
    )


def compute_kit_family_fractions(
    torso_bgr: np.ndarray,
    *,
    achromatic_saturation_max: int = 40,
    black_value_max: int = 50,
    white_value_min: int = 200,
    mask_pitch: bool = True,
    hue_table: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return L1 family fractions in ``FAMILY_ORDER`` (length 11)."""
    if torso_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    useful = np.ones(hue.shape, dtype=bool)
    if mask_pitch:
        useful &= ~pitch_mask_bgr(torso_bgr)
    if int(np.count_nonzero(useful)) < 16:
        useful = np.ones(hue.shape, dtype=bool)

    h_flat = hue[useful].reshape(-1)
    s_flat = saturation[useful].reshape(-1)
    v_flat = value[useful].reshape(-1)
    table = hue_table if hue_table is not None else _HUE_FAMILY_TABLE
    counts = {name: 0 for name in FAMILY_ORDER}
    for h_val, s_val, v_val in zip(h_flat.tolist(), s_flat.tolist(), v_flat.tolist()):
        if int(s_val) <= achromatic_saturation_max:
            if int(v_val) <= black_value_max:
                counts["black"] += 1
            elif int(v_val) >= white_value_min:
                counts["white"] += 1
            else:
                counts["gray"] += 1
        else:
            counts[str(table[int(h_val)])] += 1
    total = sum(counts.values())
    if total <= 0:
        return None
    return np.asarray(
        [counts[name] / float(total) for name in FAMILY_ORDER], dtype=np.float32
    )


def kit_feature_from_frame(
    frame_bgr: np.ndarray,
    bbox: BBox | Sequence[float],
    *,
    x_min: float = 0.20,
    x_max: float = 0.80,
    y_min: float = 0.15,
    y_max: float = 0.65,
) -> tuple[np.ndarray | None, float]:
    """Extract kit-family feature + useful-pixel fraction for one person box."""
    torso = extract_torso_bgr(
        frame_bgr, bbox, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )
    if torso is None:
        return None, 0.0
    pitch = pitch_mask_bgr(torso)
    useful_fraction = float(np.mean(~pitch)) if pitch.size else 0.0
    fractions = compute_kit_family_fractions(torso, mask_pitch=True)
    if fractions is None:
        return None, useful_fraction
    return fractions, useful_fraction


def dominant_family(fractions: np.ndarray) -> str:
    row = np.asarray(fractions, dtype=np.float32).reshape(-1)
    if row.size != len(FAMILY_ORDER):
        raise ValueError("fractions must match FAMILY_ORDER")
    return FAMILY_ORDER[int(np.argmax(row))]


def white_score(fractions: np.ndarray) -> float:
    row = np.asarray(fractions, dtype=np.float32).reshape(-1)
    idx = {name: i for i, name in enumerate(FAMILY_ORDER)}
    return float(row[idx["white"]] + row[idx["gray"]])


def colored_score(fractions: np.ndarray) -> float:
    row = np.asarray(fractions, dtype=np.float32).reshape(-1)
    idx = {name: i for i, name in enumerate(FAMILY_ORDER)}
    # Prefer yellow/orange (common away kits) then other chromatic families.
    return float(
        row[idx["yellow"]]
        + row[idx["orange"]]
        + 0.5 * sum(row[idx[name]] for name in CHROMATIC_FAMILIES if name not in ("yellow", "orange", "green"))
        + 0.25 * row[idx["green"]]
    )


def is_dark_kit_fractions(fractions: np.ndarray) -> bool:
    """Black referee / dark GK kits that must not define team centres."""
    row = np.asarray(fractions, dtype=np.float32).reshape(-1)
    idx = {name: i for i, name in enumerate(FAMILY_ORDER)}
    black = float(row[idx["black"]])
    gray = float(row[idx["gray"]])
    white = float(row[idx["white"]])
    chromatic = float(sum(row[idx[name]] for name in CHROMATIC_FAMILIES))
    if black >= 0.28:
        return True
    if black + gray >= 0.70 and white < 0.18 and chromatic < 0.22:
        return True
    return False


def bbox_contamination(
    target: Sequence[float],
    others: Sequence[Sequence[float]],
) -> float:
    """Union coverage of other person boxes inside the target box (0..1)."""
    x1, y1, x2, y2 = [int(v) for v in target]
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    if not others:
        return 0.0
    mask = np.zeros((height, width), dtype=bool)
    for other in others:
        ox1, oy1, ox2, oy2 = [int(v) for v in other]
        ix1 = max(x1, ox1)
        iy1 = max(y1, oy1)
        ix2 = min(x2, ox2)
        iy2 = min(y2, oy2)
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1] = True
    return float(np.mean(mask))
