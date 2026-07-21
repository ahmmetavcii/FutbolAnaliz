"""ReID matching helpers: torso crops, hard-negative calibration, relative scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def torso_crop_xyxy(
    bbox: Sequence[float],
    *,
    image_width: int,
    image_height: int,
    x_margin: float = 0.12,
    y_top: float = 0.12,
    y_bottom: float = 0.62,
) -> tuple[int, int, int, int] | None:
    """Return integer torso crop box focused on jersey / upper body."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width < 2.0 or height < 2.0:
        return None
    cx1 = int(np.floor(x1 + x_margin * width))
    cx2 = int(np.ceil(x2 - x_margin * width))
    cy1 = int(np.floor(y1 + y_top * height))
    cy2 = int(np.ceil(y1 + y_bottom * height))
    cx1 = max(0, min(image_width - 1, cx1))
    cy1 = max(0, min(image_height - 1, cy1))
    cx2 = max(0, min(image_width, cx2))
    cy2 = max(0, min(image_height, cy2))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    return cx1, cy1, cx2, cy2


def crop_quality_score(
    bbox: Sequence[float],
    *,
    image_width: int,
    image_height: int,
    tracking_confidence: float,
) -> float:
    """Prefer large, centred, high-confidence, lightly truncated boxes."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    if area <= 1.0:
        return 0.0
    area_score = min(1.0, area / float(max(image_width * image_height * 0.02, 1.0)))
    cx = 0.5 * (x1 + x2) / max(image_width, 1)
    cy = 0.5 * (y1 + y2) / max(image_height, 1)
    center_score = 1.0 - min(1.0, abs(cx - 0.5) * 1.5 + abs(cy - 0.45))
    border = 0.0
    border += max(0.0, 4.0 - x1) / 4.0
    border += max(0.0, 4.0 - y1) / 4.0
    border += max(0.0, x2 - (image_width - 4)) / 4.0
    border += max(0.0, y2 - (image_height - 4)) / 4.0
    truncation_penalty = min(1.0, border / 4.0)
    conf = float(np.clip(tracking_confidence, 0.0, 1.0))
    return float(0.45 * area_score + 0.25 * center_score + 0.20 * conf + 0.10 * (1.0 - truncation_penalty))


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        return arr
    return arr / norm


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    aa = l2_normalize(a)
    bb = l2_normalize(b)
    if float(np.linalg.norm(aa)) < 1e-9 or float(np.linalg.norm(bb)) < 1e-9:
        return None
    return float(np.dot(aa, bb))


def robust_mean_embedding(vectors: Iterable[np.ndarray]) -> tuple[np.ndarray | None, float]:
    """Median-pool embeddings after discarding pairwise outliers; return (emb, consistency)."""
    rows = [l2_normalize(v) for v in vectors if v is not None and np.asarray(v).size]
    if not rows:
        return None, 0.0
    matrix = np.stack(rows).astype(np.float64)
    if matrix.shape[0] == 1:
        return matrix[0].astype(np.float64), 1.0
    # Pairwise cosine to prototype median
    median = l2_normalize(np.median(matrix, axis=0))
    sims = matrix @ median
    keep = sims >= max(0.25, float(np.median(sims)) - 0.35)
    kept = matrix[keep] if np.any(keep) else matrix
    pooled = l2_normalize(np.median(kept, axis=0))
    consistency = float(np.mean(kept @ pooled))
    return pooled.astype(np.float64), float(np.clip(consistency, 0.0, 1.0))


@dataclass(frozen=True)
class HardNegativeCalibration:
    pair_count: int
    mean: float
    p50: float
    p75: float
    p90: float
    merge_threshold: float
    strong_threshold: float


def intervals_overlap(
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    tol_ms: float = 40.0,
) -> bool:
    return not (a[1] + tol_ms < b[0] or b[1] + tol_ms < a[0])


def calibrate_hard_negatives(
    embeddings: dict[int, np.ndarray],
    intervals: dict[int, tuple[float, float]],
    teams: dict[int, str | None],
    *,
    base_merge: float = 0.42,
    base_strong: float = 0.55,
    margin: float = 0.08,
    min_pairs: int = 8,
) -> HardNegativeCalibration:
    """Estimate ReID thresholds from simultaneous (must-be-different) pairs.

    Broadcast teammates look alike under Market1501 OSNet, so absolute cosine
    thresholds are unsafe. Simultaneous same-team pairs give a hard-negative
    ceiling; merge thresholds sit above that ceiling.
    """
    sims: list[float] = []
    ids = sorted(embeddings)
    for i, left in enumerate(ids):
        if left not in intervals:
            continue
        for right in ids[i + 1 :]:
            if right not in intervals:
                continue
            if not intervals_overlap(intervals[left], intervals[right]):
                continue
            # Prefer same-team hard negatives (look-alike teammates).
            if teams.get(left) and teams.get(right) and teams[left] != teams[right]:
                continue
            sim = cosine_similarity(embeddings[left], embeddings[right])
            if sim is not None:
                sims.append(sim)

    if len(sims) < min_pairs:
        # Fall back to all simultaneous pairs (including cross-team).
        sims = []
        for i, left in enumerate(ids):
            if left not in intervals:
                continue
            for right in ids[i + 1 :]:
                if right not in intervals:
                    continue
                if not intervals_overlap(intervals[left], intervals[right]):
                    continue
                sim = cosine_similarity(embeddings[left], embeddings[right])
                if sim is not None:
                    sims.append(sim)

    if not sims:
        return HardNegativeCalibration(
            pair_count=0,
            mean=0.0,
            p50=0.0,
            p75=0.0,
            p90=0.0,
            merge_threshold=base_merge,
            strong_threshold=base_strong,
        )

    arr = np.asarray(sims, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    p90 = float(np.percentile(arr, 90))
    # Sit above typical look-alike teammates, but leave headroom for true reappearances.
    merge = float(max(base_merge, min(0.90, p50 + margin)))
    strong = float(max(base_strong, min(0.95, max(merge + 0.04, p75 + margin * 0.35))))
    return HardNegativeCalibration(
        pair_count=int(len(arr)),
        mean=float(np.mean(arr)),
        p50=p50,
        p75=p75,
        p90=p90,
        merge_threshold=merge,
        strong_threshold=strong,
    )


def relative_accept(
    best_sim: float | None,
    second_sim: float | None,
    *,
    merge_threshold: float,
    strong_threshold: float,
    relative_margin: float = 0.04,
) -> tuple[bool, str]:
    """Accept when absolute AND relative (best−second) evidence is strong enough."""
    if best_sim is None:
        return False, "missing_reid"
    if best_sim >= strong_threshold:
        if second_sim is None or (best_sim - second_sim) >= relative_margin * 0.5:
            return True, "strong_reid_relative"
        # Ambiguous gallery: still allow very strong absolute matches
        if best_sim >= min(0.99, strong_threshold + 0.05):
            return True, "strong_reid_absolute"
        return False, "ambiguous_gallery"
    if best_sim >= merge_threshold:
        if second_sim is not None and (best_sim - second_sim) >= relative_margin:
            return True, "reid_relative_margin"
        return False, "below_relative_margin"
    return False, "below_merge_threshold"
