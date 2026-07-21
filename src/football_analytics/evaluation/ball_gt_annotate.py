"""Disk-verified ball GT save helpers (used by annotate_ball_gt.py and tests)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_analytics.evaluation.ball_gt import load_ball_gt_csv
from football_analytics.evaluation.ball_review_sample import sample_reviewed_count
from football_analytics.evaluation.boolean_utils import (
    BOOL_COLUMNS_BALL_GT,
    apply_boolean_dtype,
    count_true,
    is_true,
    normalize_boolean,
)


def atomic_write_ball_gt(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out = df.copy()
    for col in BOOL_COLUMNS_BALL_GT:
        if col in out.columns:
            out[col] = out[col].map(
                lambda v: True
                if is_true(v)
                else (False if normalize_boolean(v) is False else "")
            )
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)


def gt_index_for_frame(gt: pd.DataFrame, frame_idx: int) -> int:
    hits = gt.index[gt["frame_index"].astype(int) == int(frame_idx)].tolist()
    if not hits:
        raise KeyError(f"frame_index={frame_idx} missing")
    return int(hits[0])


def recount(sample: pd.DataFrame, gt: pd.DataFrame) -> dict[str, int]:
    return {
        "sample_reviewed": sample_reviewed_count(sample, gt),
        "sample_size": int(len(sample)),
        "full_reviewed": count_true(gt["reviewed"]) if "reviewed" in gt.columns else 0,
        "full_size": int(len(gt)),
    }


def save_current_annotation(
    *,
    gt: pd.DataFrame,
    sample: pd.DataFrame,
    gt_path: Path,
    frame_idx: int,
    ball_visible: bool | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    clear_bbox: bool = False,
    occluded: bool | None = None,
    difficult: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Update one frame, atomic-save, reload, verify reviewed=True, recount.

    Returns (reloaded_gt, result_dict). On failure raises RuntimeError and does
    not claim success (caller must not advance).
    """
    gt = apply_boolean_dtype(gt)
    gidx = gt_index_for_frame(gt, frame_idx)

    if clear_bbox or ball_visible is False:
        gt.at[gidx, "ball_visible"] = False
        for c in ("ball_x1", "ball_y1", "ball_x2", "ball_y2"):
            gt.at[gidx, c] = np.nan
    elif bbox is not None:
        x1, y1, x2, y2 = bbox
        gt.at[gidx, "ball_x1"] = float(x1)
        gt.at[gidx, "ball_y1"] = float(y1)
        gt.at[gidx, "ball_x2"] = float(x2)
        gt.at[gidx, "ball_y2"] = float(y2)
        gt.at[gidx, "ball_visible"] = True if ball_visible is None else bool(ball_visible)
    elif ball_visible is True and pd.notna(gt.at[gidx, "ball_x1"]):
        gt.at[gidx, "ball_visible"] = True
    elif ball_visible is True:
        raise RuntimeError("ball_visible=True but no bbox")

    if occluded is not None:
        gt.at[gidx, "occluded"] = bool(occluded)
    if difficult is not None:
        gt.at[gidx, "difficult"] = bool(difficult)

    if normalize_boolean(gt.at[gidx, "ball_visible"]) is pd.NA:
        raise RuntimeError("ball_visible_unset")

    gt.at[gidx, "reviewed"] = True
    gt = apply_boolean_dtype(gt)

    atomic_write_ball_gt(gt, gt_path)
    reloaded = load_ball_gt_csv(gt_path)
    ridx = gt_index_for_frame(reloaded, frame_idx)
    if not is_true(reloaded.at[ridx, "reviewed"]):
        raise RuntimeError("disk reviewed is not True after save")

    counts = recount(sample, reloaded)
    return reloaded, {
        "ok": True,
        "frame_idx": int(frame_idx),
        "csv_verified": True,
        **counts,
    }


def first_unreviewed_sample_index(sample: pd.DataFrame, gt: pd.DataFrame) -> int:
    gt = apply_boolean_dtype(gt)
    sample = sample.sort_values("sample_order").reset_index(drop=True)
    for i, row in sample.iterrows():
        fid = int(row["frame_idx"])
        gidx = gt_index_for_frame(gt, fid)
        if not is_true(gt.at[gidx, "reviewed"]):
            return int(i)
    return 0
