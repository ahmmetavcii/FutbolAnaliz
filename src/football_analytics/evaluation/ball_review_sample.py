"""Build a balanced 50-frame manual review sample from the 165-frame ball GT pool."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from football_analytics.evaluation.boolean_utils import apply_boolean_dtype, count_true, is_true


REVIEW_SAMPLE_COLUMNS = [
    "sample_order",
    "frame_idx",
    "timestamp",
    "sampling_reason",
    "provenance",
    "confidence",
    "trajectory_jump",
    "multiple_candidate",
    "reviewed",
]


def create_balanced_review_sample(
    *,
    ball_gt: pd.DataFrame,
    provenance: pd.DataFrame | None = None,
    detections: pd.DataFrame | None = None,
    wrong_object_report: dict[str, Any] | None = None,
    sample_size: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Select exactly ``sample_size`` unique frames from ball_gt with balanced strata."""
    if ball_gt is None or ball_gt.empty:
        raise ValueError("ball_gt is empty")
    gt = apply_boolean_dtype(ball_gt)
    frame_col = "frame_index" if "frame_index" in gt.columns else "frame_idx"
    pool = sorted({int(x) for x in gt[frame_col].tolist()})
    if len(pool) < sample_size:
        raise ValueError(f"pool has {len(pool)} frames, need {sample_size}")

    rng = random.Random(seed)

    prov_by: dict[int, str] = {}
    if provenance is not None and not provenance.empty:
        for row in provenance.itertuples(index=False):
            prov_by[int(row.frame_id)] = str(getattr(row, "provenance", "missing"))

    conf_by: dict[int, float] = {}
    multi_by: dict[int, bool] = {}
    if detections is not None and not detections.empty:
        for fid, g in detections.groupby("frame_id"):
            fid_i = int(fid)
            multi_by[fid_i] = len(g) > 1
            if "detection_confidence" in g.columns:
                conf_by[fid_i] = float(g["detection_confidence"].max())
            else:
                conf_by[fid_i] = 0.0

    # Optional jump frames from wrong-object report (frame list if present)
    jump_frames: set[int] = set()
    if wrong_object_report:
        for key in ("jump_frame_ids", "trajectory_jump_frames", "switch_frame_ids"):
            vals = wrong_object_report.get(key) or []
            for v in vals:
                jump_frames.add(int(v))

    # Heuristic jumps: large consecutive center movement among pool detections
    if detections is not None and not detections.empty and "ball_x_pixel" in detections.columns:
        best = (
            detections.sort_values("detection_confidence", ascending=False)
            .groupby("frame_id", as_index=False)
            .first()
            .sort_values("frame_id")
        )
        prev = None
        for _, r in best.iterrows():
            fid = int(r["frame_id"])
            if prev is not None:
                dist = (
                    (float(r["ball_x_pixel"]) - float(prev["ball_x_pixel"])) ** 2
                    + (float(r["ball_y_pixel"]) - float(prev["ball_y_pixel"])) ** 2
                ) ** 0.5
                if dist > 250:
                    jump_frames.add(fid)
            prev = r

    ts_by = {
        int(r[frame_col]): float(r["timestamp"]) if "timestamp" in gt.columns and pd.notna(r["timestamp"]) else 0.0
        for _, r in gt.iterrows()
    }
    min_f, max_f = min(pool), max(pool)
    third = (max_f - min_f) / 3.0 if max_f > min_f else 1.0

    buckets: dict[str, list[int]] = {
        "provenance_detected": [],
        "provenance_tracked": [],
        "provenance_predicted": [],
        "provenance_interpolated": [],
        "provenance_missing": [],
        "conf_high": [],
        "conf_mid": [],
        "conf_low": [],
        "multiple_candidate": [],
        "trajectory_jump": [],
        "possibly_invisible": [],
        "temporal_start": [],
        "temporal_mid": [],
        "temporal_end": [],
    }
    for fid in pool:
        prov = prov_by.get(fid, "missing")
        if prov == "detected":
            buckets["provenance_detected"].append(fid)
        elif prov == "tracked":
            buckets["provenance_tracked"].append(fid)
        elif prov == "predicted":
            buckets["provenance_predicted"].append(fid)
        elif prov == "interpolated":
            buckets["provenance_interpolated"].append(fid)
        else:
            buckets["provenance_missing"].append(fid)

        conf = conf_by.get(fid)
        if conf is None:
            buckets["possibly_invisible"].append(fid)
        elif conf >= 0.55:
            buckets["conf_high"].append(fid)
        elif conf >= 0.25:
            buckets["conf_mid"].append(fid)
        else:
            buckets["conf_low"].append(fid)

        if multi_by.get(fid):
            buckets["multiple_candidate"].append(fid)
        if fid in jump_frames:
            buckets["trajectory_jump"].append(fid)
        if prov in {"missing", "predicted", "interpolated"} or (conf is not None and conf < 0.2):
            buckets["possibly_invisible"].append(fid)

        if fid <= min_f + third:
            buckets["temporal_start"].append(fid)
        elif fid <= min_f + 2 * third:
            buckets["temporal_mid"].append(fid)
        else:
            buckets["temporal_end"].append(fid)

    # Target quotas (sum >= sample_size; we stop at sample_size unique)
    quotas = {
        "provenance_detected": 8,
        "provenance_tracked": 8,
        "provenance_predicted": 3,
        "provenance_interpolated": 3,
        "provenance_missing": 4,
        "conf_high": 5,
        "conf_mid": 4,
        "conf_low": 4,
        "multiple_candidate": 4,
        "trajectory_jump": 4,
        "possibly_invisible": 4,
        "temporal_start": 3,
        "temporal_mid": 3,
        "temporal_end": 3,
    }

    selected: list[tuple[int, str]] = []
    used: set[int] = set()

    def take(reason: str, n: int) -> None:
        pool_r = [f for f in buckets.get(reason, []) if f not in used]
        rng.shuffle(pool_r)
        for fid in pool_r[:n]:
            used.add(fid)
            selected.append((fid, reason))

    for reason, n in quotas.items():
        if len(selected) >= sample_size:
            break
        remain = sample_size - len(selected)
        take(reason, min(n, remain))

    if len(selected) < sample_size:
        remain_pool = [f for f in pool if f not in used]
        rng.shuffle(remain_pool)
        for fid in remain_pool[: sample_size - len(selected)]:
            selected.append((fid, "uniform_fill"))
            used.add(fid)

    selected = selected[:sample_size]
    # Deduplicate by frame (keep first reason), then sort by time
    by_frame: dict[int, str] = {}
    for fid, reason in selected:
        if fid not in by_frame:
            by_frame[fid] = reason
    if len(by_frame) < sample_size:
        remain_pool = [f for f in pool if f not in by_frame]
        rng.shuffle(remain_pool)
        for fid in remain_pool:
            if len(by_frame) >= sample_size:
                break
            by_frame[fid] = "uniform_fill"
    ordered = sorted(by_frame.items(), key=lambda x: x[0])[:sample_size]
    rows = []
    for order, (fid, reason) in enumerate(ordered, start=1):
        rows.append(
            {
                "sample_order": order,
                "frame_idx": int(fid),
                "timestamp": float(ts_by.get(fid, 0.0)),
                "sampling_reason": reason,
                "provenance": prov_by.get(fid, "unknown"),
                "confidence": conf_by.get(fid),
                "trajectory_jump": bool(fid in jump_frames),
                "multiple_candidate": bool(multi_by.get(fid, False)),
                "reviewed": False,
            }
        )
    out = pd.DataFrame(rows, columns=REVIEW_SAMPLE_COLUMNS)
    out = apply_boolean_dtype(out, columns=("reviewed", "trajectory_jump", "multiple_candidate"))
    if len(out) != sample_size or int(out["frame_idx"].nunique()) != sample_size:
        raise ValueError(
            f"failed to build unique sample: rows={len(out)} unique={out['frame_idx'].nunique()}"
        )
    if count_true(out["reviewed"]) != 0:
        raise ValueError("automatic reviewed must be 0")
    return out


def sample_reviewed_count(sample: pd.DataFrame, ball_gt: pd.DataFrame) -> int:
    """Count how many sample frames are reviewed=True in ball_gt."""
    gt = apply_boolean_dtype(ball_gt)
    frame_col = "frame_index" if "frame_index" in gt.columns else "frame_idx"
    sample_ids = set(int(x) for x in sample["frame_idx"].tolist())
    sub = gt[gt[frame_col].astype(int).isin(sample_ids)]
    return count_true(sub["reviewed"]) if "reviewed" in sub.columns else 0


def write_review_sample(
    sample: pd.DataFrame,
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write booleans as True/False strings for CSV clarity
    out = sample.copy()
    for col in ("reviewed", "trajectory_jump", "multiple_candidate"):
        if col in out.columns:
            out[col] = out[col].map(lambda v: True if is_true(v) else False)
    out.to_csv(path, index=False)
    return path
