"""Ball ground-truth sample selection and I/O."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd

from football_analytics.evaluation.boolean_utils import (
    BOOL_COLUMNS_BALL_GT,
    apply_boolean_dtype,
    count_true,
    is_true,
    normalize_boolean,
)

BALL_GT_COLUMNS = [
    "video_name",
    "frame_index",
    "timestamp",
    "frame_width",
    "frame_height",
    "ball_visible",
    "ball_x1",
    "ball_y1",
    "ball_x2",
    "ball_y2",
    "occluded",
    "difficult",
    "annotator_notes",
    "reviewed",
]

STRATA = (
    "clear_visible",
    "small_distant",
    "near_player",
    "motion_blur",
    "partial_occluded",
    "fully_missing",
    "false_positive_risk",
    "camera_motion",
    "camera_cut",
    "crowded_box",
    "empty_pitch",
)


def load_ball_gt_csv(path: Path) -> pd.DataFrame:
    return apply_boolean_dtype(pd.read_csv(path), BOOL_COLUMNS_BALL_GT)


def empty_ball_gt_row(
    *,
    video_name: str,
    frame_index: int,
    timestamp: float,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    return {
        "video_name": video_name,
        "frame_index": int(frame_index),
        "timestamp": float(timestamp),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "ball_visible": pd.NA,
        "ball_x1": None,
        "ball_y1": None,
        "ball_x2": None,
        "ball_y2": None,
        "occluded": pd.NA,
        "difficult": pd.NA,
        "annotator_notes": "",
        "reviewed": False,
    }


def ball_gt_complete(gt: pd.DataFrame, *, min_frames: int = 150) -> tuple[bool, str]:
    """Legacy full-pool completeness (all rows). Prefer review_sample_complete for 50-frame eval."""
    if gt is None or gt.empty:
        return False, "empty_gt"
    gt = apply_boolean_dtype(gt)
    if len(gt) < min_frames:
        return False, f"too_few_frames:{len(gt)}<{min_frames}"
    if "reviewed" not in gt.columns:
        return False, "missing_reviewed_column"
    reviewed_n = count_true(gt["reviewed"])
    if reviewed_n < len(gt):
        return False, f"unreviewed:{len(gt) - reviewed_n}"
    for _, row in gt.iterrows():
        if not is_true(row.get("reviewed")):
            continue
        if normalize_boolean(row.get("ball_visible")) is pd.NA:
            return False, "ball_visible_incomplete"
    return True, "complete"


def review_sample_complete(
    sample: pd.DataFrame,
    ball_gt: pd.DataFrame,
    *,
    sample_size: int = 50,
) -> tuple[bool, str]:
    if sample is None or sample.empty:
        return False, "empty_sample"
    if len(sample) != sample_size:
        return False, f"sample_size_mismatch:{len(sample)}!={sample_size}"
    if int(sample["frame_idx"].nunique()) != sample_size:
        return False, "duplicate_frames"
    from football_analytics.evaluation.ball_review_sample import sample_reviewed_count

    n = sample_reviewed_count(sample, ball_gt)
    if n < sample_size:
        return False, f"unreviewed:{sample_size - n}"
    # ball_visible must be set for reviewed sample rows
    gt = apply_boolean_dtype(ball_gt)
    frame_col = "frame_index" if "frame_index" in gt.columns else "frame_idx"
    ids = set(int(x) for x in sample["frame_idx"].tolist())
    sub = gt[gt[frame_col].astype(int).isin(ids)]
    for _, row in sub.iterrows():
        if not is_true(row.get("reviewed")):
            continue
        if normalize_boolean(row.get("ball_visible")) is pd.NA:
            return False, "ball_visible_incomplete"
    return True, "complete"


def completion_rate(gt: pd.DataFrame) -> float:
    if gt is None or gt.empty or "reviewed" not in gt.columns:
        return 0.0
    gt = apply_boolean_dtype(gt)
    return float(count_true(gt["reviewed"]) / max(len(gt), 1))


def select_stratified_frames(
    *,
    n_frames: int,
    total_frames: int,
    seed: int,
    detections: pd.DataFrame | None = None,
    provenance: pd.DataFrame | None = None,
    camera_motion: pd.DataFrame | None = None,
    shot_segments: pd.DataFrame | None = None,
    tracks: pd.DataFrame | None = None,
    per_stratum: int = 15,
) -> list[dict[str, Any]]:
    """Select ≥ n_frames with balanced hard cases (reproducible seed)."""
    rng = random.Random(seed)
    buckets: dict[str, list[int]] = {s: [] for s in STRATA}

    det_by = {}
    if detections is not None and not detections.empty and "frame_id" in detections.columns:
        for fid, g in detections.groupby("frame_id"):
            det_by[int(fid)] = g

    prov_by = {}
    if provenance is not None and not provenance.empty:
        for row in provenance.itertuples(index=False):
            prov_by[int(row.frame_id)] = str(getattr(row, "provenance", "missing"))

    motion_by = {}
    if camera_motion is not None and not camera_motion.empty:
        for row in camera_motion.itertuples(index=False):
            motion_by[int(row.frame_id)] = float(getattr(row, "dx_pixel", 0.0) or 0.0) ** 2 + float(
                getattr(row, "dy_pixel", 0.0) or 0.0
            ) ** 2

    cut_frames = set()
    if shot_segments is not None and not shot_segments.empty and "scene_cut" in shot_segments.columns:
        cut_frames = set(
            int(x) for x in shot_segments.loc[shot_segments["scene_cut"].astype(bool), "frame_id"]
        )

    person_count: dict[int, int] = {}
    if tracks is not None and not tracks.empty:
        person = tracks[tracks["object_type"].eq("person")] if "object_type" in tracks.columns else tracks
        person_count = person.groupby("frame_id").size().astype(int).to_dict()

    for fid in range(total_frames):
        dets = det_by.get(fid)
        prov = prov_by.get(fid, "missing")
        n_people = int(person_count.get(fid, 0))
        motion = float(motion_by.get(fid, 0.0))

        if fid in cut_frames:
            buckets["camera_cut"].append(fid)
        if motion > 25.0:
            buckets["camera_motion"].append(fid)
        if n_people >= 12:
            buckets["crowded_box"].append(fid)
        if n_people <= 2:
            buckets["empty_pitch"].append(fid)

        if dets is None or len(dets) == 0 or prov == "missing":
            buckets["fully_missing"].append(fid)
            continue

        # Use highest-confidence det
        conf_col = "detection_confidence" if "detection_confidence" in dets.columns else None
        row = dets.sort_values(conf_col, ascending=False).iloc[0] if conf_col else dets.iloc[0]
        w = float(row.get("bbox_w", 0) or 0)
        h = float(row.get("bbox_h", 0) or 0)
        area = w * h
        conf = float(row.get("detection_confidence", 0) or 0)
        cx = float(row.get("ball_x_pixel", 0) or 0)
        cy = float(row.get("ball_y_pixel", 0) or 0)

        if area > 0 and area < 80:
            buckets["small_distant"].append(fid)
        if conf >= 0.55 and area >= 120:
            buckets["clear_visible"].append(fid)
        if conf < 0.25:
            buckets["false_positive_risk"].append(fid)
        if conf < 0.4 and area < 200:
            buckets["motion_blur"].append(fid)
        if prov in {"interpolated", "predicted"}:
            buckets["partial_occluded"].append(fid)

        # Near player: any person bbox center near ball
        if tracks is not None and not tracks.empty:
            frm = tracks[tracks["frame_id"] == fid]
            if not frm.empty and {"x1", "y1", "x2", "y2"}.issubset(frm.columns):
                px = (frm["x1"] + frm["x2"]) / 2.0
                py = (frm["y1"] + frm["y2"]) / 2.0
                dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                if float(dist.min()) < 80:
                    buckets["near_player"].append(fid)

    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for stratum in STRATA:
        pool = [f for f in buckets[stratum] if f not in used]
        rng.shuffle(pool)
        take = pool[:per_stratum]
        for fid in take:
            used.add(fid)
            selected.append({"frame_index": fid, "stratum": stratum})

    # Top up with seeded uniform sample if needed
    if len(selected) < n_frames:
        remaining = [f for f in range(total_frames) if f not in used]
        rng.shuffle(remaining)
        for fid in remaining[: n_frames - len(selected)]:
            selected.append({"frame_index": fid, "stratum": "uniform_fill"})
            used.add(fid)

    selected = sorted(selected, key=lambda x: x["frame_index"])
    # Deduplicate frames keeping first stratum tag
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in selected:
        if item["frame_index"] in seen:
            continue
        seen.add(item["frame_index"])
        out.append(item)
    return out[: max(n_frames, len(out))]


def write_ball_gt_sample(
    *,
    video_path: Path,
    out_dir: Path,
    frame_specs: Sequence[dict[str, Any]],
    detections: pd.DataFrame | None = None,
    video_name: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    preds_dir = out_dir / "predictions"
    frames_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    name = video_name or Path(video_path).stem

    det_by: dict[int, pd.DataFrame] = {}
    if detections is not None and not detections.empty:
        for fid, g in detections.groupby("frame_id"):
            det_by[int(fid)] = g

    rows: list[dict[str, Any]] = []
    for spec in frame_specs:
        fid = int(spec["frame_index"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        ts = fid / max(fps, 1e-6)
        frame_path = frames_dir / f"frame_{fid:06d}.png"
        cv2.imwrite(str(frame_path), frame)

        overlay = frame.copy()
        dets = det_by.get(fid)
        if dets is not None and len(dets):
            for _, d in dets.iterrows():
                cx = float(d.get("ball_x_pixel", np.nan))
                cy = float(d.get("ball_y_pixel", np.nan))
                w = float(d.get("bbox_w", 20) or 20)
                h = float(d.get("bbox_h", 20) or 20)
                if not np.isfinite(cx) or not np.isfinite(cy):
                    continue
                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                x2, y2 = int(cx + w / 2), int(cy + h / 2)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
                conf = d.get("detection_confidence")
                if conf is not None and np.isfinite(float(conf)):
                    cv2.putText(
                        overlay,
                        f"{float(conf):.2f}",
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )
        pred_path = preds_dir / f"frame_{fid:06d}.png"
        cv2.imwrite(str(pred_path), overlay)

        row = empty_ball_gt_row(
            video_name=name,
            frame_index=fid,
            timestamp=ts,
            frame_width=width,
            frame_height=height,
        )
        row["stratum"] = spec.get("stratum")
        rows.append(row)

    cap.release()
    gt = pd.DataFrame(rows)
    # Keep schema columns first; stratum is helper metadata in CSV too
    cols = BALL_GT_COLUMNS + (["stratum"] if "stratum" in gt.columns else [])
    gt = gt[[c for c in cols if c in gt.columns]]
    gt_path = out_dir / "ball_gt.csv"
    gt.to_csv(gt_path, index=False)

    manifest = {
        "video_path": str(video_path),
        "video_name": name,
        "seed": int(seed),
        "frame_count": int(len(gt)),
        "frame_width": width,
        "frame_height": height,
        "fps": fps,
        "strata": sorted({str(s) for s in gt.get("stratum", pd.Series(dtype=str)).dropna().unique()}),
        "ball_gt_csv": str(gt_path),
        "frames_dir": str(frames_dir),
        "predictions_dir": str(preds_dir),
        "status": "GT_INCOMPLETE",
        "note": "Annotations empty; candidate overlays are predictions only, not ground truth.",
    }
    (out_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
