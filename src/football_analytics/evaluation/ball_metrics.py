"""Ball detection/tracking evaluation against ground truth.

Returns status=GT_INCOMPLETE and omits precision/recall when GT is unfinished.
Never labels candidate coverage as recall.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_analytics.evaluation.ball_gt import (
    completion_rate,
    load_ball_gt_csv,
    review_sample_complete,
)
from football_analytics.evaluation.ball_review_sample import sample_reviewed_count
from football_analytics.evaluation.boolean_utils import apply_boolean_dtype, is_true, normalize_boolean


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _pred_box(row: pd.Series, default_size: float = 24.0) -> tuple[float, float, float, float] | None:
    if {"ball_x1", "ball_y1", "ball_x2", "ball_y2"}.issubset(row.index) and pd.notna(
        row.get("ball_x1")
    ):
        return (
            float(row["ball_x1"]),
            float(row["ball_y1"]),
            float(row["ball_x2"]),
            float(row["ball_y2"]),
        )
    cx = row.get("ball_x_pixel")
    cy = row.get("ball_y_pixel")
    if cx is None or cy is None or not np.isfinite(float(cx)) or not np.isfinite(float(cy)):
        return None
    w = float(row.get("bbox_w", default_size) or default_size)
    h = float(row.get("bbox_h", default_size) or default_size)
    return (float(cx) - w / 2, float(cy) - h / 2, float(cx) + w / 2, float(cy) + h / 2)


def _gt_box(row: pd.Series) -> tuple[float, float, float, float] | None:
    vals = [row.get(c) for c in ("ball_x1", "ball_y1", "ball_x2", "ball_y2")]
    if any(v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v) for v in vals):
        return None
    return tuple(float(v) for v in vals)  # type: ignore[return-value]


def provenance_mutually_exclusive(prov: pd.DataFrame) -> bool:
    if prov is None or prov.empty or "frame_id" not in prov.columns:
        return True
    counts = prov.groupby("frame_id").size()
    return bool((counts <= 1).all())


def evaluate_ball_tracking(
    *,
    gt_csv: Path,
    run_dir: Path,
    out_dir: Path | None = None,
    review_sample_csv: Path | None = None,
    iou_threshold: float = 0.3,
    center_match_px: float = 40.0,
    sample_size: int = 50,
    min_frames: int = 150,  # legacy unused when review_sample provided
) -> dict[str, Any]:
    gt = load_ball_gt_csv(gt_csv)
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ball_errors").mkdir(exist_ok=True)
    (out / "ball_false_positives").mkdir(exist_ok=True)
    (out / "ball_false_negatives").mkdir(exist_ok=True)

    # Resolve review sample (required path for 50-frame workflow)
    sample_path = Path(review_sample_csv) if review_sample_csv else gt_csv.parent / "review_sample.csv"
    sample = None
    if sample_path.is_file():
        sample = apply_boolean_dtype(
            pd.read_csv(sample_path),
            columns=("reviewed", "trajectory_jump", "multiple_candidate"),
        )
        complete, reason = review_sample_complete(sample, gt, sample_size=sample_size)
        manually_reviewed = sample_reviewed_count(sample, gt)
        sample_ids = set(int(x) for x in sample["frame_idx"].tolist())
    else:
        # Fallback: legacy full-pool (still no metrics if incomplete)
        from football_analytics.evaluation.ball_gt import ball_gt_complete

        complete, reason = ball_gt_complete(gt, min_frames=min_frames)
        manually_reviewed = int(gt["reviewed"].map(is_true).sum()) if "reviewed" in gt.columns else 0
        sample_ids = set(int(x) for x in gt["frame_index"].tolist())
        sample_size = len(sample_ids)

    # Candidate / trajectory coverage (NOT recall)
    det_report = {}
    cov_report = {}
    if (run_dir / "ball_detection_report.json").is_file():
        det_report = json.loads((run_dir / "ball_detection_report.json").read_text(encoding="utf-8"))
    if (run_dir / "ball_coverage_report.json").is_file():
        cov_report = json.loads((run_dir / "ball_coverage_report.json").read_text(encoding="utf-8"))
    wo = {}
    if (run_dir / "wrong_object_ball_report.json").is_file():
        wo = json.loads((run_dir / "wrong_object_ball_report.json").read_text(encoding="utf-8"))

    provenance = (
        pd.read_parquet(run_dir / "ball_provenance.parquet")
        if (run_dir / "ball_provenance.parquet").is_file()
        else pd.DataFrame()
    )
    prov_counts = (
        provenance["provenance"].value_counts().to_dict()
        if not provenance.empty and "provenance" in provenance.columns
        else {}
    )
    frames_total = int(det_report.get("frames") or cov_report.get("frames") or len(provenance) or 0)
    candidate_coverage = float(det_report.get("raw_detection_coverage") or 0.0)
    trajectory_coverage = float(
        cov_report.get("tracked_coverage")
        or det_report.get("tracked_coverage")
        or 0.0
    )

    remaining = max(sample_size - manually_reviewed, 0)
    report: dict[str, Any] = {
        "status": "GT_INCOMPLETE" if not complete else "OK",
        "gt_incomplete_reason": None if complete else reason,
        "sample_size": int(sample_size),
        "manually_reviewed": int(manually_reviewed),
        "remaining": int(remaining),
        "completion_percentage": round(100.0 * manually_reviewed / max(sample_size, 1), 2),
        "sampling_method": "balanced_stratified_manual_review",
        "automatic_reviewed_count": 0,
        "gt_frame_count": int(len(gt)),
        "gt_completion_rate": completion_rate(gt),
        "candidate_coverage": candidate_coverage,
        "trajectory_coverage": trajectory_coverage,
        "provenance_counts": {k: int(v) for k, v in prov_counts.items()},
        "provenance_mutually_exclusive": provenance_mutually_exclusive(provenance),
        "note": (
            "candidate_coverage is not recall; trajectory_coverage is not accuracy. "
            "Precision/recall require completed 50-frame manual review sample."
        ),
    }

    if not complete:
        (out / "ball_evaluation_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report

    # --- Metrics only on reviewed sample frames ---
    preds = (
        pd.read_parquet(run_dir / "football_ball_detections.parquet")
        if (run_dir / "football_ball_detections.parquet").is_file()
        else pd.DataFrame()
    )
    if preds.empty and (run_dir / "ball_state.parquet").is_file():
        preds = pd.read_parquet(run_dir / "ball_state.parquet")

    pred_by = {int(fid): g for fid, g in preds.groupby("frame_id")} if not preds.empty else {}
    prov_map = (
        {int(r.frame_id): str(r.provenance) for r in provenance.itertuples(index=False)}
        if not provenance.empty
        else {}
    )

    eval_gt = gt[gt["frame_index"].astype(int).isin(sample_ids)].copy()
    eval_gt = eval_gt[eval_gt["reviewed"].map(is_true)]

    frame_rows: list[dict[str, Any]] = []
    tp = fp = fn = 0
    ious: list[float] = []
    center_errs: list[float] = []
    confs: list[float] = []

    for _, row in eval_gt.iterrows():
        fid = int(row["frame_index"])
        vis_n = normalize_boolean(row.get("ball_visible"))
        visible = vis_n is True
        gt_box = _gt_box(row) if visible else None

        pred_rows = pred_by.get(fid)
        best = None
        best_iou = 0.0
        best_center = None
        if pred_rows is not None and len(pred_rows):
            for _, pr in pred_rows.iterrows():
                pb = _pred_box(pr)
                if pb is None:
                    continue
                if gt_box is not None:
                    iou = _iou(gt_box, pb)
                    gcx = (gt_box[0] + gt_box[2]) / 2
                    gcy = (gt_box[1] + gt_box[3]) / 2
                    pcx = (pb[0] + pb[2]) / 2
                    pcy = (pb[1] + pb[3]) / 2
                    cerr = float(np.hypot(gcx - pcx, gcy - pcy))
                else:
                    iou, cerr = 0.0, float("inf")
                if best is None or iou > best_iou or (
                    iou == best_iou and cerr < (best_center or 1e9)
                ):
                    best = pr
                    best_iou = iou
                    best_center = cerr

        matched = False
        if visible and gt_box is not None and best is not None:
            matched = best_iou >= iou_threshold or (
                best_center is not None and best_center <= center_match_px
            )

        if visible and matched:
            tp += 1
            ious.append(best_iou)
            center_errs.append(float(best_center or 0))
            if best is not None and pd.notna(best.get("detection_confidence")):
                confs.append(float(best["detection_confidence"]))
        elif visible and not matched:
            fn += 1
        elif (not visible) and best is not None:
            fp += 1

        frame_rows.append(
            {
                "frame_index": fid,
                "gt_visible": visible,
                "matched": matched,
                "iou": best_iou if matched else None,
                "center_error_px": best_center if matched else None,
                "provenance": prov_map.get(fid),
                "prediction_present": best is not None,
            }
        )

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    report.update(
        {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
            "IoU": round(float(np.mean(ious)), 4) if ious else None,
            "mean_center_error": round(float(np.mean(center_errs)), 4) if center_errs else None,
            "median_center_error": round(float(np.median(center_errs)), 4) if center_errs else None,
            "median_center_error_px": round(float(np.median(center_errs)), 4) if center_errs else None,
            "wrong_object_switch": wo.get("wrong_object_switches")
            or cov_report.get("wrong_object_switches"),
            "trajectory_jump": wo.get("trajectory_jumps") or cov_report.get("trajectory_jumps"),
            "average_confidence": round(float(np.mean(confs)), 4) if confs else None,
            "detected_coverage": round(prov_counts.get("detected", 0) / max(frames_total, 1), 4),
            "tracked_coverage": round(prov_counts.get("tracked", 0) / max(frames_total, 1), 4),
            "predicted_coverage": round(prov_counts.get("predicted", 0) / max(frames_total, 1), 4),
            "interpolated_coverage": round(
                prov_counts.get("interpolated", 0) / max(frames_total, 1), 4
            ),
            "missing_coverage": round(prov_counts.get("missing", 0) / max(frames_total, 1), 4),
        }
    )

    pd.DataFrame(frame_rows).to_parquet(out / "ball_frame_results.parquet", index=False)
    (out / "ball_evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
