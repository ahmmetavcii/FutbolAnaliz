"""Calibration frame audit + limited homography propagation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PropagationConfig:
    max_propagate_frames: int = 25
    max_homography_jump: float = 35.0
    max_reprojection_error: float = 10.0
    min_confidence: float = 0.25


def _parse_H(raw: Any) -> np.ndarray | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    if isinstance(raw, str):
        try:
            arr = np.asarray(json.loads(raw), dtype=np.float64)
        except Exception:
            return None
    else:
        arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (3, 3):
        return None
    return arr


def _H_jump(a: np.ndarray, b: np.ndarray, width: float = 1920.0, height: float = 1080.0) -> float:
    """Mean pixel displacement of pitch corners under H_a vs H_b."""
    corners = np.asarray(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64
    ).reshape(-1, 1, 2)
    try:
        pa = cv2_perspective(corners, a)
        pb = cv2_perspective(corners, b)
    except Exception:
        return float("inf")
    return float(np.mean(np.linalg.norm(pa - pb, axis=2)))


def cv2_perspective(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.perspectiveTransform(pts, H)


def categorize_rejection(reason: str | None, *, pan: bool, zoom: bool, cut: bool) -> str:
    text = str(reason or "").lower()
    if cut:
        return "camera_cut"
    if "hold window" in text:
        return "insufficient_correspondence"
    if "reprojection" in text:
        return "high_reprojection_error"
    if "homography" in text or "invalid" in text:
        return "invalid_homography"
    if pan:
        return "camera_pan"
    if zoom:
        return "camera_zoom"
    if "replay" in text or "close" in text:
        return "close_up" if "close" in text else "replay"
    if "provider" in text or "fail" in text:
        return "provider_failure"
    if "line" in text or "pitch" in text:
        return "no_pitch_lines"
    if reason:
        return "provider_failure"
    return "ok"


def build_calibration_frame_audit(
    calibration: pd.DataFrame,
    *,
    camera_motion: pd.DataFrame | None = None,
    shot_segments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cal = calibration.sort_values("frame_id").reset_index(drop=True).copy()
    motion_by = {}
    if camera_motion is not None and not camera_motion.empty:
        for row in camera_motion.itertuples(index=False):
            motion_by[int(row.frame_id)] = row
    cut_by = {}
    if shot_segments is not None and not shot_segments.empty:
        for row in shot_segments.itertuples(index=False):
            cut_by[int(row.frame_id)] = bool(getattr(row, "scene_cut", False))

    rows = []
    prev_H = None
    for row in cal.itertuples(index=False):
        fid = int(row.frame_id)
        m = motion_by.get(fid)
        dx = float(getattr(m, "dx_pixel", 0.0) or 0.0) if m is not None else 0.0
        dy = float(getattr(m, "dy_pixel", 0.0) or 0.0) if m is not None else 0.0
        scale = float(getattr(m, "scale", 1.0) or 1.0) if m is not None else 1.0
        pan = abs(dx) + abs(dy) > 8.0
        zoom = abs(scale - 1.0) > 0.02
        cut = bool(cut_by.get(fid, False))
        H = _parse_H(getattr(row, "homography_json", None))
        jump = None
        if H is not None and prev_H is not None:
            jump = _H_jump(prev_H, H)
        if H is not None:
            prev_H = H
        provider = str(getattr(row, "provider", "") or "")
        valid = bool(getattr(row, "valid", False))
        reason = getattr(row, "invalid_reason", None)
        category = categorize_rejection(reason, pan=pan, zoom=zoom, cut=cut)
        rows.append(
            {
                "frame_index": fid,
                "timestamp": float(getattr(row, "timestamp_ms", 0.0) or 0.0) / 1000.0,
                "shot_id": int(getattr(row, "segment_id", 0) or 0),
                "calibration_valid": valid,
                "provider": provider,
                "confidence": float(getattr(row, "confidence", 0.0) or 0.0),
                "reprojection_error": getattr(row, "reprojection_error", None),
                "visible_pitch_lines": float(getattr(row, "visible_pitch_coverage", 0.0) or 0.0),
                "pan_detected": pan,
                "zoom_detected": zoom,
                "camera_cut": cut,
                "homography_jump": jump,
                "fallback_used": provider in {"propagated", "hold"},
                "rejection_reason": reason,
                "rejection_category": category if not valid else "ok",
            }
        )
    return pd.DataFrame(rows)


def propagate_calibration(
    calibration: pd.DataFrame,
    *,
    camera_motion: pd.DataFrame | None = None,
    shot_segments: pd.DataFrame | None = None,
    config: PropagationConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill short gaps by carrying nearest measured H; mark provider=propagated."""
    cfg = config or PropagationConfig()
    cal = calibration.sort_values("frame_id").reset_index(drop=True).copy()
    cuts = set()
    if shot_segments is not None and not shot_segments.empty and "scene_cut" in shot_segments.columns:
        cuts = set(
            int(x) for x in shot_segments.loc[shot_segments["scene_cut"].astype(bool), "frame_id"]
        )

    measured_mask = cal["valid"].astype(bool) & cal["provider"].astype(str).ne("propagated")
    measured_ids = cal.loc[measured_mask, "frame_id"].astype(int).tolist()
    measured_H = {
        int(r.frame_id): _parse_H(r.homography_json)
        for r in cal.loc[measured_mask].itertuples(index=False)
    }
    measured_meta = {
        int(r.frame_id): r for r in cal.loc[measured_mask].itertuples(index=False)
    }

    before = float(cal["valid"].mean()) if len(cal) else 0.0
    propagated = 0
    rejected_jump = 0

    for i, row in cal.iterrows():
        if bool(row["valid"]):
            continue
        fid = int(row["frame_id"])
        if fid in cuts:
            continue
        # nearest measured without crossing a cut
        best = None
        best_dist = None
        for mid in measured_ids:
            dist = abs(mid - fid)
            if dist > cfg.max_propagate_frames:
                continue
            lo, hi = sorted((mid, fid))
            if any(c for c in cuts if lo < c <= hi):
                continue
            if best_dist is None or dist < best_dist:
                best = mid
                best_dist = dist
        if best is None:
            continue
        H = measured_H.get(best)
        meta = measured_meta.get(best)
        if H is None or meta is None:
            continue
        if float(getattr(meta, "reprojection_error", 0) or 0) > cfg.max_reprojection_error:
            continue
        if float(getattr(meta, "confidence", 0) or 0) < cfg.min_confidence:
            continue
        # Optional jump check vs previous valid neighbor
        prev_valid = cal.loc[(cal["frame_id"] < fid) & cal["valid"]].tail(1)
        if not prev_valid.empty:
            Hp = _parse_H(prev_valid.iloc[0]["homography_json"])
            if Hp is not None:
                jump = _H_jump(Hp, H)
                if jump > cfg.max_homography_jump:
                    rejected_jump += 1
                    continue
        cal.at[i, "valid"] = True
        cal.at[i, "provider"] = "propagated"
        cal.at[i, "homography_json"] = json.dumps(H.tolist())
        cal.at[i, "confidence"] = float(getattr(meta, "confidence", 0.0) or 0.0) * 0.85
        cal.at[i, "reprojection_error"] = getattr(meta, "reprojection_error", None)
        cal.at[i, "visible_pitch_coverage"] = getattr(meta, "visible_pitch_coverage", None)
        cal.at[i, "invalid_reason"] = None
        cal.at[i, "source_method"] = "propagated"
        propagated += 1

    after = float(cal["valid"].mean()) if len(cal) else 0.0
    measured_cov = float(measured_mask.mean()) if len(cal) else 0.0
    prop_mask = cal["provider"].astype(str).eq("propagated") & cal["valid"].astype(bool)
    stats = {
        "calibration_coverage_before": round(before, 4),
        "calibration_coverage_after": round(after, 4),
        "measured_coverage": round(measured_cov, 4),
        "propagated_coverage": round(float(prop_mask.mean()) if len(cal) else 0.0, 4),
        "propagated_frames": int(propagated),
        "rejected_homography_jumps": int(rejected_jump),
    }
    return cal, stats


def continuous_calibrated_seconds(calibration: pd.DataFrame, fps: float = 25.0) -> float:
    if calibration is None or calibration.empty:
        return 0.0
    valid = calibration.sort_values("frame_id")["valid"].astype(bool).tolist()
    best = run = 0
    for v in valid:
        if v:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best / max(fps, 1e-6)


def write_calibration_audit_bundle(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
    apply_propagation: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    cal = pd.read_parquet(run_dir / "calibration.parquet")
    motion = (
        pd.read_parquet(run_dir / "camera_motion.parquet")
        if (run_dir / "camera_motion.parquet").is_file()
        else None
    )
    shots = (
        pd.read_parquet(run_dir / "shot_segments.parquet")
        if (run_dir / "shot_segments.parquet").is_file()
        else None
    )
    before_cov = float(cal["valid"].mean()) if len(cal) else 0.0
    stats = {
        "calibration_coverage_before": round(before_cov, 4),
        "measured_coverage": round(
            float(
                (
                    cal["valid"].astype(bool)
                    & cal["provider"].astype(str).ne("propagated")
                ).mean()
            ),
            4,
        )
        if len(cal)
        else 0.0,
    }
    if apply_propagation:
        cal, prop_stats = propagate_calibration(cal, camera_motion=motion, shot_segments=shots)
        stats.update(prop_stats)
        cal.to_parquet(out / "calibration_propagated.parquet", index=False)
    audit = build_calibration_frame_audit(cal, camera_motion=motion, shot_segments=shots)
    audit.to_parquet(out / "calibration_frame_audit.parquet", index=False)
    stats["continuous_calibrated_seconds"] = continuous_calibrated_seconds(cal)
    stats["rejection_categories"] = (
        audit.loc[~audit["calibration_valid"], "rejection_category"].value_counts().to_dict()
    )
    (out / "calibration_evaluation_report.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return stats
