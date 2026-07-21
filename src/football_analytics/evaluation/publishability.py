"""Independent publishability flags (no single misleading stats_publishable)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class PublishabilityFlags:
    identity_publishable: bool = False
    ball_detection_publishable: bool = False
    ball_tracking_publishable: bool = False
    touch_publishable: bool = False
    action_stats_publishable: bool = False
    calibration_publishable: bool = False
    physical_metrics_publishable: bool = False
    tactical_metrics_publishable: bool = False
    overall_publishable: bool = False
    reasons: dict[str, str] = field(default_factory=dict)
    gt_incomplete: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_thresholds(path: Path | None = None) -> dict[str, Any]:
    default = Path(__file__).resolve().parents[3] / "configs/quality/publishability_thresholds.yaml"
    cfg_path = path or default
    if not cfg_path.is_file():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def compute_publishability(
    *,
    ball_gt_complete: bool = False,
    ball_eval: Mapping[str, Any] | None = None,
    identity_gt_complete: bool = False,
    identity_eval: Mapping[str, Any] | None = None,
    touch_review_complete: bool = False,
    touch_eval: Mapping[str, Any] | None = None,
    calibration_coverage: float | None = None,
    measured_calibration_coverage: float | None = None,
    player_position_coverage: float | None = None,
    continuous_calibrated_seconds: float | None = None,
    max_reprojection_error: float | None = None,
    speed_spike_candidates: int | None = None,
    identity_quality: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> PublishabilityFlags:
    thr = dict(thresholds or load_thresholds())
    flags = PublishabilityFlags()
    flags.gt_incomplete = {
        "ball": not ball_gt_complete,
        "identity": not identity_gt_complete,
        "touch": not touch_review_complete,
    }

    # --- Ball detection / tracking (requires completed GT eval) ---
    ball_thr = thr.get("ball") or {}
    if not ball_gt_complete:
        flags.ball_detection_publishable = False
        flags.ball_tracking_publishable = False
        flags.reasons["ball_detection"] = "GT_INCOMPLETE"
        flags.reasons["ball_tracking"] = "GT_INCOMPLETE"
    else:
        be = dict(ball_eval or {})
        prec = be.get("precision")
        rec = be.get("recall")
        ok_det = (
            prec is not None
            and rec is not None
            and float(prec) >= float(ball_thr.get("min_precision", 0.7))
            and float(rec) >= float(ball_thr.get("min_recall", 0.6))
        )
        flags.ball_detection_publishable = bool(ok_det)
        flags.reasons["ball_detection"] = "ok" if ok_det else "below_threshold"
        traj = be.get("trajectory_coverage")
        ok_trk = traj is not None and float(traj) >= float(
            ball_thr.get("min_trajectory_coverage", 0.5)
        )
        flags.ball_tracking_publishable = bool(ok_det and ok_trk)
        flags.reasons["ball_tracking"] = "ok" if flags.ball_tracking_publishable else "below_threshold"

    # --- Identity ---
    id_thr = thr.get("identity") or {}
    iq = dict(identity_quality or {})
    if not identity_gt_complete:
        flags.identity_publishable = False
        flags.reasons["identity"] = "GT_INCOMPLETE"
    else:
        ie = dict(identity_eval or {})
        idf1 = ie.get("idf1")
        ok = idf1 is not None and float(idf1) >= float(id_thr.get("min_idf1", 0.5))
        # Still require validated counts ≤ 11 when GT eval exists
        validated = iq.get("validated_by_team") or {}
        over = any(
            int(v) > int(id_thr.get("max_validated_per_team", 11)) for v in validated.values()
        )
        flags.identity_publishable = bool(ok and not over)
        flags.reasons["identity"] = (
            "ok" if flags.identity_publishable else ("over_11" if over else "below_idf1")
        )

    # --- Touch ---
    if not touch_review_complete:
        flags.touch_publishable = False
        flags.reasons["touch"] = "REVIEW_INCOMPLETE"
    else:
        te = dict(touch_eval or {})
        prec = te.get("precision")
        touch_thr = thr.get("touch") or {}
        ok = prec is not None and float(prec) >= float(touch_thr.get("min_precision", 0.7))
        flags.touch_publishable = bool(ok)
        flags.reasons["touch"] = "ok" if ok else "below_threshold"

    # --- Calibration (coverage-based; not accuracy claim) ---
    cal_thr = thr.get("calibration") or {}
    frame_cov = calibration_coverage
    measured = measured_calibration_coverage
    pos_cov = player_position_coverage
    cont_s = continuous_calibrated_seconds
    reproj = max_reprojection_error
    cal_ok = True
    cal_reason = "ok"
    if frame_cov is None or float(frame_cov) < float(cal_thr.get("min_frame_coverage", 0.5)):
        cal_ok = False
        cal_reason = "low_frame_coverage"
    elif measured is not None and float(measured) < float(cal_thr.get("min_measured_coverage", 0.25)):
        cal_ok = False
        cal_reason = "low_measured_coverage"
    elif cont_s is not None and float(cont_s) < float(
        cal_thr.get("min_continuous_calibrated_seconds", 2.0)
    ):
        cal_ok = False
        cal_reason = "short_continuous_calibration"
    elif reproj is not None and float(reproj) > float(cal_thr.get("max_reprojection_error", 8.0)):
        cal_ok = False
        cal_reason = "high_reprojection_error"
    elif pos_cov is not None and float(pos_cov) < float(
        cal_thr.get("min_player_position_coverage", 0.4)
    ):
        cal_ok = False
        cal_reason = "low_player_position_coverage"
    flags.calibration_publishable = bool(cal_ok)
    flags.reasons["calibration"] = cal_reason

    # --- Physical metrics ---
    phys_thr = thr.get("physical_metrics") or {}
    phys_ok = True
    phys_reason = "ok"
    if phys_thr.get("require_calibration_publishable", True) and not flags.calibration_publishable:
        phys_ok = False
        phys_reason = "calibration_not_publishable"
    elif (
        phys_thr.get("require_speed_spike_audit_pass", True)
        and speed_spike_candidates is not None
        and int(speed_spike_candidates) > int(phys_thr.get("max_speed_spike_candidates", 0))
    ):
        phys_ok = False
        phys_reason = "speed_spike_audit_failed"
    elif pos_cov is not None and float(pos_cov) < float(
        phys_thr.get("min_player_position_coverage", 0.4)
    ):
        phys_ok = False
        phys_reason = "low_player_position_coverage"
    flags.physical_metrics_publishable = bool(phys_ok)
    flags.reasons["physical_metrics"] = phys_reason

    # --- Tactical / action ---
    tac = thr.get("tactical_metrics") or {}
    flags.tactical_metrics_publishable = bool(
        (not tac.get("require_calibration_publishable", True) or flags.calibration_publishable)
        and (not tac.get("require_identity_publishable", False) or flags.identity_publishable)
    )
    flags.reasons["tactical_metrics"] = "ok" if flags.tactical_metrics_publishable else "gated"

    act = thr.get("action_stats") or {}
    flags.action_stats_publishable = bool(
        (not act.get("require_identity_publishable", True) or flags.identity_publishable)
        and (not act.get("require_touch_publishable", False) or flags.touch_publishable)
    )
    flags.reasons["action_stats"] = "ok" if flags.action_stats_publishable else "gated"

    # --- Overall: never true without GT-backed accuracy gates ---
    overall_req = (thr.get("overall") or {}).get("require") or [
        "ball_detection_publishable",
        "identity_publishable",
        "calibration_publishable",
    ]
    mapping = {
        "ball_detection_publishable": flags.ball_detection_publishable,
        "identity_publishable": flags.identity_publishable,
        "calibration_publishable": flags.calibration_publishable,
        "touch_publishable": flags.touch_publishable,
        "physical_metrics_publishable": flags.physical_metrics_publishable,
    }
    flags.overall_publishable = all(bool(mapping.get(k, False)) for k in overall_req)
    if not ball_gt_complete or not identity_gt_complete:
        flags.overall_publishable = False
        flags.reasons["overall"] = "GT_INCOMPLETE"
    else:
        flags.reasons["overall"] = "ok" if flags.overall_publishable else "gates_failed"

    return flags
