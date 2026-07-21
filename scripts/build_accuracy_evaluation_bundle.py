#!/usr/bin/env python3
"""Build touch review pack + classify identity fragments + calibration audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_analytics.evaluation.calibration_audit import write_calibration_audit_bundle
from football_analytics.evaluation.identity_fragments import classify_team_identities
from football_analytics.evaluation.touch_review import build_touch_review_pack
from football_analytics.evaluation.ball_metrics import evaluate_ball_tracking
from football_analytics.evaluation.identity_metrics import evaluate_global_identity
from football_analytics.evaluation.publishability import compute_publishability
from football_analytics.utils.io import write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument(
        "--ball-gt",
        type=Path,
        default=Path("configs/evaluation/short_clip_gt_template/football_ball/ball_gt.csv"),
    )
    ap.add_argument(
        "--identity-gt",
        type=Path,
        default=Path("configs/evaluation/short_clip_gt_template/football_identity/identity_gt.csv"),
    )
    args = ap.parse_args()
    out = args.run_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    touch = build_touch_review_pack(run_dir=args.run_dir, video_path=args.video, out_dir=out / "touch_review")
    frags = classify_team_identities(args.run_dir, team_id="team_1", out_dir=out)
    # also team_0
    frags0 = classify_team_identities(args.run_dir, team_id="team_0", out_dir=out)
    cal = write_calibration_audit_bundle(args.run_dir, out_dir=out, apply_propagation=True)
    ball = evaluate_ball_tracking(gt_csv=args.ball_gt, run_dir=args.run_dir, out_dir=out)
    ident = evaluate_global_identity(gt_csv=args.identity_gt, run_dir=args.run_dir, out_dir=out)

    from football_analytics.evaluation.ball_gt import ball_gt_complete
    from football_analytics.evaluation.identity_gt import identity_gt_complete
    from football_analytics.evaluation.touch_review import touch_review_complete
    import pandas as pd

    ball_ok = ball_gt_complete(pd.read_csv(args.ball_gt))[0] if args.ball_gt.is_file() else False
    id_ok = (
        identity_gt_complete(pd.read_csv(args.identity_gt))[0] if args.identity_gt.is_file() else False
    )
    touch_ok = touch_review_complete(out / "touch_review" / "touch_review.csv")[0]

    iq = {}
    if (args.run_dir / "identity_quality.json").is_file():
        iq = json.loads((args.run_dir / "identity_quality.json").read_text())

    flags = compute_publishability(
        ball_gt_complete=ball_ok,
        ball_eval=ball if ball.get("status") == "OK" else None,
        identity_gt_complete=id_ok,
        identity_eval=ident if ident.get("status") == "OK" else None,
        touch_review_complete=touch_ok,
        calibration_coverage=cal.get("calibration_coverage_after"),
        measured_calibration_coverage=cal.get("measured_coverage"),
        player_position_coverage=None,
        continuous_calibrated_seconds=cal.get("continuous_calibrated_seconds"),
        speed_spike_candidates=None,
        identity_quality=iq,
    )
    if (args.run_dir / "game_state.parquet").is_file():
        gs = pd.read_parquet(args.run_dir / "game_state.parquet")
        if "valid" in gs.columns:
            flags = compute_publishability(
                ball_gt_complete=ball_ok,
                identity_gt_complete=id_ok,
                touch_review_complete=touch_ok,
                calibration_coverage=cal.get("calibration_coverage_after"),
                measured_calibration_coverage=cal.get("measured_coverage"),
                player_position_coverage=float(gs["valid"].mean()),
                continuous_calibrated_seconds=cal.get("continuous_calibrated_seconds"),
                speed_spike_candidates=len(
                    list((args.run_dir / "speed_spike_audit.csv").read_text().splitlines())
                )
                - 1
                if (args.run_dir / "speed_spike_audit.csv").is_file()
                else 0,
                identity_quality=iq,
            )
    write_json(out / "publishability_flags.json", flags.to_dict())
    write_json(args.run_dir / "publishability_flags.json", flags.to_dict())

    summary = {
        "touch_review": touch,
        "team_1_fragments": frags,
        "team_0_fragments": frags0,
        "calibration": cal,
        "ball_eval_status": ball.get("status"),
        "identity_eval_status": ident.get("status"),
        "publishability": flags.to_dict(),
    }
    write_json(out / "accuracy_remediation_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
