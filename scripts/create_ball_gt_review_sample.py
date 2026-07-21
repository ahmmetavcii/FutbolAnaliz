#!/usr/bin/env python3
"""Create a balanced 50-frame ball GT review sample from the 165-frame pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from football_analytics.evaluation.ball_gt import load_ball_gt_csv
from football_analytics.evaluation.ball_review_sample import (
    create_balanced_review_sample,
    sample_reviewed_count,
    write_review_sample,
)
from football_analytics.evaluation.boolean_utils import count_true


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("configs/evaluation/short_clip_gt_template/football_ball"),
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/home/ahmet/workspace/opta_analytics_smoke/run_20260720_154807_15747b"),
    )
    ap.add_argument("--sample-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: <gt-dir>/review_sample.csv",
    )
    args = ap.parse_args()

    gt_path = args.gt_dir / "ball_gt.csv"
    out_path = args.out or (args.gt_dir / "review_sample.csv")
    gt = load_ball_gt_csv(gt_path)

    provenance = detections = None
    wo = None
    if args.run_dir.is_dir():
        if (args.run_dir / "ball_provenance.parquet").is_file():
            provenance = pd.read_parquet(args.run_dir / "ball_provenance.parquet")
        if (args.run_dir / "football_ball_detections.parquet").is_file():
            detections = pd.read_parquet(args.run_dir / "football_ball_detections.parquet")
        wo_path = args.run_dir / "wrong_object_ball_report.json"
        if wo_path.is_file():
            wo = json.loads(wo_path.read_text(encoding="utf-8"))

    sample = create_balanced_review_sample(
        ball_gt=gt,
        provenance=provenance,
        detections=detections,
        wrong_object_report=wo,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    write_review_sample(sample, out_path)

    # Do NOT mark ball_gt rows as reviewed
    auto_reviewed = count_true(sample["reviewed"])
    linked = sample_reviewed_count(sample, gt)
    summary = {
        "path": str(out_path),
        "sample_size": int(len(sample)),
        "unique_frames": int(sample["frame_idx"].nunique()),
        "automatic_reviewed_count": int(auto_reviewed),
        "already_reviewed_in_ball_gt": int(linked),
        "sampling_reasons": sample["sampling_reason"].value_counts().to_dict(),
    }
    print(json.dumps(summary, indent=2))
    if auto_reviewed != 0:
        raise SystemExit("automatic_reviewed_count must be 0")
    if len(sample) != args.sample_size or sample["frame_idx"].nunique() != args.sample_size:
        raise SystemExit("sample uniqueness/size check failed")


if __name__ == "__main__":
    main()
