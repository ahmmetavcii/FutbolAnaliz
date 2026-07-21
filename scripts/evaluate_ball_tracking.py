#!/usr/bin/env python3
"""Evaluate ball tracking on the 50-frame review sample; incomplete → no P/R."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_analytics.evaluation.ball_metrics import evaluate_ball_tracking


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-csv", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--sample",
        type=Path,
        default=None,
        help="review_sample.csv (default: alongside gt-csv)",
    )
    ap.add_argument("--sample-size", type=int, default=50)
    args = ap.parse_args()
    report = evaluate_ball_tracking(
        gt_csv=args.gt_csv,
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        review_sample_csv=args.sample,
        sample_size=args.sample_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
