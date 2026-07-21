#!/usr/bin/env python3
"""Evaluate global identity; GT incomplete → no IDF1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_analytics.evaluation.identity_metrics import evaluate_global_identity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-csv", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    report = evaluate_global_identity(
        gt_csv=args.gt_csv, run_dir=args.run_dir, out_dir=args.out_dir
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
