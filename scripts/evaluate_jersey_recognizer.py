#!/usr/bin/env python3
"""Evaluate a jersey recognizer checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_analytics.jersey.evaluate import evaluate_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--output-dir", default="artifacts/jersey/evaluation")
    args = parser.parse_args()
    result = evaluate_checkpoint(
        args.checkpoint,
        dataset_root=args.dataset_root,
        split=args.split,
        subset_size=args.subset,
        device=args.device,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
