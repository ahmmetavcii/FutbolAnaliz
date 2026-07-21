#!/usr/bin/env python3
"""Run jersey inference on one tracklet directory or a directory of tracklets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_analytics.jersey.dataset import IMAGE_SUFFIXES  # noqa: E402
from football_analytics.jersey.infer import record_from_directory, run_inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Tracklet folder or parent of tracklet folders")
    parser.add_argument("--output", default="artifacts/jersey/inference/predictions.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence-threshold", type=float)
    args = parser.parse_args()
    source = Path(args.input)
    has_images = any(
        item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES for item in source.iterdir()
    )
    folders = [source] if has_images else sorted(item for item in source.iterdir() if item.is_dir())
    records = [record_from_directory(folder) for folder in folders]
    predictions = run_inference(
        args.checkpoint,
        records,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        output_path=args.output,
    )
    print(json.dumps([prediction.to_dict() for prediction in predictions], indent=2))


if __name__ == "__main__":
    main()
