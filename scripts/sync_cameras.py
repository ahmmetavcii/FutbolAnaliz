#!/usr/bin/env python3
"""Estimate or apply camera time offsets for a prepared full-match run."""

from __future__ import annotations

import argparse
from pathlib import Path

from _full_match_cli import call_api, emit, require_dir, require_file, resolve_api, run_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("audio", "timecode", "manual"),
        default="audio",
        help="Synchronization signal",
    )
    parser.add_argument("--reference-camera", help="Defaults to the first manifest camera")
    parser.add_argument("--offsets", type=Path, help="Required JSON/YAML offsets for manual mode")
    parser.add_argument("--output", type=Path, help="Override synchronization artifact path")
    parser.add_argument("--max-offset-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared_dir = require_dir(args.prepared_dir, "prepared directory")
    offsets = require_file(args.offsets, "offsets") if args.offsets else None
    if args.method == "manual" and offsets is None:
        raise ValueError("--offsets is required when --method=manual")
    api = resolve_api(
        "football_analytics.multicamera",
        ("synchronize_cameras", "sync_cameras"),
    )
    return emit(
        call_api(
            api,
            prepared_dir=prepared_dir,
            method=args.method,
            reference_camera=args.reference_camera,
            offsets_path=offsets,
            output_path=args.output,
            max_offset_seconds=args.max_offset_seconds,
        )
    )


if __name__ == "__main__":
    run_cli(main)
