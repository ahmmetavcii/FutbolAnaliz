#!/usr/bin/env python3
"""Calibrate cameras in a prepared full-match run."""

from __future__ import annotations

import argparse
from pathlib import Path

from _full_match_cli import call_api, emit, require_dir, require_file, resolve_api, run_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument(
        "--provider",
        choices=("auto", "metadata", "manual", "sn_calibration", "pnlcalib"),
        default="auto",
    )
    parser.add_argument("--camera-id", action="append", help="Limit calibration to selected cameras")
    parser.add_argument("--manual-calibration", type=Path, help="Verified calibration JSON")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared_dir = require_dir(args.prepared_dir, "prepared directory")
    manual = (
        require_file(args.manual_calibration, "manual calibration")
        if args.manual_calibration
        else None
    )
    if args.provider == "manual" and manual is None:
        raise ValueError("--manual-calibration is required when --provider=manual")
    api = resolve_api(
        "football_analytics.multicamera",
        ("calibrate_cameras", "calibrate_run"),
    )
    return emit(
        call_api(
            api,
            prepared_dir=prepared_dir,
            provider=args.provider,
            camera_ids=args.camera_id,
            manual_calibration=manual,
            output_dir=args.output_dir,
            force=args.force,
        )
    )


if __name__ == "__main__":
    run_cli(main)
