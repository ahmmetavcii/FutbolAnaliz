#!/usr/bin/env python3
"""Recompute downstream events and exports after operator corrections."""

from __future__ import annotations

import argparse
from pathlib import Path

from _full_match_cli import call_api, emit, require_dir, require_file, resolve_api, run_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument(
        "--from-stage",
        default="events",
        help="First derived stage to invalidate (default: events)",
    )
    parser.add_argument("--output-run-dir", type=Path, help="Clone derived outputs to a new run")
    parser.add_argument("--in-place", action="store_true", help="Update the existing run")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.output_run_dir) == bool(args.in_place):
        raise ValueError("choose exactly one of --output-run-dir or --in-place")
    api = resolve_api(
        "football_analytics.events",
        ("recompute_after_manual_correction", "recompute_events"),
    )
    return emit(
        call_api(
            api,
            run_dir=require_dir(args.run_dir, "run directory"),
            corrections_path=require_file(args.corrections, "corrections"),
            from_stage=args.from_stage,
            output_run_dir=args.output_run_dir,
            in_place=args.in_place,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    run_cli(main)
