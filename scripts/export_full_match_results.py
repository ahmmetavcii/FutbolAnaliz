#!/usr/bin/env python3
"""Export a validated full-match run to a selected interchange format."""

from __future__ import annotations

import argparse
from pathlib import Path

from _full_match_cli import call_api, emit, require_dir, resolve_api, run_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--format",
        action="append",
        choices=("canonical", "json", "csv", "parquet", "soccernet"),
        dest="formats",
        help="Repeat to produce multiple formats; defaults to canonical",
    )
    parser.add_argument("--include-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api = resolve_api(
        "football_analytics.export",
        ("export_full_match_results", "export_run"),
    )
    return emit(
        call_api(
            api,
            run_dir=require_dir(args.run_dir, "run directory"),
            output_dir=args.output_dir.expanduser().resolve(),
            formats=args.formats or ["canonical"],
            include_video=args.include_video,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    run_cli(main)
