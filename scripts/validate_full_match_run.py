#!/usr/bin/env python3
"""Validate full-match manifests, checksums, schemas, and completion state."""

from __future__ import annotations

import argparse
from pathlib import Path

from _full_match_cli import call_api, emit, require_dir, resolve_api, run_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checksums", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-media", action="store_true", help="Decode sample frames from media")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as validation failures")
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api = resolve_api(
        "football_analytics.full_match",
        ("validate_full_match_run", "validate_run"),
    )
    return emit(
        call_api(
            api,
            run_dir=require_dir(args.run_dir, "run directory"),
            verify_checksums=args.checksums,
            open_media=args.open_media,
            strict=args.strict,
            report_path=args.report,
        )
    )


if __name__ == "__main__":
    run_cli(main)
