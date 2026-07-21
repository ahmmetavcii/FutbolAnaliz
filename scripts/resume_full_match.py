#!/usr/bin/env python3
"""Resume a full-match run from validated stage and chunk manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from _full_match_cli import call_api, emit, require_dir, resolve_api, run_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rerun-from-stage", help="Invalidate this stage and all downstream stages")
    parser.add_argument("--from-chunk", type=int, help="Resume no earlier than this chunk index")
    parser.add_argument("--repair-manifests", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = require_dir(args.run_dir, "run directory")
    api = resolve_api(
        "football_analytics.full_match",
        ("resume_full_match", "resume_run"),
    )
    return emit(
        call_api(
            api,
            run_dir=run_dir,
            rerun_from_stage=args.rerun_from_stage,
            from_chunk=args.from_chunk,
            repair_manifests=args.repair_manifests,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    run_cli(main)
