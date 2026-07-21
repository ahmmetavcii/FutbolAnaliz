#!/usr/bin/env python3
"""Orchestrate a prepared full match without substituting placeholder model stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from _full_match_cli import (
    ROOT,
    call_api,
    emit,
    load_yaml,
    require_dir,
    require_file,
    resolve_api,
    run_cli,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/full_match/single_camera.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--from-chunk", type=int)
    parser.add_argument("--until-chunk", type=int)
    parser.add_argument(
        "--chunk-pipeline-config",
        type=Path,
        help=(
            "Opt in to the existing per-chunk pipeline adapter. The full-match package "
            "must provide the adapter; this CLI does not claim missing model stages."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the orchestration request only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared_dir = require_dir(args.prepared_dir, "prepared directory")
    config_path = require_file(args.config, "config")
    config = load_yaml(config_path)
    adapter_config = (
        require_file(args.chunk_pipeline_config, "chunk pipeline config")
        if args.chunk_pipeline_config
        else None
    )
    request = {
        "status": "PASS",
        "mode": "infrastructure_dry_run" if args.dry_run else "orchestrate",
        "prepared_dir": str(prepared_dir),
        "run_dir": str(args.run_dir.expanduser().resolve()),
        "profile": config.get("full_match", {}).get("profile"),
        "chunk_seconds": config.get("full_match", {}).get("chunk_seconds"),
        "sync": not args.skip_sync,
        "calibration": not args.skip_calibration,
        "from_chunk": args.from_chunk,
        "until_chunk": args.until_chunk,
        "chunk_pipeline_adapter": str(adapter_config) if adapter_config else None,
    }
    if args.dry_run:
        return emit(request)

    api = resolve_api(
        "football_analytics.full_match",
        ("run_full_match", "orchestrate_full_match"),
    )
    result = call_api(
        api,
        prepared_dir=prepared_dir,
        config=config,
        config_path=config_path,
        run_dir=args.run_dir.expanduser().resolve(),
        synchronize=not args.skip_sync,
        calibrate=not args.skip_calibration,
        from_chunk=args.from_chunk,
        until_chunk=args.until_chunk,
        chunk_pipeline_config=adapter_config,
    )
    return emit(result)


if __name__ == "__main__":
    run_cli(main)
