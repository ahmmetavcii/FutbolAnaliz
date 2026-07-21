#!/usr/bin/env python3
"""Probe camera inputs and prepare resumable full-match chunk manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from _full_match_cli import (
    ROOT,
    CliError,
    call_api,
    emit,
    load_yaml,
    require_file,
    resolve_api,
    run_cli,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Camera video; repeat per camera")
    parser.add_argument("--camera-id", action="append", help="Camera ID in --input order")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/full_match/single_camera.yaml",
        help="Full-match YAML profile",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--force", action="store_true", help="Replace an incomplete preparation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = require_file(args.config, "config")
    config = load_yaml(config_path)
    inputs = [require_file(value, "camera input") for value in args.input]
    camera_ids = args.camera_id or [f"camera_{index + 1}" for index in range(len(inputs))]
    if len(camera_ids) != len(inputs) or len(set(camera_ids)) != len(camera_ids):
        raise CliError("--camera-id values must be unique and match the number of --input values")
    expected = config.get("cameras", {}).get("expected_count")
    if expected is not None and int(expected) != len(inputs):
        raise CliError(f"profile expects {expected} camera input(s), received {len(inputs)}")

    api = resolve_api(
        "football_analytics.full_match",
        ("prepare_full_match", "prepare_run"),
    )
    result = call_api(
        api,
        inputs=inputs,
        camera_inputs=inputs,
        camera_ids=camera_ids,
        config=config,
        config_path=config_path,
        output_dir=args.output_dir.expanduser().resolve(),
        match_id=args.match_id,
        force=args.force,
    )
    return emit(result)


if __name__ == "__main__":
    run_cli(main)
