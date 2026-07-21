#!/usr/bin/env python
"""CLI wrapper for the single-button panel analysis flow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full panel analysis flow")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--match-dir", required=True, type=Path)
    parser.add_argument("--camera-id", default="camera_1")
    parser.add_argument("--chunk-seconds", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from football_analytics.full_match.panel_driver import run_analysis

    state = run_analysis(
        args.video,
        args.match_dir,
        camera_id=args.camera_id,
        chunk_seconds=args.chunk_seconds,
        resume=args.resume,
    )
    return 0 if state.get("status") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
