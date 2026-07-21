#!/usr/bin/env python3
"""Create player identity GT sample frames for a continuous clip window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from football_analytics.evaluation.identity_gt import pick_identity_window, write_identity_gt_sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("configs/evaluation/short_clip_gt_template/football_identity"),
    )
    ap.add_argument("--target-seconds", type=float, default=20.0)
    ap.add_argument("--frame-stride", type=int, default=5)
    args = ap.parse_args()

    tracks = pd.read_parquet(args.run_dir / "tracks.parquet")
    shots = (
        pd.read_parquet(args.run_dir / "shot_segments.parquet")
        if (args.run_dir / "shot_segments.parquet").is_file()
        else None
    )
    start, end = pick_identity_window(
        tracks, shots, target_seconds=args.target_seconds, fps=25.0, min_players=10
    )
    manifest = write_identity_gt_sample(
        video_path=args.video,
        out_dir=args.out_dir,
        start_frame=start,
        end_frame=end,
        frame_stride=args.frame_stride,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
