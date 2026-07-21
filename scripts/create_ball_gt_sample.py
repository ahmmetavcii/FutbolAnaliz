#!/usr/bin/env python3
"""Create stratified football ball GT sample (empty annotations)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from football_analytics.evaluation.ball_gt import select_stratified_frames, write_ball_gt_sample


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("configs/evaluation/short_clip_gt_template/football_ball"),
    )
    p.add_argument("--n-frames", type=int, default=165)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-stratum", type=int, default=15)
    args = p.parse_args()

    detections = provenance = camera_motion = shots = tracks = None
    total_frames = None
    if args.run_dir and args.run_dir.is_dir():
        r = args.run_dir
        if (r / "football_ball_detections.parquet").is_file():
            detections = pd.read_parquet(r / "football_ball_detections.parquet")
        if (r / "ball_provenance.parquet").is_file():
            provenance = pd.read_parquet(r / "ball_provenance.parquet")
        if (r / "camera_motion.parquet").is_file():
            camera_motion = pd.read_parquet(r / "camera_motion.parquet")
        if (r / "shot_segments.parquet").is_file():
            shots = pd.read_parquet(r / "shot_segments.parquet")
        if (r / "tracks.parquet").is_file():
            tracks = pd.read_parquet(r / "tracks.parquet")
        if (r / "ball_detection_report.json").is_file():
            import json

            total_frames = int(
                json.loads((r / "ball_detection_report.json").read_text()).get("frames") or 0
            )

    if total_frames is None:
        import cv2

        cap = cv2.VideoCapture(str(args.video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

    specs = select_stratified_frames(
        n_frames=args.n_frames,
        total_frames=total_frames,
        seed=args.seed,
        detections=detections,
        provenance=provenance,
        camera_motion=camera_motion,
        shot_segments=shots,
        tracks=tracks,
        per_stratum=args.per_stratum,
    )
    manifest = write_ball_gt_sample(
        video_path=args.video,
        out_dir=args.out_dir,
        frame_specs=specs,
        detections=detections,
        seed=args.seed,
    )
    print(json_dumps(manifest))


def json_dumps(obj: dict) -> str:
    import json

    return json.dumps(obj, indent=2)


if __name__ == "__main__":
    main()
