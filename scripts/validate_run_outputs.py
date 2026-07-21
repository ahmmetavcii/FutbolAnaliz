#!/usr/bin/env python3
"""Validate MVP-1 run artifacts open and Parquet schemas are readable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_analytics.contracts.schemas import (  # noqa: E402
    validate_detections_frame,
    validate_tracks_frame,
)


def validate_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read first frame: {path}")
    return {"path": str(path), "opened": True, "shape": list(frame.shape)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    required = [
        "video_manifest.json",
        "detections.parquet",
        "tracks.parquet",
        "annotated_video.mp4",
        "run_report.json",
        "stages/ingest/stage_manifest.json",
        "stages/detection/stage_manifest.json",
        "stages/tracking/stage_manifest.json",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise SystemExit(f"FAIL missing artifacts: {missing}")

    detections = pd.read_parquet(run_dir / "detections.parquet")
    tracks = pd.read_parquet(run_dir / "tracks.parquet")
    validate_detections_frame(list(detections.columns))
    validate_tracks_frame(list(tracks.columns))
    det_table = pq.read_table(run_dir / "detections.parquet")
    track_table = pq.read_table(run_dir / "tracks.parquet")

    videos = [
        run_dir / "annotated_video.mp4",
        run_dir / "detection_annotated.mp4",
        run_dir / "input" / "test_clip.mp4",
    ]
    video_results = []
    for path in videos:
        if path.exists():
            video_results.append(validate_video(path))

    for tracker in ("bytetrack", "botsort"):
        annotated = run_dir / "trackers" / tracker / "annotated_video.mp4"
        tracks_path = run_dir / "trackers" / tracker / "tracks.parquet"
        if annotated.exists():
            video_results.append(validate_video(annotated))
        if tracks_path.exists():
            frame = pd.read_parquet(tracks_path)
            validate_tracks_frame(list(frame.columns))

    summary = {
        "status": "PASS",
        "run_dir": str(run_dir),
        "detections_rows": int(len(detections)),
        "tracks_rows": int(len(tracks)),
        "unique_track_ids": int(tracks["track_id"].nunique()),
        "detections_parquet_rows": int(det_table.num_rows),
        "tracks_parquet_rows": int(track_table.num_rows),
        "videos_ok": video_results,
        "run_report_status": json.loads((run_dir / "run_report.json").read_text()).get(
            "status"
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
