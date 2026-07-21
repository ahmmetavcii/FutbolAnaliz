#!/usr/bin/env python3
"""Validate canonical MVP-2 artifacts and report measured coverage."""

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

from football_analytics.contracts.schemas import MVP2_SCHEMAS, validate_mvp2_columns


def ratio(series: pd.Series) -> float:
    return float(series.fillna(False).mean()) if len(series) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    artifacts: dict[str, dict[str, object]] = {}
    for name, schema in MVP2_SCHEMAS.items():
        path = run_dir / f"{name}.parquet"
        if not path.is_file():
            raise RuntimeError(f"Missing canonical artifact: {path}")
        frame = pd.read_parquet(path)
        validate_mvp2_columns(name, list(frame.columns))
        table = pq.read_table(path)
        if table.schema.names != schema.names:
            raise RuntimeError(f"{name} column order/schema mismatch")
        artifacts[name] = {"rows": len(frame), "readable": True}

    videos = {}
    for name in ("analytics_annotated.mp4", "tactical_preview.mp4"):
        path = run_dir / name
        capture = cv2.VideoCapture(str(path))
        ok, image = capture.read()
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
        if not ok or image is None:
            raise RuntimeError(f"Unreadable video: {path}")
        videos[name] = {"frames": frame_count, "shape": list(image.shape)}

    for stage in (
        "ingest",
        "shot_classification",
        "detection",
        "tracking",
        "track_quality",
        "reid",
        "team_identity",
        "camera_motion",
        "calibration",
        "ball_state",
        "possession",
        "metrics",
        "analytics_render",
    ):
        path = run_dir / "stages" / stage / "stage_manifest.json"
        if not path.is_file():
            raise RuntimeError(f"Missing stage manifest: {path}")
        if json.loads(path.read_text()).get("status") != "completed":
            raise RuntimeError(f"Stage not completed: {stage}")

    identities = pd.read_parquet(run_dir / "track_identities.parquet")
    ball = pd.read_parquet(run_dir / "ball_state.parquet")
    possession = pd.read_parquet(run_dir / "possession_timeline.parquet")
    calibration = pd.read_parquet(run_dir / "calibration.parquet")
    player_metrics = pd.read_parquet(run_dir / "player_metrics.parquet")
    summary = {
        "status": "PASS",
        "run_dir": str(run_dir),
        "artifacts": artifacts,
        "videos": videos,
        "coverage": {
            "valid_calibration": ratio(calibration["valid"]),
            "team_assignment": float(identities["team_id"].notna().mean())
            if len(identities)
            else 0.0,
            "ball_detected": float((ball["visibility_state"] == "detected").mean())
            if len(ball)
            else 0.0,
            "ball_known": float(
                ball["visibility_state"]
                .isin(["detected", "predicted", "occluded_short", "airborne"])
                .mean()
            )
            if len(ball)
            else 0.0,
            "possession_known": float(
                (
                    ~possession["possession_state"].isin(
                        ["unknown", "loose_ball"]
                    )
                ).mean()
            )
            if len(possession)
            else 0.0,
            "speed_valid": ratio(player_metrics["valid"]),
            "possession_unknown": float(
                (possession["possession_state"] == "unknown").mean()
            )
            if len(possession)
            else 0.0,
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
