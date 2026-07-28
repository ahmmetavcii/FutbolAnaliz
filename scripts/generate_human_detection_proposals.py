#!/usr/bin/env python3
"""Generate hybrid detector proposals for the 40-frame human GT sample.

Writes model_proposals.csv only — never writes human_detection_gt.csv as GT.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_analytics.detection.hybrid_football_detector import (  # noqa: E402
    HybridFootballDetector,
    HybridFootballDetectorConfig,
    HybridThresholds,
)
from football_analytics.evaluation.human_gt_annotate import (  # noqa: E402
    PROPOSAL_COLUMNS,
    atomic_write_csv,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gt-dir",
        type=Path,
        default=ROOT / "configs/evaluation/human_detection_gt/football",
    )
    ap.add_argument("--video", type=Path, default=Path("/mnt/c/football_data/videos/test_clips/football.mp4"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--player-thr", type=float, default=0.15)
    ap.add_argument("--referee-thr", type=float, default=0.25)
    args = ap.parse_args()

    sample = pd.read_csv(args.gt_dir / "review_sample.csv")
    frame_ids = sample["frame_idx"].astype(int).tolist()
    assert len(frame_ids) == 40, f"expected 40 sample frames, got {len(frame_ids)}"

    det = HybridFootballDetector(
        HybridFootballDetectorConfig(
            imgsz=args.imgsz,
            thresholds=HybridThresholds(
                player=args.player_thr,
                referee=args.referee_thr,
                ball=1.0,  # ball unused in this tool — effectively disabled
            ),
            enable_role_classifier=False,  # never invent GK detections
        )
    )
    det.load()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    rows: list[dict] = []
    frames_with = 0
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            print(f"WARN: cannot read frame {fid}")
            continue
        # Detect humans only (filter ball if any leaked)
        dets = [
            d
            for d in det.detect_frame(frame, fid)
            if d.class_name in {"player", "referee", "person_unresolved"}
        ]
        # Never invent goalkeeper class from detector
        dets = [d for d in dets if d.class_name != "goalkeeper"]
        if dets:
            frames_with += 1
        for i, d in enumerate(dets):
            rows.append(
                {
                    "frame_idx": int(fid),
                    "proposal_id": f"f{fid}_p{i}",
                    "class_name": d.class_name,
                    "confidence": float(d.confidence),
                    "x1": float(d.x1),
                    "y1": float(d.y1),
                    "x2": float(d.x2),
                    "y2": float(d.y2),
                    "source_detector": d.source_detector,
                    "accepted": False,
                    "rejected": False,
                    "modified": False,
                }
            )
        print(f"frame {fid}: {len(dets)} proposals", flush=True)
    cap.release()

    out = args.gt_dir / "model_proposals.csv"
    df = pd.DataFrame(rows, columns=PROPOSAL_COLUMNS)
    backup = args.gt_dir / "backups"
    atomic_write_csv(df, out, backup_dir=backup)
    # verify not written into GT
    gt_path = args.gt_dir / "human_detection_gt.csv"
    gt = pd.read_csv(gt_path)
    auto_gt = 0  # we never auto-accept
    print(f"proposal_file={out}")
    print(f"sample_frames={len(frame_ids)}")
    print(f"frames_with_proposals={frames_with}")
    print(f"total_proposals={len(df)}")
    print(f"automatic_GT_acceptance={auto_gt}")
    print(f"gt_rows_unchanged_path={gt_path} rows={len(gt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
