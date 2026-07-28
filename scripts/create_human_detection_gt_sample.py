#!/usr/bin/env python3
"""Create a balanced 40-frame human-detection GT review sample for football.mp4.

Does NOT mark any box as reviewed=true. Model proposals are suggestions only.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path("/home/ahmet/projects/football-analytics")
VIDEO = Path("/mnt/c/football_data/videos/test_clips/football.mp4")
OUT = ROOT / "configs/evaluation/human_detection_gt/football"
BEST = ROOT / "third_party/authorized/match-node-tracker/train/weights/best.pt"
PROD = Path("/home/ahmet/models/yolo11n.pt")

# Balanced strata → target frame indices (750 frames @ 25fps)
STRATA = {
    "wide_main": [10, 40, 80],
    "small_distant": [120, 200, 280],
    "dense_midfield": [150, 220, 320],
    "penalty_area": [180, 260, 400],
    "occlusion": [90, 240, 360],
    "fast_pan": [300, 420, 500],
    "zoom": [330, 450],
    "referee_visible": [60, 160, 350, 520],
    "goalkeeper_view": [100, 380, 560],
    "close_up": [480, 600],
    "bench_area": [50, 540],
    "stand_hard_negative": [5, 700],
    "scene_cut_or_replay": [250, 620, 680],
}


def pick_frames() -> list[tuple[int, str]]:
    picks: list[tuple[int, str]] = []
    seen = set()
    for stratum, idxs in STRATA.items():
        for i in idxs:
            if i not in seen:
                picks.append((i, stratum))
                seen.add(i)
    # pad / trim to exactly 40
    if len(picks) < 40:
        for i in range(0, 750, 15):
            if i not in seen:
                picks.append((i, "uniform_fill"))
                seen.add(i)
            if len(picks) >= 40:
                break
    picks = sorted(picks, key=lambda x: x[0])[:40]
    return picks


def main() -> None:
    if OUT.exists():
        # keep existing reviewed GT if any
        gt_path = OUT / "human_detection_gt.csv"
        if gt_path.exists():
            existing = pd.read_csv(gt_path)
            if "reviewed" in existing.columns and bool((existing["reviewed"] == True).any()):  # noqa: E712
                print("Existing reviewed GT found — not wiping. Update sample_manifest only if needed.")
                return

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "frames").mkdir(parents=True)

    cap = cv2.VideoCapture(str(VIDEO))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    picks = pick_frames()
    assert len(picks) == 40

    sample_rows = []
    gt_rows = []  # empty boxes; annotator fills — reviewed=false
    for order, (frame_idx, stratum) in enumerate(picks, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        fname = f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(OUT / "frames" / fname), frame)
        sample_rows.append(
            {
                "sample_order": order,
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps,
                "sampling_reason": stratum,
                "frame_file": fname,
                "reviewed": False,
            }
        )
        # placeholder row so schema exists (no invented boxes)
        gt_rows.append(
            {
                "frame_idx": frame_idx,
                "gt_id": f"f{frame_idx}_placeholder",
                "class_name": "not_target",
                "x1": np.nan,
                "y1": np.nan,
                "x2": np.nan,
                "y2": np.nan,
                "occluded": False,
                "difficult": False,
                "reviewed": False,
                "notes": "placeholder — replace with real boxes via annotate_human_detection_gt.py",
            }
        )
    cap.release()

    sample = pd.DataFrame(sample_rows)
    gt = pd.DataFrame(gt_rows)
    sample.to_csv(OUT / "review_sample.csv", index=False)
    gt.to_csv(OUT / "human_detection_gt.csv", index=False)
    manifest = {
        "video": str(VIDEO),
        "video_name": "football",
        "n_sample_frames": int(len(sample)),
        "frame_width": w,
        "frame_height": h,
        "fps": fps,
        "classes": ["player", "goalkeeper", "referee", "ignore_person", "not_target"],
        "reviewed_policy": "Model proposals are NEVER auto-accepted as GT",
        "completion": {"reviewed_frames": 0, "target_frames": 40},
        "strata": STRATA,
        "upstream_commit": "2777aa3f1e9cc563eba07a675cebdf4bfd9306bf",
    }
    (OUT / "sample_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {OUT} with {len(sample)} frames, reviewed=0/40")


if __name__ == "__main__":
    main()
