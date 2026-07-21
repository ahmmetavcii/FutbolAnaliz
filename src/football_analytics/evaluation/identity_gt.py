"""Player identity GT I/O and sample selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import pandas as pd

IDENTITY_GT_COLUMNS = [
    "frame_index",
    "timestamp",
    "gt_player_id",
    "team_id",
    "role",
    "x1",
    "y1",
    "x2",
    "y2",
    "visible",
    "occluded",
    "difficult",
    "reviewed",
]


def identity_gt_complete(gt: pd.DataFrame, *, min_frames: int = 50, min_players: int = 8) -> tuple[bool, str]:
    if gt is None or gt.empty:
        return False, "empty_gt"
    frames = gt["frame_index"].nunique() if "frame_index" in gt.columns else 0
    players = gt["gt_player_id"].nunique() if "gt_player_id" in gt.columns else 0
    if frames < min_frames:
        return False, f"too_few_frames:{frames}<{min_frames}"
    if players < min_players:
        return False, f"too_few_players:{players}<{min_players}"
    if "reviewed" not in gt.columns:
        return False, "missing_reviewed"
    if not bool(gt["reviewed"].fillna(False).astype(bool).all()):
        return False, "unreviewed_rows"
    return True, "complete"


def pick_identity_window(
    tracks: pd.DataFrame,
    shot_segments: pd.DataFrame | None,
    *,
    target_seconds: float = 20.0,
    fps: float = 25.0,
    min_players: int = 10,
) -> tuple[int, int]:
    """Pick a continuous window with many players and few cuts."""
    total = int(tracks["frame_id"].max()) + 1 if not tracks.empty else 0
    win = max(int(target_seconds * fps), 50)
    if total <= win:
        return 0, max(total - 1, 0)

    cuts = set()
    if shot_segments is not None and not shot_segments.empty and "scene_cut" in shot_segments.columns:
        cuts = set(
            int(x) for x in shot_segments.loc[shot_segments["scene_cut"].astype(bool), "frame_id"]
        )

    person = tracks[tracks["object_type"].eq("person")] if "object_type" in tracks.columns else tracks
    best_start, best_score = 0, -1.0
    step = max(win // 4, 1)
    for start in range(0, total - win, step):
        end = start + win
        cut_pen = sum(1 for c in cuts if start <= c < end)
        n_players = person.loc[
            (person["frame_id"] >= start) & (person["frame_id"] < end), "track_id"
        ].nunique()
        score = float(n_players) - 5.0 * cut_pen
        if n_players >= min_players:
            score += 5.0
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start + win - 1


def write_identity_gt_sample(
    *,
    video_path: Path,
    out_dir: Path,
    start_frame: int,
    end_frame: int,
    frame_stride: int = 5,
    video_name: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    name = video_name or Path(video_path).stem
    saved = []
    for fid in range(start_frame, end_frame + 1, max(frame_stride, 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            continue
        path = frames_dir / f"frame_{fid:06d}.png"
        cv2.imwrite(str(path), frame)
        saved.append(fid)
    cap.release()

    # Empty annotation table (one placeholder row documents schema)
    gt = pd.DataFrame(columns=IDENTITY_GT_COLUMNS)
    gt_path = out_dir / "identity_gt.csv"
    gt.to_csv(gt_path, index=False)
    manifest = {
        "video_path": str(video_path),
        "video_name": name,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "frame_stride": int(frame_stride),
        "sampled_frames": saved,
        "frame_count": len(saved),
        "duration_seconds": (end_frame - start_frame + 1) / max(fps, 1e-6),
        "fps": fps,
        "status": "GT_INCOMPLETE",
        "identity_gt_csv": str(gt_path),
        "frames_dir": str(frames_dir),
        "note": "Annotate with annotate_player_identity_gt.py; GT player IDs must be stable across the clip.",
    }
    (out_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
