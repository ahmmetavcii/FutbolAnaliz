"""Helpers for video-mode human detection GT annotation.

Overlay detections are display-only and must never be written as GT unless the
user explicitly accepts/saves on a review_sample frame.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from football_analytics.evaluation.human_gt_annotate import (
    BoxState,
    FrameEditState,
    is_true,
    load_proposals_for_frame,
    recount_completion,
    truthy_mask,
)

VIDEO_OVERLAY_COLUMNS = [
    "frame_idx",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "source_detector",
]

# Preferred full-video hybrid cache (750 frames for football.mp4).
DEFAULT_OVERLAY_CANDIDATES = [
    Path("/home/ahmet/workspace/hybrid_detector_validation/artifacts/common_human_detections.parquet"),
    Path("/home/ahmet/workspace/hybrid_detector_validation/runs/track_bytetrack/detections.parquet"),
    Path("/home/ahmet/workspace/hybrid_detector_validation/runs/hybrid/detections.parquet"),
]

DEFAULT_TRACK_CANDIDATES = [
    Path("/home/ahmet/workspace/hybrid_detector_validation/runs/track_bytetrack/local_tracks.parquet"),
    Path("/home/ahmet/workspace/hybrid_detector_validation/runs/track_botsort/local_tracks.parquet"),
]


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m:02d}:{s:05.2f}"


def frame_to_time(frame_idx: int, fps: float) -> float:
    if fps <= 0:
        return 0.0
    return float(frame_idx) / float(fps)


def time_to_frame(seconds: float, fps: float, n_frames: int) -> int:
    if fps <= 0:
        return 0
    return int(np.clip(round(float(seconds) * float(fps)), 0, max(0, n_frames - 1)))


def clamp_frame(frame_idx: int, n_frames: int) -> int:
    return int(np.clip(int(frame_idx), 0, max(0, n_frames - 1)))


def sample_frame_indices(sample: pd.DataFrame) -> list[int]:
    return [int(x) for x in sample["frame_idx"].astype(int).tolist()]


def sample_ordinal(sample: pd.DataFrame, frame_idx: int) -> int | None:
    ids = sample_frame_indices(sample)
    try:
        return ids.index(int(frame_idx)) + 1  # 1-based
    except ValueError:
        return None


def is_sample_frame(sample: pd.DataFrame, frame_idx: int) -> bool:
    return int(frame_idx) in set(sample_frame_indices(sample))


def first_unreviewed_sample_idx(sample: pd.DataFrame) -> int | None:
    """Return absolute video frame_idx of first unreviewed sample, or None."""
    for _, r in sample.iterrows():
        if not is_true(r["reviewed"]):
            return int(r["frame_idx"])
    return None


def next_sample_frame(sample: pd.DataFrame, frame_idx: int, *, direction: int = 1) -> int | None:
    ids = sample_frame_indices(sample)
    if not ids:
        return None
    if direction >= 0:
        for fid in ids:
            if fid > int(frame_idx):
                return fid
        return None
    for fid in reversed(ids):
        if fid < int(frame_idx):
            return fid
    return None


def next_unreviewed_after(sample: pd.DataFrame, frame_idx: int) -> int | None:
    for _, r in sample.iterrows():
        fid = int(r["frame_idx"])
        if fid > int(frame_idx) and not is_true(r["reviewed"]):
            return fid
    return first_unreviewed_sample_idx(sample)


def normalize_overlay_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Map various detection schemas into VIDEO_OVERLAY_COLUMNS."""
    out = df.copy()
    if "frame_idx" not in out.columns and "frame_id" in out.columns:
        out = out.rename(columns={"frame_id": "frame_idx"})
    if "class_name" not in out.columns:
        if "object_type" in out.columns:
            out["class_name"] = out["object_type"]
        elif "mapped" in out.columns:
            out["class_name"] = out["mapped"]
        elif "raw_name" in out.columns:
            out["class_name"] = out["raw_name"]
        else:
            out["class_name"] = "player"
    if "confidence" not in out.columns:
        if "detection_confidence" in out.columns:
            out["confidence"] = out["detection_confidence"]
        elif "conf" in out.columns:
            out["confidence"] = out["conf"]
        else:
            out["confidence"] = 0.0
    for a, b in (("bbox_x1", "x1"), ("bbox_y1", "y1"), ("bbox_x2", "x2"), ("bbox_y2", "y2")):
        if a in out.columns and b not in out.columns:
            out[b] = out[a]
    if "source_detector" not in out.columns:
        if "source_model" in out.columns:
            out["source_detector"] = out["source_model"].astype(str)
        else:
            out["source_detector"] = "overlay"
    # Prefer role as display class when present (goalkeeper role etc.)
    if "role" in out.columns:
        role = out["role"].astype(str)
        mask = role.isin({"player", "goalkeeper", "referee", "person_unresolved"})
        out.loc[mask, "class_name"] = role[mask]
    keep = out[VIDEO_OVERLAY_COLUMNS].copy()
    keep["frame_idx"] = keep["frame_idx"].astype(int)
    keep["confidence"] = keep["confidence"].astype(float)
    for c in ("x1", "y1", "x2", "y2"):
        keep[c] = keep[c].astype(float)
    keep["class_name"] = keep["class_name"].astype(str)
    keep["source_detector"] = keep["source_detector"].astype(str)
    return keep


def find_existing_overlay_source(candidates: Iterable[Path] | None = None) -> Path | None:
    for p in candidates or DEFAULT_OVERLAY_CANDIDATES:
        if Path(p).exists():
            return Path(p)
    return None


def ensure_video_proposals_cache(
    gt_dir: Path,
    *,
    source: Path | None = None,
    candidates: Iterable[Path] | None = None,
) -> Path:
    """Ensure ``video_proposals.parquet`` exists under gt_dir (display-only)."""
    dest = Path(gt_dir) / "video_proposals.parquet"
    if dest.exists():
        return dest
    src = Path(source) if source is not None else find_existing_overlay_source(candidates)
    if src is None:
        # Empty overlay cache — playback still works, no boxes outside samples.
        empty = pd.DataFrame(columns=VIDEO_OVERLAY_COLUMNS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(dest, index=False)
        return dest
    df = normalize_overlay_dataframe(pd.read_parquet(src))
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return dest


def load_overlay_by_frame(path: Path) -> dict[int, pd.DataFrame]:
    if not Path(path).exists():
        return {}
    df = normalize_overlay_dataframe(pd.read_parquet(path))
    out: dict[int, pd.DataFrame] = {}
    for fid, g in df.groupby(df["frame_idx"].astype(int)):
        out[int(fid)] = g.reset_index(drop=True)
    return out


def overlay_rows_for_frame(index: dict[int, pd.DataFrame], frame_idx: int) -> pd.DataFrame:
    g = index.get(int(frame_idx))
    if g is None:
        return pd.DataFrame(columns=VIDEO_OVERLAY_COLUMNS)
    return g


def load_tracks_by_frame(path: Path | None) -> dict[int, list[dict[str, Any]]]:
    """Optional tracking overlay; never written to GT."""
    if path is None or not Path(path).exists():
        return {}
    df = pd.read_parquet(path)
    # tolerate schemas
    frame_col = "frame_idx" if "frame_idx" in df.columns else ("frame_id" if "frame_id" in df.columns else None)
    if frame_col is None:
        return {}
    tid_col = next((c for c in ("track_id", "id", "tracker_id") if c in df.columns), None)
    out: dict[int, list[dict[str, Any]]] = {}
    for _, r in df.iterrows():
        fid = int(r[frame_col])
        item = {
            "track_id": int(r[tid_col]) if tid_col is not None else -1,
            "x1": float(r.get("x1", r.get("bbox_x1", 0))),
            "y1": float(r.get("y1", r.get("bbox_y1", 0))),
            "x2": float(r.get("x2", r.get("bbox_x2", 0))),
            "y2": float(r.get("y2", r.get("bbox_y2", 0))),
            "class_name": str(r.get("class_name", r.get("object_type", "player"))),
        }
        out.setdefault(fid, []).append(item)
    return out


def find_track_source(candidates: Iterable[Path] | None = None) -> Path | None:
    for p in candidates or DEFAULT_TRACK_CANDIDATES:
        if Path(p).exists():
            return Path(p)
    return None


def build_edit_state_for_sample(
    *,
    frame_idx: int,
    proposals: pd.DataFrame,
    gt: pd.DataFrame,
    sample: pd.DataFrame,
) -> FrameEditState:
    """Load annotation edit state for a sample frame (proposals or saved GT)."""
    boxes = load_proposals_for_frame(proposals, frame_idx)
    reviewed = False
    row = sample.loc[sample["frame_idx"].astype(int) == int(frame_idx)]
    if not row.empty:
        reviewed = is_true(row.iloc[0]["reviewed"])
    if reviewed:
        g = gt[
            (gt["frame_idx"].astype(int) == int(frame_idx))
            & truthy_mask(gt["reviewed"])
            & gt["x1"].notna()
        ]
        saved: list[BoxState] = []
        for _, r in g.iterrows():
            saved.append(
                BoxState(
                    proposal_id=str(r["gt_id"]),
                    class_name=str(r["class_name"]),
                    confidence=1.0,
                    x1=float(r["x1"]),
                    y1=float(r["y1"]),
                    x2=float(r["x2"]),
                    y2=float(r["y2"]),
                    source_detector="saved_gt",
                    accepted=True,
                    rejected=False,
                    modified=False,
                    manual=str(r.get("notes", "")).startswith("manual"),
                    occluded=is_true(r.get("occluded", False)),
                    difficult=is_true(r.get("difficult", False)),
                )
            )
        if saved:
            boxes = saved
        # else keep proposals so empty reviewed can be overwritten via R→A→S
    return FrameEditState(frame_idx=int(frame_idx), boxes=boxes, dirty=False)


def gt_only_contains_sample_frames(gt: pd.DataFrame, sample: pd.DataFrame) -> bool:
    if gt is None or gt.empty:
        return True
    sample_ids = set(sample_frame_indices(sample))
    return set(gt["frame_idx"].astype(int)) <= sample_ids


def resume_start_frame(
    sample: pd.DataFrame,
    *,
    mode: str,
    fps: float,
    n_frames: int,
    explicit_frame: int | None = None,
    lead_seconds: float = 2.0,
) -> int:
    """Compute initial frame for resume modes.

    mode: start | first_unreviewed | explicit | review_only
    """
    if mode == "start":
        return 0
    if mode == "explicit" and explicit_frame is not None:
        return clamp_frame(explicit_frame, n_frames)
    target = first_unreviewed_sample_idx(sample)
    if target is None:
        ids = sample_frame_indices(sample)
        target = ids[0] if ids else 0
    lead = int(round(float(lead_seconds) * float(fps))) if fps > 0 else 0
    return clamp_frame(int(target) - lead, n_frames)


@dataclass
class AutoPauseController:
    """Tracks auto-pause when crossing into sample frames during playback."""

    sample_ids: set[int]
    paused_for_review: bool = False
    last_auto_paused_frame: int | None = None

    def on_frame(self, frame_idx: int, *, playing: bool) -> bool:
        """Return True if playback should auto-pause for GT review."""
        fid = int(frame_idx)
        if fid not in self.sample_ids:
            self.paused_for_review = False
            return False
        if not playing and self.last_auto_paused_frame == fid:
            self.paused_for_review = True
            return True
        # Auto-pause when arriving at a sample frame while playing, or first land
        if self.last_auto_paused_frame != fid:
            self.last_auto_paused_frame = fid
            self.paused_for_review = True
            return True
        return self.paused_for_review

    def clear_after_continue(self) -> None:
        self.paused_for_review = False


@dataclass
class VideoHudState:
    frame_idx: int
    n_frames: int
    fps: float
    playing: bool
    sample: pd.DataFrame
    show_overlay: bool = True
    show_tracks: bool = False

    def lines(self) -> list[str]:
        t = format_timestamp(frame_to_time(self.frame_idx, self.fps))
        total = format_timestamp(frame_to_time(max(0, self.n_frames - 1), self.fps))
        comp = recount_completion(self.sample)
        nxt = next_sample_frame(self.sample, self.frame_idx, direction=1)
        ord_ = sample_ordinal(self.sample, self.frame_idx)
        lines = [
            f"{t} / {total}",
            f"Frame {self.frame_idx}",
            f"FPS {self.fps:.2f}",
            "PLAYING" if self.playing else "PAUSED",
            f"Completion: {comp['reviewed']}/{comp['total']}",
        ]
        if ord_ is not None:
            lines.append(f"Current sample: {ord_}/{comp['total']}")
        if nxt is not None:
            lines.append(f"Next GT frame: {nxt} ({format_timestamp(frame_to_time(nxt, self.fps))})")
        else:
            lines.append("Next GT frame: —")
        return lines


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file_fingerprint(path: Path) -> tuple[int, str]:
    p = Path(path)
    if not p.exists():
        return (0, "")
    return (p.stat().st_size, sha256_file(p))
