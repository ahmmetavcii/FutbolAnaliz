"""Human-detection GT annotation helpers (proposals ≠ GT)."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROPOSAL_COLUMNS = [
    "frame_idx",
    "proposal_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "source_detector",
    "accepted",
    "rejected",
    "modified",
]

GT_COLUMNS = [
    "frame_idx",
    "gt_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "occluded",
    "difficult",
    "reviewed",
    "notes",
]

CLASSES = [
    "player",
    "goalkeeper",
    "referee",
    "ignore_person",
    "not_target",
    "person_unresolved",
]


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def truthy_mask(series: pd.Series) -> pd.Series:
    return series.map(is_true)


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def atomic_write_csv(df: pd.DataFrame, path: Path, *, backup_dir: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup_dir is not None and path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy2(path, backup_dir / f"{path.name}.{stamp}.bak")
    tmp = path.with_suffix(path.suffix + f".tmp.{int(time.time() * 1000)}")
    df.to_csv(tmp, index=False)
    check = pd.read_csv(tmp)
    if len(check) != len(df):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("atomic write verification failed: row count mismatch")
    tmp.replace(path)


def recount_completion(sample: pd.DataFrame) -> dict[str, int]:
    reviewed = int(truthy_mask(sample["reviewed"]).sum())
    return {"reviewed": reviewed, "total": int(len(sample))}


@dataclass
class BoxState:
    proposal_id: str
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    source_detector: str = "manual"
    accepted: bool = False
    rejected: bool = False
    modified: bool = False
    manual: bool = False
    conflict: bool = False
    duplicate_warn: bool = False
    occluded: bool = False
    difficult: bool = False

    def as_proposal_row(self, frame_idx: int) -> dict[str, Any]:
        return {
            "frame_idx": int(frame_idx),
            "proposal_id": self.proposal_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "x1": float(self.x1),
            "y1": float(self.y1),
            "x2": float(self.x2),
            "y2": float(self.y2),
            "source_detector": self.source_detector,
            "accepted": bool(self.accepted),
            "rejected": bool(self.rejected),
            "modified": bool(self.modified),
        }

    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class FrameEditState:
    frame_idx: int
    boxes: list[BoxState] = field(default_factory=list)
    dirty: bool = False

    def visible_proposals(self) -> list[BoxState]:
        return [b for b in self.boxes if not b.rejected and not b.manual]

    def accepted_boxes(self) -> list[BoxState]:
        return [b for b in self.boxes if b.accepted and not b.rejected]

    def manual_boxes(self) -> list[BoxState]:
        return [b for b in self.boxes if b.manual and not b.rejected]

    def counts(self) -> dict[str, int]:
        return {
            "proposals": len([b for b in self.boxes if not b.manual]),
            "accepted": len(self.accepted_boxes()),
            "rejected": len([b for b in self.boxes if b.rejected]),
            "manual": len(self.manual_boxes()),
        }


def load_proposals_for_frame(proposals: pd.DataFrame, frame_idx: int) -> list[BoxState]:
    if proposals is None or proposals.empty:
        return []
    g = proposals[proposals["frame_idx"].astype(int) == int(frame_idx)]
    out: list[BoxState] = []
    for _, r in g.iterrows():
        out.append(
            BoxState(
                proposal_id=str(r["proposal_id"]),
                class_name=str(r["class_name"]),
                confidence=float(r.get("confidence", 0.0) or 0.0),
                x1=float(r["x1"]),
                y1=float(r["y1"]),
                x2=float(r["x2"]),
                y2=float(r["y2"]),
                source_detector=str(r.get("source_detector", "hybrid")),
                accepted=is_true(r.get("accepted", False)),
                rejected=is_true(r.get("rejected", False)),
                modified=is_true(r.get("modified", False)),
                manual=False,
            )
        )
    return out


def accept_visible_proposals(state: FrameEditState) -> None:
    """A-key: temporarily accept visible proposals. Does NOT set reviewed/save."""
    for box in state.boxes:
        if box.rejected or box.manual:
            continue
        box.accepted = True
    state.dirty = True


def save_guard(state: FrameEditState) -> str:
    """Decide whether a normal S-save is allowed.

    Returns:
      - ``ok``: accepted and/or manual boxes exist
      - ``need_accept``: proposals exist but none accepted (and no manual)
      - ``need_empty_confirm``: genuinely empty — requires double-E, not S
    """
    counts = state.counts()
    if counts["accepted"] > 0 or counts["manual"] > 0:
        return "ok"
    # Unaccepted (or only-rejected) proposals still on the frame
    has_visible_unaccepted = any(
        (not b.manual and not b.rejected and not b.accepted) for b in state.boxes
    )
    has_any_proposal = counts["proposals"] > 0
    if has_visible_unaccepted or (has_any_proposal and counts["accepted"] == 0):
        return "need_accept"
    return "need_empty_confirm"


def state_snapshot(state: FrameEditState) -> tuple:
    """Hashable snapshot used to clear dirty after successful save."""
    rows = []
    for b in state.boxes:
        rows.append(
            (
                b.proposal_id,
                b.class_name,
                round(b.x1, 2),
                round(b.y1, 2),
                round(b.x2, 2),
                round(b.y2, 2),
                b.accepted,
                b.rejected,
                b.modified,
                b.manual,
            )
        )
    return (int(state.frame_idx), tuple(rows))


def reset_frame_from_proposals(state: FrameEditState, proposals: pd.DataFrame) -> FrameEditState:
    fresh = FrameEditState(frame_idx=state.frame_idx, boxes=load_proposals_for_frame(proposals, state.frame_idx))
    fresh.dirty = True
    return fresh


def mark_duplicate_warnings(state: FrameEditState, iou_thr: float = 0.90) -> None:
    for b in state.boxes:
        b.duplicate_warn = False
    active = [b for b in state.boxes if not b.rejected]
    for i, a in enumerate(active):
        for b in active[i + 1 :]:
            if a.class_name != b.class_name:
                continue
            if iou_xyxy(a.xyxy(), b.xyxy()) > iou_thr:
                a.duplicate_warn = True
                b.duplicate_warn = True


def mark_player_referee_conflicts(state: FrameEditState, iou_thr: float = 0.55) -> None:
    for b in state.boxes:
        b.conflict = False
    active = [b for b in state.boxes if not b.rejected]
    for i, a in enumerate(active):
        for b in active[i + 1 :]:
            classes = {a.class_name, b.class_name}
            if classes != {"player", "referee"}:
                continue
            if iou_xyxy(a.xyxy(), b.xyxy()) >= iou_thr:
                a.conflict = True
                b.conflict = True


def build_gt_rows_for_frame(state: FrameEditState) -> list[dict[str, Any]]:
    """Convert accepted proposals + manual boxes into GT rows (reviewed handled by caller)."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for box in state.boxes:
        if box.rejected:
            continue
        if not box.accepted and not box.manual:
            continue
        key = (
            box.class_name,
            round(box.x1, 1),
            round(box.y1, 1),
            round(box.x2, 1),
            round(box.y2, 1),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "frame_idx": int(state.frame_idx),
                "gt_id": box.proposal_id if not box.manual else f"manual_{box.proposal_id}",
                "class_name": box.class_name,
                "x1": float(box.x1),
                "y1": float(box.y1),
                "x2": float(box.x2),
                "y2": float(box.y2),
                "occluded": bool(box.occluded),
                "difficult": bool(box.difficult),
                "reviewed": True,
                "notes": "from_proposal" if not box.manual else "manual",
            }
        )
    return rows


def merge_gt_preserving_other_reviewed(
    gt: pd.DataFrame,
    frame_idx: int,
    new_rows: list[dict[str, Any]],
    sample_frame_ids: set[int],
) -> pd.DataFrame:
    """Replace only this frame's rows; keep other reviewed frames intact.

    Rows outside the 40-frame sample are kept but should not be treated as
    evaluator inputs by callers that filter on review_sample.
    """
    keep = gt[gt["frame_idx"].astype(int) != int(frame_idx)].copy()
    # Drop placeholders for this frame
    if new_rows:
        add = pd.DataFrame(new_rows)
    else:
        add = pd.DataFrame(
            [
                {
                    "frame_idx": int(frame_idx),
                    "gt_id": f"f{frame_idx}_empty",
                    "class_name": "not_target",
                    "x1": np.nan,
                    "y1": np.nan,
                    "x2": np.nan,
                    "y2": np.nan,
                    "occluded": False,
                    "difficult": False,
                    "reviewed": True,
                    "notes": "explicit_empty_frame",
                }
            ]
        )
    out = pd.concat([keep, add], ignore_index=True)
    # Ensure columns
    for col in GT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan if col.startswith("x") or col.startswith("y") else False
    return out[GT_COLUMNS]


def evaluator_gt_frames(gt: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Only reviewed boxes on sample frames are evaluator-eligible."""
    sample_ids = set(sample["frame_idx"].astype(int))
    reviewed_sample = set(sample.loc[truthy_mask(sample["reviewed"]), "frame_idx"].astype(int))
    g = gt[gt["frame_idx"].astype(int).isin(sample_ids)].copy()
    g = g[truthy_mask(g["reviewed"])]
    g = g[g["frame_idx"].astype(int).isin(reviewed_sample)]
    # drop empty placeholders
    if "x1" in g.columns:
        g = g[g["x1"].notna()]
    return g


def proposals_are_not_gt(proposals: pd.DataFrame, gt: pd.DataFrame) -> bool:
    """Sanity: proposal file must not be written into GT automatically."""
    if proposals is None or proposals.empty:
        return True
    if gt is None or gt.empty:
        return True
    # If GT only has placeholders / unreviewed, proposals weren't auto-accepted
    real = gt[truthy_mask(gt["reviewed"]) & gt["x1"].notna()] if "x1" in gt.columns else gt.iloc[0:0]
    if real.empty:
        return True
    # Proposals themselves always have accepted/rejected columns and are a separate file
    return "proposal_id" not in gt.columns
