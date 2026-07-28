#!/usr/bin/env python3
"""Annotate human detection GT with hybrid detector proposals as suggestions.

Proposals are NEVER auto-written as ground truth. Only S (save) commits GT
with reviewed=true after human confirmation.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_analytics.evaluation.human_gt_annotate import (  # noqa: E402
    CLASSES,
    BoxState,
    FrameEditState,
    accept_visible_proposals,
    atomic_write_csv,
    build_gt_rows_for_frame,
    evaluator_gt_frames,
    is_true,
    load_proposals_for_frame,
    mark_duplicate_warnings,
    mark_player_referee_conflicts,
    merge_gt_preserving_other_reviewed,
    recount_completion,
    reset_frame_from_proposals,
    save_guard,
    state_snapshot,
    truthy_mask,
)

DEFAULT_GT_DIR = ROOT / "configs/evaluation/human_detection_gt/football"

# BGR colors per spec
COLORS = {
    "player": (255, 0, 0),  # blue
    "referee": (0, 255, 255),  # yellow
    "goalkeeper": (0, 255, 0),  # green
    "person_unresolved": (0, 165, 255),  # orange
    "ignore_person": (128, 128, 128),  # gray
    "not_target": (60, 60, 60),
}

HELP = """
1 player | 2 goalkeeper | 3 referee | 4 ignore_person | 5 not_target | U unresolved
Click=select | drag corner=resize | drag empty=new box | X/Del=reject
A=accept visible proposals (NOT reviewed) | R=reset proposals | M=toggle proposals
S=save accepted+manual as reviewed | E+E=confirm empty frame
N/P=next/prev | Q/ESC=quit
"""

NO_ACCEPTED_MSG = (
    "NO ACCEPTED BOXES.\n"
    "Press A to accept proposals, edit them, then press S.\n"
    "Press E only if this frame is genuinely empty."
)


class Annotator:
    def __init__(self, gt_dir: Path) -> None:
        self.gt_dir = gt_dir
        self.backup_dir = gt_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.sample = pd.read_csv(gt_dir / "review_sample.csv")
        self.gt = pd.read_csv(gt_dir / "human_detection_gt.csv")
        prop_path = gt_dir / "model_proposals.csv"
        if not prop_path.exists():
            raise SystemExit(
                f"Missing {prop_path}. Run scripts/generate_human_detection_proposals.py first."
            )
        self.proposals = pd.read_csv(prop_path)
        self.frames_dir = gt_dir / "frames"
        self.show_proposals = True
        self.class_i = 0
        self.selected = 0
        self.drawing = False
        self.resizing = False
        self.resize_corner: int | None = None
        self.pt0 = (0, 0)
        self.cur_box: tuple[int, int, int, int] | None = None
        self.empty_confirmation_pending = False
        self.status_msg = ""
        self.saved_snapshot = None
        self.idx = 0
        for i, r in self.sample.iterrows():
            if not is_true(r["reviewed"]):
                self.idx = int(i)
                break
        self.state = FrameEditState(frame_idx=0)
        self._load_frame_state()

    def unsaved_changes(self) -> bool:
        """True when edits diverge from last clean/saved snapshot."""
        if self.saved_snapshot is None:
            return bool(self.state.dirty)
        if state_snapshot(self.state) != self.saved_snapshot:
            return True
        return bool(self.state.dirty)

    def frame_idx(self) -> int:
        return int(self.sample.iloc[self.idx]["frame_idx"])

    def frame_reviewed_on_disk(self) -> bool:
        return bool(is_true(self.sample.iloc[self.idx]["reviewed"]))

    def _mark_clean(self) -> None:
        self.state.dirty = False
        self.empty_confirmation_pending = False
        self.saved_snapshot = state_snapshot(self.state)
        self.status_msg = ""

    def _load_frame_state(self) -> None:
        fid = self.frame_idx()
        boxes = load_proposals_for_frame(self.proposals, fid)
        # If frame already reviewed with real boxes, show saved GT as editable accepted boxes
        if self.frame_reviewed_on_disk():
            g = self.gt[
                (self.gt["frame_idx"].astype(int) == fid)
                & truthy_mask(self.gt["reviewed"])
                & self.gt["x1"].notna()
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
            # else: empty reviewed frame — keep model proposals so user can R→A→S overwrite
        self.state = FrameEditState(frame_idx=fid, boxes=boxes, dirty=False)
        mark_duplicate_warnings(self.state)
        mark_player_referee_conflicts(self.state)
        self.selected = 0
        self.cur_box = None
        self.empty_confirmation_pending = False
        self.saved_snapshot = state_snapshot(self.state)
        self.status_msg = ""

    def _hit_box(self, x: int, y: int) -> int | None:
        for i, b in enumerate(self.state.boxes):
            if b.rejected:
                continue
            if not self.show_proposals and not b.manual and not b.accepted:
                continue
            if b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2:
                return i
        return None

    def _hit_corner(self, x: int, y: int, thr: int = 12) -> tuple[int, int] | None:
        """Return (box_index, corner_id) corner_id: 0=TL 1=TR 2=BR 3=BL."""
        for i, b in enumerate(self.state.boxes):
            if b.rejected:
                continue
            corners = [
                (b.x1, b.y1),
                (b.x2, b.y1),
                (b.x2, b.y2),
                (b.x1, b.y2),
            ]
            for c, (cx, cy) in enumerate(corners):
                if abs(x - cx) <= thr and abs(y - cy) <= thr:
                    return i, c
        return None

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            corner = self._hit_corner(x, y)
            if corner is not None:
                self.selected, self.resize_corner = corner
                self.resizing = True
                return
            hit = self._hit_box(x, y)
            if hit is not None:
                self.selected = hit
                return
            self.drawing = True
            self.pt0 = (x, y)
            self.cur_box = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.resizing and self.resize_corner is not None:
                b = self.state.boxes[self.selected]
                if self.resize_corner == 0:
                    b.x1, b.y1 = float(x), float(y)
                elif self.resize_corner == 1:
                    b.x2, b.y1 = float(x), float(y)
                elif self.resize_corner == 2:
                    b.x2, b.y2 = float(x), float(y)
                else:
                    b.x1, b.y2 = float(x), float(y)
                # normalize
                x1, x2 = sorted([b.x1, b.x2])
                y1, y2 = sorted([b.y1, b.y2])
                b.x1, b.y1, b.x2, b.y2 = x1, y1, x2, y2
                b.modified = True
                if not b.manual:
                    b.accepted = True
                self.state.dirty = True
            elif self.drawing:
                self.cur_box = (self.pt0[0], self.pt0[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.resizing:
                self.resizing = False
                self.resize_corner = None
                mark_duplicate_warnings(self.state)
                mark_player_referee_conflicts(self.state)
                return
            if not self.drawing:
                return
            self.drawing = False
            x1, y1 = self.pt0
            x2, y2 = x, y
            self.cur_box = None
            if abs(x2 - x1) < 4 or abs(y2 - y1) < 4:
                return
            box = BoxState(
                proposal_id=f"manual_{self.frame_idx()}_{int(time.time() * 1000) % 100000}",
                class_name=CLASSES[self.class_i],
                confidence=1.0,
                x1=float(min(x1, x2)),
                y1=float(min(y1, y2)),
                x2=float(max(x1, x2)),
                y2=float(max(y1, y2)),
                source_detector="manual",
                accepted=True,
                rejected=False,
                modified=True,
                manual=True,
            )
            self.state.boxes.append(box)
            self.selected = len(self.state.boxes) - 1
            self.state.dirty = True
            mark_duplicate_warnings(self.state)
            mark_player_referee_conflicts(self.state)

    def render(self) -> np.ndarray:
        row = self.sample.iloc[self.idx]
        path = self.frames_dir / str(row["frame_file"])
        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, f"missing {path.name}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        proposals_visible = [b for b in self.state.boxes if not b.manual and not b.rejected]
        if self.show_proposals and not proposals_visible and not any(b.manual for b in self.state.boxes):
            cv2.putText(
                img,
                "NO MODEL PROPOSALS — DRAW BOXES MANUALLY",
                (40, img.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )

        for i, b in enumerate(self.state.boxes):
            if b.rejected:
                continue
            if not self.show_proposals and not b.manual and not b.accepted:
                continue
            color = COLORS.get(b.class_name, (255, 255, 255))
            if b.conflict:
                color = (0, 165, 255)  # orange conflict warning — do not auto-delete
            thick = 3 if i == self.selected else 2
            x1, y1, x2, y2 = map(int, (b.x1, b.y1, b.x2, b.y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
            if b.duplicate_warn:
                cv2.putText(img, "DUP?", (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            if b.conflict:
                cv2.putText(img, "CONFLICT", (x1, y2 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            label = f"{b.class_name} {b.confidence:.2f} {b.proposal_id}"
            if b.accepted:
                label = "[A] " + label
            if b.manual:
                label = "[M] " + label
            cv2.putText(img, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            # corner handles when selected
            if i == self.selected:
                for cx, cy in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
                    cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)

        if self.cur_box is not None:
            x1, y1, x2, y2 = self.cur_box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        c = self.state.counts()
        comp = recount_completion(self.sample)
        saved_yes = self.frame_reviewed_on_disk() and not self.unsaved_changes()
        hud = [
            f"Frame: {self.idx + 1}/40  class={CLASSES[self.class_i]}",
            f"Proposals: {c['proposals']}  Accepted: {c['accepted']}  Rejected: {c['rejected']}  Manual boxes: {c['manual']}",
            f"Saved/reviewed: {'Yes' if saved_yes else 'No'}",
            f"Completion: {comp['reviewed']}/{comp['total']}  proposals_visible={self.show_proposals}  dirty={self.state.dirty}",
            "1-5/U class | A accept | R reset | M toggle | S save | E empty | N/P nav | X reject | Q quit",
        ]
        if self.status_msg:
            hud.append(self.status_msg)
        if self.empty_confirmation_pending:
            hud.append("EMPTY FRAME? PRESS E AGAIN TO CONFIRM")
        y = 22
        for line in hud:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 22
        return img

    def save_current_frame(self, *, allow_empty: bool = False) -> bool:
        """Save accepted+manual boxes as reviewed GT. Returns True on success.

        When proposals exist but none are accepted, blocks unless allow_empty
        (only via confirmed double-E).
        """
        fid = self.frame_idx()
        guard = save_guard(self.state)
        if guard == "need_accept" and not allow_empty:
            self.status_msg = NO_ACCEPTED_MSG
            print(self.status_msg, flush=True)
            return False
        if guard == "need_empty_confirm" and not allow_empty:
            self.status_msg = NO_ACCEPTED_MSG
            print(self.status_msg, flush=True)
            return False

        was_reviewed = self.frame_reviewed_on_disk()

        # Persist proposal accept/reject/modified flags for this frame
        prop = self.proposals.copy()
        prop = prop[prop["frame_idx"].astype(int) != fid]
        prop_rows = [b.as_proposal_row(fid) for b in self.state.boxes if not b.manual]
        if prop_rows:
            prop = pd.concat([prop, pd.DataFrame(prop_rows)], ignore_index=True)
        atomic_write_csv(prop, self.gt_dir / "model_proposals.csv", backup_dir=self.backup_dir)
        self.proposals = pd.read_csv(self.gt_dir / "model_proposals.csv")

        new_rows = build_gt_rows_for_frame(self.state)
        if allow_empty and not new_rows:
            # explicit empty confirmation
            new_rows = []

        sample_ids = set(self.sample["frame_idx"].astype(int))
        new_gt = merge_gt_preserving_other_reviewed(self.gt, fid, new_rows, sample_ids)
        atomic_write_csv(new_gt, self.gt_dir / "human_detection_gt.csv", backup_dir=self.backup_dir)
        self.gt = pd.read_csv(self.gt_dir / "human_detection_gt.csv")

        self.sample.loc[self.sample["frame_idx"].astype(int) == fid, "reviewed"] = True
        atomic_write_csv(self.sample, self.gt_dir / "review_sample.csv", backup_dir=self.backup_dir)
        self.sample = pd.read_csv(self.gt_dir / "review_sample.csv")

        if not bool(
            truthy_mask(self.sample.loc[self.sample["frame_idx"].astype(int) == fid, "reviewed"]).all()
        ):
            raise RuntimeError("save verify failed: reviewed flag")

        self._mark_clean()
        # Reload state from disk so accepted GT boxes are shown; dirty stays false
        self._load_frame_state()
        comp = recount_completion(self.sample)
        n_boxes = int(self.gt[(self.gt.frame_idx.astype(int) == fid) & self.gt.x1.notna()].shape[0])
        print(
            f"saved frame {fid}: gt_boxes={n_boxes} completion={comp['reviewed']}/{comp['total']} dirty=false"
            + (" (overwrite)" if was_reviewed else ""),
            flush=True,
        )
        return True

    def handle_empty_confirm(self) -> None:
        """Double-E confirmation for genuinely empty frames (even if proposals are visible)."""
        guard = save_guard(self.state)
        if guard == "ok":
            self.status_msg = "Frame has accepted/manual boxes — use S to save."
            self.empty_confirmation_pending = False
            print(self.status_msg, flush=True)
            return
        # need_accept or need_empty_confirm: allow empty via double-E
        if not self.empty_confirmation_pending:
            self.empty_confirmation_pending = True
            self.status_msg = "EMPTY FRAME? PRESS E AGAIN TO CONFIRM"
            print(self.status_msg, flush=True)
            return
        # second E
        self.empty_confirmation_pending = False
        self.save_current_frame(allow_empty=True)

    def reject_selected(self) -> None:
        if not self.state.boxes:
            return
        b = self.state.boxes[self.selected]
        b.rejected = True
        b.accepted = False
        self.state.dirty = True
        self.empty_confirmation_pending = False
        self.selected = max(0, min(self.selected, len(self.state.boxes) - 1))

    def set_class(self, name: str) -> None:
        self.class_i = CLASSES.index(name) if name in CLASSES else self.class_i
        if self.state.boxes:
            b = self.state.boxes[self.selected]
            if not b.rejected:
                b.class_name = name
                b.modified = True
                if not b.manual:
                    b.accepted = True
                self.state.dirty = True
                self.empty_confirmation_pending = False
                mark_duplicate_warnings(self.state)
                mark_player_referee_conflicts(self.state)

    def run(self) -> None:
        print(HELP)
        win = "human_detection_gt"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self.on_mouse)
        while True:
            cv2.imshow(win, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                if self.unsaved_changes():
                    print("unsaved changes on quit — not auto-saving empty; press S/E explicitly next time")
                break
            if key == ord("1"):
                self.set_class("player")
            elif key == ord("2"):
                self.set_class("goalkeeper")
            elif key == ord("3"):
                self.set_class("referee")
            elif key == ord("4"):
                self.set_class("ignore_person")
            elif key == ord("5"):
                self.set_class("not_target")
            elif key in (ord("u"), ord("U")):
                self.set_class("person_unresolved")
            elif key in (ord("a"), ord("A")):
                accept_visible_proposals(self.state)
                self.empty_confirmation_pending = False
                self.status_msg = "A: accepted visible proposals (reviewed NOT set)"
                print(self.status_msg, flush=True)
            elif key in (ord("r"), ord("R")):
                self.state = reset_frame_from_proposals(self.state, self.proposals)
                mark_duplicate_warnings(self.state)
                mark_player_referee_conflicts(self.state)
                self.selected = 0
                self.empty_confirmation_pending = False
                self.status_msg = "R: reset proposals from model_proposals.csv"
                print(self.status_msg, flush=True)
            elif key in (ord("m"), ord("M")):
                self.show_proposals = not self.show_proposals
            elif key in (ord("s"), ord("S")):
                self.empty_confirmation_pending = False
                self.save_current_frame(allow_empty=False)
            elif key in (ord("e"), ord("E")):
                self.handle_empty_confirm()
            elif key in (ord("n"), ord("N")):
                if self.unsaved_changes():
                    print("unsaved — press S to save or R to reset before N", flush=True)
                else:
                    self.idx = min(len(self.sample) - 1, self.idx + 1)
                    self._load_frame_state()
            elif key in (ord("p"), ord("P")):
                if self.unsaved_changes():
                    print("unsaved — press S to save or R to reset before P", flush=True)
                else:
                    self.idx = max(0, self.idx - 1)
                    self._load_frame_state()
            elif key in (ord("x"), ord("X"), 8, 255):
                self.reject_selected()
        cv2.destroyAllWindows()
        comp = recount_completion(self.sample)
        eligible = evaluator_gt_frames(self.gt, self.sample)
        print(f"real_GT_reviewed={comp['reviewed']}/40")
        print(f"evaluator_eligible_boxes={len(eligible)}")
        print("automatic_GT_acceptance=0")


def main() -> None:
    ap = argparse.ArgumentParser(description="Human detection GT annotator with hybrid proposals")
    ap.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    ap.add_argument(
        "--generate-proposals",
        action="store_true",
        help="Run proposal generation first if model_proposals.csv missing",
    )
    args = ap.parse_args()
    prop = args.gt_dir / "model_proposals.csv"
    if args.generate_proposals or not prop.exists():
        from subprocess import check_call

        check_call(
            [
                sys.executable,
                str(ROOT / "scripts/generate_human_detection_proposals.py"),
                "--gt-dir",
                str(args.gt_dir),
            ]
        )
    Annotator(args.gt_dir).run()


if __name__ == "__main__":
    main()
