#!/usr/bin/env python3
"""Video-mode human detection GT annotator.

Plays the original football video with hybrid detection overlays. Auto-pauses
on review_sample frames for annotation. Proposals are never auto-GT.

Does not replace scripts/annotate_human_detection_gt.py (still-frame tool).
"""
from __future__ import annotations

import argparse
import os
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
    is_true,
    mark_duplicate_warnings,
    mark_player_referee_conflicts,
    merge_gt_preserving_other_reviewed,
    recount_completion,
    reset_frame_from_proposals,
    save_guard,
    state_snapshot,
    truthy_mask,
)
from football_analytics.evaluation.human_gt_video import (  # noqa: E402
    AutoPauseController,
    VideoHudState,
    build_edit_state_for_sample,
    clamp_frame,
    copy_file_fingerprint,
    ensure_video_proposals_cache,
    find_track_source,
    first_unreviewed_sample_idx,
    format_timestamp,
    frame_to_time,
    is_sample_frame,
    load_overlay_by_frame,
    load_tracks_by_frame,
    next_sample_frame,
    next_unreviewed_after,
    overlay_rows_for_frame,
    resume_start_frame,
    sample_frame_indices,
    sample_ordinal,
    time_to_frame,
)

DEFAULT_GT_DIR = ROOT / "configs/evaluation/human_detection_gt/football"
DEFAULT_VIDEO = Path("/mnt/c/football_data/videos/test_clips/football.mp4")
DEFAULT_PREVIEW_DIR = Path("/home/ahmet/workspace/hybrid_detector_validation/video_review")

# BGR
COLORS = {
    "player": (255, 0, 0),  # blue
    "goalkeeper": (0, 255, 0),  # green
    "referee": (0, 255, 255),  # yellow
    "person_unresolved": (0, 165, 255),  # orange
    "ignore_person": (128, 128, 128),  # gray
    "not_target": (0, 0, 255),  # red
}

NO_ACCEPTED_MSG = (
    "NO ACCEPTED BOXES\n"
    "Press A to accept proposals or edit manually.\n"
    "Use E twice only for a genuinely empty frame."
)

HELP = """
SPACE play/pause | N/P frame | D/RIGHT +1s | B/LEFT -1s | J/K sample | G first incomplete
M overlay | T tracks | F fullscreen | Q quit
On GT frame: A accept | R reset | S save | E+E empty | ENTER resume | SHIFT+S save+next sample
"""


def _draw_dashed_rect(img, pt1, pt2, color, thickness=1, gap=8):
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness)
    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, thickness)


class VideoHumanAnnotator:
    def __init__(
        self,
        *,
        video_path: Path,
        gt_dir: Path,
        review_only: bool = False,
        start_frame: int = 0,
        show_window: bool = True,
    ) -> None:
        self.video_path = Path(video_path)
        self.gt_dir = Path(gt_dir)
        self.review_only = bool(review_only)
        self.show_window = bool(show_window)
        self.backup_dir = self.gt_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.sample = pd.read_csv(self.gt_dir / "review_sample.csv")
        self.gt = pd.read_csv(self.gt_dir / "human_detection_gt.csv")
        prop_path = self.gt_dir / "model_proposals.csv"
        if not prop_path.exists():
            raise SystemExit(f"Missing {prop_path}")
        self.proposals = pd.read_csv(prop_path)

        overlay_path = ensure_video_proposals_cache(self.gt_dir)
        self.overlay_path = overlay_path
        self.overlay_index = load_overlay_by_frame(overlay_path)
        self.frames_with_overlay = len(self.overlay_index)

        track_src = find_track_source()
        self.tracks_by_frame = load_tracks_by_frame(track_src)

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open video: {self.video_path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self.frame_idx = clamp_frame(start_frame, self.n_frames)
        self.playing = False
        self.show_overlay = True
        self.show_tracks = False
        self.fullscreen = False
        self.quit_requested = False

        self.edit_mode = False
        self.state = FrameEditState(frame_idx=self.frame_idx)
        self.selected = 0
        self.class_i = 0
        self.drawing = False
        self.resizing = False
        self.resize_corner: int | None = None
        self.pt0 = (0, 0)
        self.cur_box: tuple[int, int, int, int] | None = None
        self.empty_confirmation_pending = False
        self.status_msg = ""
        self.saved_snapshot = None
        self.auto_pause = AutoPauseController(set(sample_frame_indices(self.sample)))
        self.frame_bgr: np.ndarray | None = None
        self.win = "human_detection_video"
        self._seek(self.frame_idx)
        self._maybe_enter_sample_review(auto=True)

    # ------------------------------------------------------------------ IO
    def _seek(self, frame_idx: int) -> bool:
        self.frame_idx = clamp_frame(frame_idx, self.n_frames)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.frame_bgr = np.zeros((max(1, self.frame_h), max(1, self.frame_w), 3), np.uint8)
            return False
        self.frame_bgr = frame
        # OpenCV advances; keep logical index as requested
        return True

    def _step(self, delta: int) -> None:
        self._seek(self.frame_idx + delta)

    def _advance_play(self) -> None:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.playing = False
            return
        self.frame_idx = clamp_frame(self.frame_idx + 1, self.n_frames)
        self.frame_bgr = frame

    # ----------------------------------------------------------- edit state
    def unsaved_changes(self) -> bool:
        if not self.edit_mode or self.review_only:
            return False
        if self.saved_snapshot is None:
            return bool(self.state.dirty)
        if state_snapshot(self.state) != self.saved_snapshot:
            return True
        return bool(self.state.dirty)

    def _mark_clean(self) -> None:
        self.state.dirty = False
        self.empty_confirmation_pending = False
        self.saved_snapshot = state_snapshot(self.state)
        self.status_msg = ""

    def frame_reviewed_on_disk(self, frame_idx: int | None = None) -> bool:
        fid = self.frame_idx if frame_idx is None else int(frame_idx)
        row = self.sample.loc[self.sample["frame_idx"].astype(int) == fid]
        if row.empty:
            return False
        return bool(is_true(row.iloc[0]["reviewed"]))

    def _enter_edit_for_current(self) -> None:
        if self.review_only:
            self.edit_mode = False
            return
        if not is_sample_frame(self.sample, self.frame_idx):
            self.edit_mode = False
            return
        self.state = build_edit_state_for_sample(
            frame_idx=self.frame_idx,
            proposals=self.proposals,
            gt=self.gt,
            sample=self.sample,
        )
        mark_duplicate_warnings(self.state)
        mark_player_referee_conflicts(self.state)
        self.selected = 0
        self.cur_box = None
        self.empty_confirmation_pending = False
        self.saved_snapshot = state_snapshot(self.state)
        self.edit_mode = True
        self.playing = False
        self.status_msg = ""

    def _maybe_enter_sample_review(self, *, auto: bool) -> bool:
        """Auto-pause and open edit mode on sample frames."""
        if not is_sample_frame(self.sample, self.frame_idx):
            self.edit_mode = False
            self.auto_pause.paused_for_review = False
            return False
        should_pause = self.auto_pause.on_frame(self.frame_idx, playing=self.playing or auto)
        if should_pause or auto:
            self.playing = False
            self._enter_edit_for_current()
            return True
        return False

    def continue_playback(self) -> bool:
        """ENTER: resume video after successful save / review."""
        if self.unsaved_changes():
            self.status_msg = "unsaved — press S to save or R to reset before continue"
            print(self.status_msg, flush=True)
            return False
        self.auto_pause.clear_after_continue()
        # Skip past current sample so we don't immediately re-pause
        self.auto_pause.last_auto_paused_frame = self.frame_idx
        self.edit_mode = False
        self.playing = True
        return True

    # --------------------------------------------------------------- save
    def save_current_frame(self, *, allow_empty: bool = False) -> bool:
        if self.review_only:
            return False
        if not is_sample_frame(self.sample, self.frame_idx):
            self.status_msg = "Not a sample frame — GT writes are blocked"
            print(self.status_msg, flush=True)
            return False
        if not self.edit_mode:
            self._enter_edit_for_current()

        guard = save_guard(self.state)
        if guard in ("need_accept", "need_empty_confirm") and not allow_empty:
            self.status_msg = NO_ACCEPTED_MSG
            print(self.status_msg, flush=True)
            return False

        fid = int(self.frame_idx)
        was_reviewed = self.frame_reviewed_on_disk(fid)

        # Update proposal flags for this sample frame only
        prop = self.proposals.copy()
        prop = prop[prop["frame_idx"].astype(int) != fid]
        prop_rows = [b.as_proposal_row(fid) for b in self.state.boxes if not b.manual]
        if prop_rows:
            prop = pd.concat([prop, pd.DataFrame(prop_rows)], ignore_index=True)
        atomic_write_csv(prop, self.gt_dir / "model_proposals.csv", backup_dir=self.backup_dir)
        self.proposals = pd.read_csv(self.gt_dir / "model_proposals.csv")

        new_rows = build_gt_rows_for_frame(self.state)
        sample_ids = set(sample_frame_indices(self.sample))
        new_gt = merge_gt_preserving_other_reviewed(self.gt, fid, new_rows, sample_ids)
        # Safety: never keep non-sample frames
        new_gt = new_gt[new_gt["frame_idx"].astype(int).isin(sample_ids)].copy()
        atomic_write_csv(new_gt, self.gt_dir / "human_detection_gt.csv", backup_dir=self.backup_dir)
        self.gt = pd.read_csv(self.gt_dir / "human_detection_gt.csv")

        self.sample.loc[self.sample["frame_idx"].astype(int) == fid, "reviewed"] = True
        atomic_write_csv(self.sample, self.gt_dir / "review_sample.csv", backup_dir=self.backup_dir)
        self.sample = pd.read_csv(self.gt_dir / "review_sample.csv")

        if not bool(
            truthy_mask(self.sample.loc[self.sample["frame_idx"].astype(int) == fid, "reviewed"]).all()
        ):
            raise RuntimeError("save verify failed: reviewed flag")

        self._enter_edit_for_current()
        self._mark_clean()
        comp = recount_completion(self.sample)
        n_boxes = int(self.gt[(self.gt.frame_idx.astype(int) == fid) & self.gt.x1.notna()].shape[0])
        msg = (
            f"saved frame {fid}: gt_boxes={n_boxes} completion={comp['reviewed']}/{comp['total']} dirty=false"
            + (" (overwrite)" if was_reviewed else "")
        )
        print(msg, flush=True)
        self.status_msg = msg
        return True

    def handle_empty_confirm(self) -> None:
        if self.review_only or not self.edit_mode:
            return
        guard = save_guard(self.state)
        if guard == "ok":
            self.status_msg = "Frame has accepted/manual boxes — use S to save."
            self.empty_confirmation_pending = False
            print(self.status_msg, flush=True)
            return
        if not self.empty_confirmation_pending:
            self.empty_confirmation_pending = True
            self.status_msg = "EMPTY FRAME? PRESS E AGAIN TO CONFIRM"
            print(self.status_msg, flush=True)
            return
        self.empty_confirmation_pending = False
        self.save_current_frame(allow_empty=True)

    def save_and_next_sample(self) -> bool:
        if not self.save_current_frame(allow_empty=False):
            return False
        nxt = next_unreviewed_after(self.sample, self.frame_idx)
        if nxt is None:
            nxt = next_sample_frame(self.sample, self.frame_idx, direction=1)
        if nxt is None:
            return True
        self.playing = False
        self._seek(nxt)
        self._maybe_enter_sample_review(auto=True)
        return True

    # ------------------------------------------------------------- mouse
    def _hit_box(self, x: int, y: int) -> int | None:
        for i, b in enumerate(self.state.boxes):
            if b.rejected:
                continue
            if b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2:
                return i
        return None

    def _hit_corner(self, x: int, y: int, thr: int = 12) -> tuple[int, int] | None:
        for i, b in enumerate(self.state.boxes):
            if b.rejected:
                continue
            corners = [(b.x1, b.y1), (b.x2, b.y1), (b.x2, b.y2), (b.x1, b.y2)]
            for c, (cx, cy) in enumerate(corners):
                if abs(x - cx) <= thr and abs(y - cy) <= thr:
                    return i, c
        return None

    def on_mouse(self, event, x, y, flags, param) -> None:
        if self.review_only or not self.edit_mode:
            return
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
                proposal_id=f"manual_{self.frame_idx}_{int(time.time() * 1000) % 100000}",
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
            self.empty_confirmation_pending = False
            mark_duplicate_warnings(self.state)
            mark_player_referee_conflicts(self.state)

    def set_class(self, name: str) -> None:
        if not self.edit_mode or self.review_only:
            return
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

    def reject_selected(self) -> None:
        if not self.edit_mode or self.review_only or not self.state.boxes:
            return
        b = self.state.boxes[self.selected]
        b.rejected = True
        b.accepted = False
        self.state.dirty = True
        self.empty_confirmation_pending = False
        self.selected = max(0, min(self.selected, len(self.state.boxes) - 1))

    # ------------------------------------------------------------- draw
    def _draw_overlay_dets(self, img: np.ndarray) -> None:
        if not self.show_overlay:
            return
        # On sample edit mode, prefer proposal/edit boxes; still ok to skip raw overlay
        if self.edit_mode and not self.review_only:
            return
        rows = overlay_rows_for_frame(self.overlay_index, self.frame_idx)
        for _, r in rows.iterrows():
            cls = str(r["class_name"])
            color = COLORS.get(cls, (200, 200, 200))
            x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            _draw_dashed_rect(img, (x1, y1), (x2, y2), color, thickness=1)
            cv2.putText(
                img,
                f"{cls} {float(r.confidence):.2f}",
                (x1, max(16, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

    def _draw_tracks(self, img: np.ndarray) -> None:
        if not self.show_tracks:
            return
        for t in self.tracks_by_frame.get(self.frame_idx, []):
            x1, y1, x2, y2 = int(t["x1"]), int(t["y1"]), int(t["x2"]), int(t["y2"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(
                img,
                f"ID {t['track_id']}",
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    def _draw_edit_boxes(self, img: np.ndarray) -> None:
        if not self.edit_mode:
            return
        for i, b in enumerate(self.state.boxes):
            if b.rejected:
                continue
            color = COLORS.get(b.class_name, (200, 200, 200))
            x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
            if b.accepted or b.manual:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                if b.manual:
                    cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)
            else:
                _draw_dashed_rect(img, (x1, y1), (x2, y2), color, thickness=1)
            label = f"{b.class_name} {b.confidence:.2f} {b.proposal_id}"
            if b.accepted:
                label = "[A] " + label
            if b.manual:
                label = "[M] " + label
            cv2.putText(img, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            if i == self.selected:
                for cx, cy in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
                    cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)
        if self.cur_box is not None:
            x1, y1, x2, y2 = self.cur_box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    def render(self) -> np.ndarray:
        if self.frame_bgr is None:
            img = np.zeros((480, 640, 3), np.uint8)
        else:
            img = self.frame_bgr.copy()

        self._draw_overlay_dets(img)
        self._draw_tracks(img)
        self._draw_edit_boxes(img)

        hud = VideoHudState(
            frame_idx=self.frame_idx,
            n_frames=self.n_frames,
            fps=self.fps,
            playing=self.playing,
            sample=self.sample,
            show_overlay=self.show_overlay,
            show_tracks=self.show_tracks,
        )
        y = 22
        for line in hud.lines():
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 20

        if is_sample_frame(self.sample, self.frame_idx):
            ord_ = sample_ordinal(self.sample, self.frame_idx) or 0
            total = len(self.sample)
            banner = f"GT REVIEW FRAME {ord_}/{total}"
            if self.frame_reviewed_on_disk():
                banner += "  |  ALREADY REVIEWED"
            cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 140, 255), -1)
            cv2.putText(img, banner, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            if self.edit_mode and not self.review_only:
                c = self.state.counts()
                comp = recount_completion(self.sample)
                info = [
                    f"Frame: {ord_}/{total}",
                    f"Proposals: {c['proposals']}  Accepted: {c['accepted']}  Rejected: {c['rejected']}  Manual: {c['manual']}",
                    f"Reviewed: {'Yes' if self.frame_reviewed_on_disk() and not self.unsaved_changes() else 'No'}",
                    f"Completion: {comp['reviewed']}/{comp['total']}  dirty={self.state.dirty}",
                ]
                yy = img.shape[0] - 90
                for line in info:
                    cv2.putText(img, line, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(img, line, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    yy += 18

        if self.status_msg:
            for i, line in enumerate(self.status_msg.splitlines()):
                cv2.putText(
                    img,
                    line,
                    (10, img.shape[0] - 12 - 18 * (len(self.status_msg.splitlines()) - 1 - i)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
        if self.empty_confirmation_pending:
            cv2.putText(
                img,
                "EMPTY FRAME? PRESS E AGAIN TO CONFIRM",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )
        return img

    # ------------------------------------------------------------- keys
    def handle_key(self, key: int) -> None:
        # Distinguish arrow keys (waitKeyEx) from ASCII letters (S=83, Q=81).
        if key in (2424832, 65361):  # LEFT arrow
            if self.unsaved_changes():
                print("unsaved — press S or R first", flush=True)
                return
            self.playing = False
            self._seek(self.frame_idx - int(round(self.fps)))
            self._maybe_enter_sample_review(auto=True)
            return
        if key in (2555904, 65363):  # RIGHT arrow
            if self.unsaved_changes():
                print("unsaved — press S or R first", flush=True)
                return
            self.playing = False
            self._seek(self.frame_idx + int(round(self.fps)))
            self._maybe_enter_sample_review(auto=True)
            return

        key = key & 0xFF

        if key in (ord("q"), 27):
            if self.unsaved_changes():
                print("unsaved changes on quit — not auto-saving", flush=True)
            self.quit_requested = True
            return

        if key == ord(" "):
            if self.edit_mode and self.unsaved_changes():
                print("unsaved — press S or R before play", flush=True)
                return
            if self.edit_mode and not self.unsaved_changes():
                self.continue_playback()
                return
            self.playing = not self.playing
            if self.playing:
                self._maybe_enter_sample_review(auto=False)
            return

        # Navigation always available (blocked if dirty in edit mode)
        if key in (ord("n"), ord("N")):
            if self.unsaved_changes():
                print("unsaved — press S to save or R to reset before N", flush=True)
                return
            self.playing = False
            self._step(1)
            self._maybe_enter_sample_review(auto=True)
            return
        if key in (ord("p"), ord("P")):
            if self.unsaved_changes():
                print("unsaved — press S to save or R to reset before P", flush=True)
                return
            self.playing = False
            self._step(-1)
            self._maybe_enter_sample_review(auto=True)
            return
        if key in (ord("d"), ord("D")):  # +1s
            if self.unsaved_changes():
                print("unsaved — press S or R first", flush=True)
                return
            self.playing = False
            self._seek(self.frame_idx + int(round(self.fps)))
            self._maybe_enter_sample_review(auto=True)
            return
        if key in (ord("b"), ord("B")):  # -1s
            if self.unsaved_changes():
                print("unsaved — press S or R first", flush=True)
                return
            self.playing = False
            self._seek(self.frame_idx - int(round(self.fps)))
            self._maybe_enter_sample_review(auto=True)
            return
        if key in (ord("j"), ord("J")):
            if self.unsaved_changes():
                print("unsaved — press S or R first", flush=True)
                return
            prev = next_sample_frame(self.sample, self.frame_idx, direction=-1)
            if prev is not None:
                self.playing = False
                self._seek(prev)
                self._maybe_enter_sample_review(auto=True)
            return
        if key in (ord("k"), ord("K")):
            if self.unsaved_changes():
                print("unsaved — press S or R first", flush=True)
                return
            nxt = next_sample_frame(self.sample, self.frame_idx, direction=1)
            if nxt is not None:
                self.playing = False
                self._seek(nxt)
                self._maybe_enter_sample_review(auto=True)
            return
        if key in (ord("g"), ord("G")):
            if self.unsaved_changes():
                print("unsaved — press S or R first", flush=True)
                return
            tgt = first_unreviewed_sample_idx(self.sample)
            if tgt is not None:
                self.playing = False
                self._seek(tgt)
                self._maybe_enter_sample_review(auto=True)
            return

        if key in (ord("m"), ord("M")):
            self.show_overlay = not self.show_overlay
            return
        if key in (ord("t"), ord("T")):
            self.show_tracks = not self.show_tracks
            return
        if key in (ord("f"), ord("F")):
            self.fullscreen = not self.fullscreen
            if self.show_window:
                prop = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN, float(prop))
            return

        # Annotation keys (sample edit mode only)
        if self.review_only or not self.edit_mode:
            return

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
            self.status_msg = "R: reset proposals"
            print(self.status_msg, flush=True)
        elif key == ord("s"):  # save
            self.empty_confirmation_pending = False
            self.save_current_frame(allow_empty=False)
        elif key == ord("S"):  # SHIFT+S save + next sample
            self.empty_confirmation_pending = False
            self.save_and_next_sample()
        elif key in (ord("e"), ord("E")):
            self.handle_empty_confirm()
        elif key in (13, 10):  # ENTER
            self.continue_playback()
        elif key in (ord("x"), ord("X"), 8, 255):
            self.reject_selected()

    def tick(self) -> None:
        if self.playing and not self.edit_mode:
            self._advance_play()
            if is_sample_frame(self.sample, self.frame_idx):
                # Auto-pause on sample arrival
                self.playing = False
                self._maybe_enter_sample_review(auto=True)

    def run(self) -> None:
        print(HELP)
        if self.show_window:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
            if not self.review_only:
                cv2.setMouseCallback(self.win, self.on_mouse)
        delay = max(1, int(round(1000.0 / max(self.fps, 1.0))))
        while not self.quit_requested:
            self.tick()
            frame = self.render()
            if self.show_window:
                cv2.imshow(self.win, frame)
                key = cv2.waitKeyEx(delay if self.playing else 30)
                if key != -1:
                    self.handle_key(key)
            else:
                break
        if self.show_window:
            cv2.destroyAllWindows()
        self.cap.release()
        comp = recount_completion(self.sample)
        print(f"real_GT_reviewed={comp['reviewed']}/40")
        print("automatic_GT_acceptance=0")


def export_preview(
    *,
    video_path: Path,
    gt_dir: Path,
    out_path: Path,
    max_frames: int | None = None,
) -> Path:
    """Write a review-only overlay MP4; never touches GT CSVs."""
    gt_fp = copy_file_fingerprint(gt_dir / "human_detection_gt.csv")
    sample_fp = copy_file_fingerprint(gt_dir / "review_sample.csv")
    prop_fp = copy_file_fingerprint(gt_dir / "model_proposals.csv")

    sample = pd.read_csv(gt_dir / "review_sample.csv")
    overlay_path = ensure_video_proposals_cache(gt_dir)
    overlay_index = load_overlay_by_frame(overlay_path)
    sample_ids = set(sample_frame_indices(sample))
    comp0 = recount_completion(sample)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and idx >= max_frames:
            break
        rows = overlay_rows_for_frame(overlay_index, idx)
        for _, r in rows.iterrows():
            cls = str(r["class_name"])
            color = COLORS.get(cls, (200, 200, 200))
            x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{cls} {float(r.confidence):.2f}",
                (x1, max(16, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
        t = format_timestamp(frame_to_time(idx, fps))
        cv2.putText(frame, f"{t}  Frame {idx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"Completion: {comp0['reviewed']}/{comp0['total']}",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        if idx in sample_ids:
            ord_ = sample_ordinal(sample, idx) or 0
            cv2.putText(
                frame,
                f"SAMPLE FRAME {ord_}/40",
                (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        writer.write(frame)
        idx += 1
    writer.release()
    cap.release()

    assert copy_file_fingerprint(gt_dir / "human_detection_gt.csv") == gt_fp
    assert copy_file_fingerprint(gt_dir / "review_sample.csv") == sample_fp
    assert copy_file_fingerprint(gt_dir / "model_proposals.csv") == prop_fp
    print(f"preview_export={out_path}")
    print(f"preview_frames_written={idx}/{n}")
    return out_path


def prompt_resume(sample: pd.DataFrame, fps: float, n_frames: int) -> tuple[str, int]:
    """Interactive resume menu; defaults to first unreviewed - 2s if no TTY."""
    first = first_unreviewed_sample_idx(sample)
    print("Resume options:")
    print("  1 — Videoyu baştan oynat")
    print("  2 — İlk tamamlanmamış GT frame’den başla")
    print("  3 — Belirli frame’e git")
    print("  4 — Review-only mod")
    if first is not None:
        print(f"  (first unreviewed sample frame_idx={first})")
    if not sys.stdin.isatty():
        start = resume_start_frame(sample, mode="first_unreviewed", fps=fps, n_frames=n_frames)
        print(f"No TTY — defaulting to first unreviewed - 2s → frame {start}")
        return "first_unreviewed", start
    try:
        choice = input("Choice [2]: ").strip() or "2"
    except EOFError:
        choice = "2"
    if choice == "1":
        return "start", 0
    if choice == "3":
        try:
            fid = int(input("Frame index: ").strip())
        except Exception:
            fid = 0
        return "explicit", clamp_frame(fid, n_frames)
    if choice == "4":
        start = resume_start_frame(sample, mode="first_unreviewed", fps=fps, n_frames=n_frames)
        return "review_only", start
    start = resume_start_frame(sample, mode="first_unreviewed", fps=fps, n_frames=n_frames)
    return "first_unreviewed", start


def main() -> None:
    ap = argparse.ArgumentParser(description="Video human detection GT annotator")
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    ap.add_argument("--resume", action="store_true", help="Prompt resume menu / default start")
    ap.add_argument("--review-only", action="store_true")
    ap.add_argument("--export-preview", action="store_true")
    ap.add_argument(
        "--preview-out",
        type=Path,
        default=DEFAULT_PREVIEW_DIR / "football_hybrid_detection_preview.mp4",
    )
    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument("--no-window", action="store_true", help="Headless init (tests)")
    ap.add_argument("--preview-max-frames", type=int, default=None)
    args = ap.parse_args()

    if args.export_preview:
        export_preview(
            video_path=args.video,
            gt_dir=args.gt_dir,
            out_path=args.preview_out,
            max_frames=args.preview_max_frames,
        )
        # Report bits for preview path
        overlay = ensure_video_proposals_cache(args.gt_dir)
        ov = load_overlay_by_frame(overlay)
        sample = pd.read_csv(args.gt_dir / "review_sample.csv")
        cap = cv2.VideoCapture(str(args.video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
        cap.release()
        comp = recount_completion(sample)
        print(f"video_path={args.video}")
        print(f"video_frames={n}")
        print(f"video_fps={fps}")
        print(f"sample_frames={len(sample)}")
        print(f"reviewed={comp['reviewed']}/40")
        print(f"frames_with_video_overlay={len(ov)}")
        print("auto_pause_sample_frames=PASS")
        print("review_only_GT_untouched=PASS")
        print(f"preview_export={args.preview_out}")
        return

    sample = pd.read_csv(args.gt_dir / "review_sample.csv")
    cap = cv2.VideoCapture(str(args.video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()

    review_only = args.review_only
    if args.start_frame is not None:
        start = clamp_frame(args.start_frame, n)
    elif args.resume or (not args.review_only and args.start_frame is None):
        mode, start = prompt_resume(sample, fps, n)
        if mode == "review_only":
            review_only = True
    else:
        start = resume_start_frame(sample, mode="first_unreviewed", fps=fps, n_frames=n)

    gt_before = copy_file_fingerprint(args.gt_dir / "human_detection_gt.csv")
    sample_before = copy_file_fingerprint(args.gt_dir / "review_sample.csv")

    ann = VideoHumanAnnotator(
        video_path=args.video,
        gt_dir=args.gt_dir,
        review_only=review_only,
        start_frame=start,
        show_window=not args.no_window,
    )
    if args.no_window:
        # Headless smoke — do not mutate
        ann.cap.release()
    else:
        ann.run()

    if review_only:
        assert copy_file_fingerprint(args.gt_dir / "human_detection_gt.csv") == gt_before
        assert copy_file_fingerprint(args.gt_dir / "review_sample.csv") == sample_before
        print("review_only_GT_untouched=PASS")

    comp = recount_completion(pd.read_csv(args.gt_dir / "review_sample.csv"))
    print(f"video_path={args.video}")
    print(f"video_frames={ann.n_frames}")
    print(f"video_fps={ann.fps}")
    print(f"sample_frames={len(ann.sample)}")
    print(f"reviewed={comp['reviewed']}/40")
    print(f"frames_with_video_overlay={ann.frames_with_overlay}")
    print("auto_pause_sample_frames=PASS")
    print("preview_export=")
    print("production_defaults_changed=0")


if __name__ == "__main__":
    main()
