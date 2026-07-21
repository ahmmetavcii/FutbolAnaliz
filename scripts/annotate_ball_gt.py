#!/usr/bin/env python3
"""Ball GT annotator — 50-frame balanced review sample only.

Root cause of previous save counter bug:
  gt['reviewed'].fillna(False).astype(bool) treats the string "False" as True
  because bool("False") is True in Python. Also S did not set reviewed=True.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from football_analytics.evaluation.ball_gt import load_ball_gt_csv
from football_analytics.evaluation.ball_gt_annotate import (
    atomic_write_ball_gt,
    first_unreviewed_sample_index,
    gt_index_for_frame,
    recount,
    save_current_annotation as lib_save,
)
from football_analytics.evaluation.boolean_utils import (
    apply_boolean_dtype,
    is_true,
)

HELP = """
SPACE/ENTER/S  accept suggestion (or current manual box) → reviewed=True → save+verify → next
V              ball not visible → clear bbox → reviewed=True → save+verify → next
Mouse drag     draw manual GT box (does NOT set reviewed; press S/SPACE to confirm)
O / F          toggle occluded / difficult (in memory only until S/SPACE/V)
R              clear manual GT bbox only (keep suggestion)
M              toggle model overlay
P/A            previous sample
N/D            next sample (blocked if current unsaved; use K to skip)
K              skip without reviewing
Q/ESC          quit (does not mark unreviewed as reviewed)
"""


def _counts(sample: pd.DataFrame, gt: pd.DataFrame) -> tuple[int, int, int, int]:
    c = recount(sample, gt)
    return c["sample_reviewed"], c["sample_size"], c["full_reviewed"], c["full_size"]


def _suggestion_for_frame(
    frame_idx: int,
    detections: pd.DataFrame | None,
    provenance: pd.DataFrame | None,
) -> dict:
    sug = {
        "suggested_x1": None,
        "suggested_y1": None,
        "suggested_x2": None,
        "suggested_y2": None,
        "suggestion_source": None,
        "suggestion_confidence": None,
        "candidates": [],
    }
    if detections is None or detections.empty:
        return sug
    g = detections[detections["frame_id"].astype(int) == int(frame_idx)]
    if g.empty:
        return sug
    cands = []
    for _, r in g.iterrows():
        cx = float(r.get("ball_x_pixel", np.nan))
        cy = float(r.get("ball_y_pixel", np.nan))
        if not np.isfinite(cx) or not np.isfinite(cy):
            continue
        w = float(r.get("bbox_w", 24) or 24)
        h = float(r.get("bbox_h", 24) or 24)
        conf = float(r.get("detection_confidence", 0) or 0)
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        cands.append({"box": box, "conf": conf, "source": r.get("detector_source")})
    if not cands:
        return sug
    cands.sort(key=lambda c: c["conf"], reverse=True)
    best = cands[0]
    sug["candidates"] = cands
    sug["suggested_x1"], sug["suggested_y1"], sug["suggested_x2"], sug["suggested_y2"] = best["box"]
    sug["suggestion_confidence"] = best["conf"]
    src = "detection"
    if provenance is not None and not provenance.empty:
        prow = provenance[provenance["frame_id"].astype(int) == int(frame_idx)]
        if not prow.empty:
            src = str(prow.iloc[0]["provenance"])
    sug["suggestion_source"] = src
    return sug


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate ball GT (50-frame review sample)")
    ap.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("configs/evaluation/short_clip_gt_template/football_ball"),
    )
    ap.add_argument(
        "--sample",
        type=Path,
        default=None,
        help="Path to review_sample.csv",
    )
    ap.add_argument("--resume", action="store_true", help="Start at first unreviewed sample frame")
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/home/ahmet/workspace/opta_analytics_smoke/run_20260720_154807_15747b"),
    )
    args = ap.parse_args()

    gt_path = args.gt_dir / "ball_gt.csv"
    sample_path = args.sample or (args.gt_dir / "review_sample.csv")
    frames_dir = args.gt_dir / "frames"
    preds_dir = args.gt_dir / "predictions"

    if not sample_path.is_file():
        raise SystemExit(
            f"missing {sample_path} — run scripts/create_ball_gt_review_sample.py first"
        )

    gt = load_ball_gt_csv(gt_path)
    sample = apply_boolean_dtype(
        pd.read_csv(sample_path),
        columns=("reviewed", "trajectory_jump", "multiple_candidate"),
    )
    sample = sample.sort_values("sample_order").reset_index(drop=True)
    if len(sample) != 50 or int(sample["frame_idx"].nunique()) != 50:
        raise SystemExit(f"review_sample must have 50 unique frames, got {len(sample)}")

    detections = provenance = None
    if args.run_dir.is_dir():
        if (args.run_dir / "football_ball_detections.parquet").is_file():
            detections = pd.read_parquet(args.run_dir / "football_ball_detections.parquet")
        if (args.run_dir / "ball_provenance.parquet").is_file():
            provenance = pd.read_parquet(args.run_dir / "ball_provenance.parquet")

    sample_idx = 0
    if args.resume:
        sample_idx = first_unreviewed_sample_index(sample, gt)

    show_pred = True
    drawing = False
    draft_box: list[int] | None = None
    dirty = False
    banner: str | None = None
    banner_until = 0.0

    def current_frame_idx() -> int:
        return int(sample.iloc[sample_idx]["frame_idx"])

    def save_current_annotation(
        *,
        accept_suggestion: bool = False,
        mark_invisible: bool = False,
        use_draft: bool = False,
    ) -> bool:
        nonlocal gt, dirty, banner, banner_until, draft_box
        fid = current_frame_idx()
        sug = _suggestion_for_frame(fid, detections, provenance)
        bbox = None
        clear = False
        visible: bool | None = None
        if mark_invisible:
            visible = False
            clear = True
        elif use_draft and draft_box is not None:
            x1, y1, x2, y2 = draft_box
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            bbox = (float(x1), float(y1), float(x2), float(y2))
            visible = True
        elif accept_suggestion and sug["suggested_x1"] is not None:
            bbox = (
                float(sug["suggested_x1"]),
                float(sug["suggested_y1"]),
                float(sug["suggested_x2"]),
                float(sug["suggested_y2"]),
            )
            visible = True
        else:
            gidx = gt_index_for_frame(gt, fid)
            if pd.notna(gt.at[gidx, "ball_x1"]):
                bbox = (
                    float(gt.at[gidx, "ball_x1"]),
                    float(gt.at[gidx, "ball_y1"]),
                    float(gt.at[gidx, "ball_x2"]),
                    float(gt.at[gidx, "ball_y2"]),
                )
                visible = True
            else:
                banner = "SAVE FAILED — no bbox/suggestion"
                banner_until = time.time() + 1.5
                print(f"[SAVE FAILED]\nframe_idx={fid}\nreason=no_bbox_or_suggestion")
                return False
        try:
            gt, res = lib_save(
                gt=gt,
                sample=sample,
                gt_path=gt_path,
                frame_idx=fid,
                ball_visible=visible,
                bbox=bbox,
                clear_bbox=clear,
            )
        except Exception as exc:  # noqa: BLE001
            banner = f"SAVE FAILED — {exc}"
            banner_until = time.time() + 2.0
            print(f"[SAVE FAILED]\nframe_idx={fid}\nreason={exc}")
            return False

        dirty = False
        draft_box = None
        banner = f"SAVED — sample reviewed {res['sample_reviewed']}/{res['sample_size']}"
        banner_until = time.time() + 1.0
        print(
            f"[SAVED]\nframe_idx={fid}\nsample_reviewed={res['sample_reviewed']}/{res['sample_size']}\n"
            f"full_reviewed={res['full_reviewed']}/{res['full_size']}\ncsv_verified=True"
        )
        return True

    def goto_next_unreviewed(after: int | None = None) -> None:
        nonlocal sample_idx
        start = (after if after is not None else sample_idx) + 1
        for j in list(range(start, len(sample))) + list(range(0, start)):
            fid = int(sample.iloc[j]["frame_idx"])
            gidx = gt_index_for_frame(gt, fid)
            if not is_true(gt.at[gidx, "reviewed"]):
                sample_idx = j
                return
        sample_idx = min(sample_idx, len(sample) - 1)

    def on_mouse(event, x, y, flags, param):  # noqa: ARG001
        nonlocal drawing, draft_box, dirty
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            draft_box = [x, y, x, y]
            dirty = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing and draft_box is not None:
            draft_box[2], draft_box[3] = x, y
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            if draft_box is not None:
                x1, x2 = sorted((draft_box[0], draft_box[2]))
                y1, y2 = sorted((draft_box[1], draft_box[3]))
                draft_box = [x1, y1, x2, y2]
                dirty = True
            # Do NOT set reviewed / do NOT auto-save

    win = "ball_gt_annotator"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(HELP)

    while True:
        fid = current_frame_idx()
        gidx = gt_index_for_frame(gt, fid)
        row = gt.loc[gidx]
        sug = _suggestion_for_frame(fid, detections, provenance)

        frame_path = frames_dir / f"frame_{fid:06d}.png"
        img = cv2.imread(str(frame_path))
        if img is None:
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        vis = img.copy()

        if show_pred:
            # other candidates red, selected suggestion yellow thick
            for i, c in enumerate(sug.get("candidates") or []):
                x1, y1, x2, y2 = [int(v) for v in c["box"]]
                color = (0, 255, 255) if i == 0 else (0, 0, 255)
                thick = 3 if i == 0 else 1
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
            pred_path = preds_dir / f"frame_{fid:06d}.png"
            if pred_path.is_file() and not sug.get("candidates"):
                pred = cv2.imread(str(pred_path))
                if pred is not None:
                    vis = cv2.addWeighted(vis, 0.65, pred, 0.35, 0)

        # Manual / saved GT green
        if pd.notna(row.get("ball_x1")) and is_true(row.get("ball_visible")):
            cv2.rectangle(
                vis,
                (int(row.ball_x1), int(row.ball_y1)),
                (int(row.ball_x2), int(row.ball_y2)),
                (0, 255, 0),
                2,
            )
        if draft_box is not None:
            cv2.rectangle(
                vis,
                (draft_box[0], draft_box[1]),
                (draft_box[2], draft_box[3]),
                (255, 128, 0),
                2,
            )

        s_rev, s_n, f_rev, f_n = _counts(sample, gt)
        rem = s_n - s_rev
        pct = 100.0 * s_rev / max(s_n, 1)
        lines = [
            f"Frame: {sample_idx + 1}/{s_n}   Video frame: {fid}",
            f"Sample reviewed: {s_rev}/{s_n}   remaining: {rem}   Completion: {pct:.1f}%",
            f"Full GT reviewed: {f_rev}/{f_n}",
            f"visible={row.get('ball_visible')} reviewed={is_true(row.get('reviewed'))} "
            f"dirty={dirty} pred={'on' if show_pred else 'off'}",
        ]
        y0 = 28
        for line in lines:
            cv2.putText(vis, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
            cv2.putText(vis, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            y0 += 28
        if banner and time.time() < banner_until:
            cv2.putText(vis, banner, (10, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            # Persist dataframe as-is without forcing reviewed on unreviewed frames
            atomic_write_ball_gt(gt, gt_path)
            break

        if key in (ord(" "), 13, ord("s"), ord("S")):
            use_draft = draft_box is not None
            ok = save_current_annotation(
                accept_suggestion=not use_draft,
                use_draft=use_draft,
                mark_invisible=False,
            )
            if ok:
                goto_next_unreviewed(sample_idx)

        if key in (ord("v"), ord("V")):
            ok = save_current_annotation(mark_invisible=True)
            if ok:
                goto_next_unreviewed(sample_idx)

        if key in (ord("o"), ord("O")):
            cur = is_true(gt.at[gidx, "occluded"])
            gt.at[gidx, "occluded"] = not cur
            dirty = True

        if key in (ord("f"), ord("F")):
            cur = is_true(gt.at[gidx, "difficult"])
            gt.at[gidx, "difficult"] = not cur
            dirty = True

        if key in (ord("r"), ord("R")):
            for c in ("ball_x1", "ball_y1", "ball_x2", "ball_y2"):
                gt.at[gidx, c] = np.nan
            gt.at[gidx, "ball_visible"] = pd.NA
            draft_box = None
            dirty = True

        if key in (ord("m"), ord("M")):
            show_pred = not show_pred

        if key in (ord("p"), ord("a"), ord("P"), ord("A")):
            sample_idx = max(0, sample_idx - 1)
            draft_box = None
            dirty = False

        if key in (ord("n"), ord("d"), ord("N"), ord("D")):
            if dirty or not is_true(gt.at[gidx, "reviewed"]):
                banner = "Bu kare henuz kaydedilmedi. Once S veya SPACE kullanin. (K=skip)"
                banner_until = time.time() + 1.5
            else:
                sample_idx = min(len(sample) - 1, sample_idx + 1)
                draft_box = None
                dirty = False

        if key in (ord("k"), ord("K")):
            sample_idx = min(len(sample) - 1, sample_idx + 1)
            draft_box = None
            dirty = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
