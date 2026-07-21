#!/usr/bin/env python3
"""Simple OpenCV player identity GT annotator (evaluation only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from football_analytics.evaluation.identity_gt import IDENTITY_GT_COLUMNS

HELP = """
Keys:
  n       next frame
  p       previous frame
  drag    draw bbox
  1-9     set gt_player_id (1..9); 0 = 10
  t       cycle team_id team_0/team_1/referee
  r       cycle role outfield/goalkeeper/referee
  c       copy boxes from previous annotated frame
  m       toggle model overlay (if predictions exist)
  s       save
  x       mark wrong merge/split note
  DEL/d   delete last box on frame
  q       quit + save
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("configs/evaluation/short_clip_gt_template/football_identity"),
    )
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()

    frames_dir = args.gt_dir / "frames"
    gt_path = args.gt_dir / "identity_gt.csv"
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise SystemExit("no frames — run create_player_id_gt_sample.py first")

    if gt_path.is_file():
        gt = pd.read_csv(gt_path)
    else:
        gt = pd.DataFrame(columns=IDENTITY_GT_COLUMNS)

    frame_ids = [int(p.stem.split("_")[1]) for p in frame_files]
    idx = 0
    drawing = False
    ix = iy = 0
    draft: list[int] | None = None
    cur_pid = 1
    cur_team = "team_0"
    cur_role = "outfield"
    show_model = True

    tracks = None
    gmap = {}
    if args.run_dir and (Path(args.run_dir) / "tracks.parquet").is_file():
        tracks = pd.read_parquet(Path(args.run_dir) / "tracks.parquet")
        mp = Path(args.run_dir) / "global_identity_map.parquet"
        if mp.is_file():
            mdf = pd.read_parquet(mp)
            lt = "local_track_id" if "local_track_id" in mdf.columns else "track_id"
            gt_col = "global_player_id" if "global_player_id" in mdf.columns else "global_id"
            gmap = {int(r[lt]): int(r[gt_col]) for _, r in mdf.iterrows()}

    def rows_for(fid: int) -> pd.DataFrame:
        if gt.empty or "frame_index" not in gt.columns:
            return gt.iloc[0:0]
        return gt[gt["frame_index"] == fid]

    def save() -> None:
        cols = [c for c in IDENTITY_GT_COLUMNS if c in gt.columns] or IDENTITY_GT_COLUMNS
        out = gt.reindex(columns=cols)
        out.to_csv(gt_path, index=False)

    def on_mouse(event, x, y, flags, param):  # noqa: ARG001
        nonlocal drawing, ix, iy, draft, gt, cur_pid
        fid = frame_ids[idx]
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            draft = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and drawing and draft is not None:
            draft[2], draft[3] = x, y
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            if draft is None:
                return
            x1, x2 = sorted((draft[0], draft[2]))
            y1, y2 = sorted((draft[1], draft[3]))
            if abs(x2 - x1) < 4 or abs(y2 - y1) < 4:
                return
            new = {
                "frame_index": fid,
                "timestamp": fid / 25.0,
                "gt_player_id": cur_pid,
                "team_id": cur_team,
                "role": cur_role,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "visible": True,
                "occluded": False,
                "difficult": False,
                "reviewed": True,
            }
            gt = pd.concat([gt, pd.DataFrame([new])], ignore_index=True)
            draft = None
            save()

    win = "identity_gt_annotator"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(HELP)

    while True:
        fid = frame_ids[idx]
        img = cv2.imread(str(frame_files[idx]))
        vis = img.copy()
        if show_model and tracks is not None:
            frm = tracks[tracks["frame_id"] == fid]
            for _, t in frm.iterrows():
                if not {"x1", "y1", "x2", "y2"}.issubset(t.index):
                    continue
                tid = int(t.track_id)
                gid = gmap.get(tid, tid)
                cv2.rectangle(
                    vis,
                    (int(t.x1), int(t.y1)),
                    (int(t.x2), int(t.y2)),
                    (255, 128, 0),
                    1,
                )
                cv2.putText(
                    vis,
                    f"L{tid}/G{gid}",
                    (int(t.x1), max(0, int(t.y1) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 128, 0),
                    1,
                )
        for _, r in rows_for(fid).iterrows():
            cv2.rectangle(
                vis,
                (int(r.x1), int(r.y1)),
                (int(r.x2), int(r.y2)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                vis,
                f"GT{r.gt_player_id}/{r.team_id}/{r.role}",
                (int(r.x1), max(0, int(r.y1) - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )
        if draft is not None:
            cv2.rectangle(vis, (draft[0], draft[1]), (draft[2], draft[3]), (0, 0, 255), 1)

        reviewed_frames = (
            gt.loc[gt["reviewed"].fillna(False).astype(bool), "frame_index"].nunique()
            if not gt.empty and "reviewed" in gt.columns
            else 0
        )
        status = (
            f"{idx+1}/{len(frame_ids)} frame={fid} pid={cur_pid} team={cur_team} "
            f"role={cur_role} reviewed_frames={reviewed_frames}"
        )
        cv2.putText(vis, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            save()
            break
        if key == ord("n"):
            idx = min(len(frame_ids) - 1, idx + 1)
            draft = None
        if key == ord("p"):
            idx = max(0, idx - 1)
            draft = None
        if key == ord("t"):
            cur_team = {"team_0": "team_1", "team_1": "referee", "referee": "team_0"}[cur_team]
        if key == ord("r"):
            cur_role = {
                "outfield": "goalkeeper",
                "goalkeeper": "referee",
                "referee": "outfield",
            }[cur_role]
        if ord("1") <= key <= ord("9"):
            cur_pid = key - ord("0")
        if key == ord("0"):
            cur_pid = 10
        if key == ord("m"):
            show_model = not show_model
        if key == ord("s"):
            save()
            print(f"saved {gt_path}")
        if key == ord("x"):
            # Annotator note row
            note = {
                "frame_index": fid,
                "timestamp": fid / 25.0,
                "gt_player_id": cur_pid,
                "team_id": cur_team,
                "role": cur_role,
                "x1": None,
                "y1": None,
                "x2": None,
                "y2": None,
                "visible": False,
                "occluded": False,
                "difficult": True,
                "reviewed": True,
                "annotator_notes": "flagged_merge_or_split",
            }
            # only if column exists — keep schema; notes via difficult+role
            gt = pd.concat([gt, pd.DataFrame([{k: note.get(k) for k in IDENTITY_GT_COLUMNS}])], ignore_index=True)
            save()
        if key in (ord("d"), 8, 255):
            sub = rows_for(fid)
            if not sub.empty:
                drop_idx = sub.index[-1]
                gt = gt.drop(index=drop_idx).reset_index(drop=True)
                save()
        if key == ord("c"):
            # copy from previous frame with annotations
            prev = None
            for j in range(idx - 1, -1, -1):
                if not rows_for(frame_ids[j]).empty:
                    prev = frame_ids[j]
                    break
            if prev is not None:
                copies = rows_for(prev).copy()
                copies["frame_index"] = fid
                copies["timestamp"] = fid / 25.0
                copies["reviewed"] = True
                gt = pd.concat([gt, copies], ignore_index=True)
                save()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
