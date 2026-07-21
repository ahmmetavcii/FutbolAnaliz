"""Build manual touch review packages (no precision until reviewed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import pandas as pd


TOUCH_REVIEW_COLUMNS = [
    "touch_id",
    "start_frame",
    "end_frame",
    "peak_frame",
    "predicted_global_player_id",
    "predicted_team_id",
    "correct_touch",
    "correct_player",
    "actual_global_player_id",
    "false_positive_reason",
    "reviewed",
    "notes",
]


def touch_review_complete(csv_path: Path) -> tuple[bool, str]:
    if not Path(csv_path).is_file():
        return False, "missing_review_csv"
    df = pd.read_csv(csv_path)
    if df.empty:
        return False, "empty_review"
    if "reviewed" not in df.columns:
        return False, "missing_reviewed"
    if not bool(df["reviewed"].fillna(False).astype(bool).all()):
        return False, f"unreviewed:{int((~df['reviewed'].fillna(False).astype(bool)).sum())}"
    return True, "complete"


def build_touch_review_pack(
    *,
    run_dir: Path,
    video_path: Path,
    out_dir: Path | None = None,
    pad_seconds: float = 0.5,
    fps: float = 25.0,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / "evaluation" / "touch_review"
    out.mkdir(parents=True, exist_ok=True)

    touches = pd.read_parquet(run_dir / "touch_events.parquet")
    tracks = pd.read_parquet(run_dir / "tracks.parquet")
    ball = (
        pd.read_parquet(run_dir / "ball_provenance.parquet")
        if (run_dir / "ball_provenance.parquet").is_file()
        else pd.DataFrame()
    )
    ball_state = (
        pd.read_parquet(run_dir / "ball_state.parquet")
        if (run_dir / "ball_state.parquet").is_file()
        else pd.DataFrame()
    )
    pad = int(round(pad_seconds * fps))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    review_rows: list[dict[str, Any]] = []
    for i, t in enumerate(touches.itertuples(index=False), start=1):
        touch_id = str(getattr(t, "touch_id", f"touch-{i:04d}"))
        start = int(getattr(t, "start_frame", getattr(t, "frame_id", 0)))
        end = int(getattr(t, "end_frame", start))
        peak = int(getattr(t, "peak_confidence_frame", start))
        clip_start = max(0, start - pad)
        clip_end = end + pad
        folder = out / f"touch_{i:04d}"
        folder.mkdir(exist_ok=True)

        # Contact sheet from existing debug if present
        dbg = list((run_dir / "touch_debug").glob(f"*{touch_id}*"))
        if not dbg:
            dbg = list((run_dir / "touch_debug").glob(f"*f{peak}*"))

        # Write short mp4
        writer = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for fid in range(clip_start, clip_end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok:
                continue
            # overlays
            tid = getattr(t, "track_id", None)
            if tid is not None:
                tr = tracks[(tracks["frame_id"] == fid) & (tracks["track_id"] == int(tid))]
                if not tr.empty and {"x1", "y1", "x2", "y2"}.issubset(tr.columns):
                    r = tr.iloc[0]
                    cv2.rectangle(
                        frame,
                        (int(r.x1), int(r.y1)),
                        (int(r.x2), int(r.y2)),
                        (0, 255, 0),
                        2,
                    )
            if not ball_state.empty:
                bs = ball_state[ball_state["frame_id"] == fid]
                if not bs.empty:
                    bx = bs.iloc[0].get("ball_x_pixel")
                    by = bs.iloc[0].get("ball_y_pixel")
                    if bx is not None and by is not None and pd.notna(bx) and pd.notna(by):
                        cv2.circle(frame, (int(bx), int(by)), 8, (0, 255, 255), 2)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(folder / "clip.mp4"), fourcc, fps, (w, h))
            writer.write(frame)
            if fid == peak:
                cv2.imwrite(str(folder / "peak_frame.png"), frame)
        if writer is not None:
            writer.release()

        meta = {
            "touch_id": touch_id,
            "start_frame": start,
            "end_frame": end,
            "peak_frame": peak,
            "predicted_global_player_id": getattr(t, "global_player_id", None),
            "predicted_team_id": getattr(t, "team_id", None),
            "local_track_id": getattr(t, "track_id", None),
            "confidence": getattr(t, "confidence", None),
            "distance_px": getattr(t, "distance_px", None),
            "distance_m": getattr(t, "distance_m", None),
            "ball_velocity_change": getattr(t, "ball_velocity_change", None),
            "provenance": None,
        }
        if not ball.empty:
            prow = ball[ball["frame_id"] == peak]
            if not prow.empty:
                meta["provenance"] = str(prow.iloc[0]["provenance"])
        (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

        review_rows.append(
            {
                "touch_id": touch_id,
                "start_frame": start,
                "end_frame": end,
                "peak_frame": peak,
                "predicted_global_player_id": getattr(t, "global_player_id", None),
                "predicted_team_id": getattr(t, "team_id", None),
                "correct_touch": "",
                "correct_player": "",
                "actual_global_player_id": "",
                "false_positive_reason": "",
                "reviewed": False,
                "notes": "",
            }
        )

    cap.release()
    review_csv = out / "touch_review.csv"
    pd.DataFrame(review_rows, columns=TOUCH_REVIEW_COLUMNS).to_csv(review_csv, index=False)
    summary = {
        "touch_count": len(review_rows),
        "review_csv": str(review_csv),
        "status": "REVIEW_INCOMPLETE",
        "note": "Do not compute touch precision/recall until all rows are reviewed.",
    }
    (out / "touch_review_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
