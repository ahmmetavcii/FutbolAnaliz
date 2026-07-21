"""Manual ground-truth schema and evaluation for short Opta accuracy clips."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_analytics.utils.io import write_json


GT_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class GroundTruthSpec:
    """Expected files under a GT directory."""

    players_csv: str = "gt_players.csv"
    ball_csv: str = "gt_ball.csv"
    touches_csv: str = "gt_touches.csv"
    passes_csv: str = "gt_passes.csv"
    identity_csv: str = "gt_identity.csv"


def write_ground_truth_template(out_dir: Path) -> dict[str, str]:
    """Create empty CSV templates + README for a 15–30s labelled clip."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = GroundTruthSpec()
    players = pd.DataFrame(
        columns=[
            "frame_id",
            "timestamp_ms",
            "gt_player_id",
            "team_id",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "pitch_x",
            "pitch_y",
        ]
    )
    ball = pd.DataFrame(
        columns=[
            "frame_id",
            "timestamp_ms",
            "ball_x_pixel",
            "ball_y_pixel",
            "pitch_x",
            "pitch_y",
            "visible",
        ]
    )
    touches = pd.DataFrame(
        columns=[
            "touch_id",
            "timestamp_ms",
            "gt_player_id",
            "team_id",
            "touch_type",
        ]
    )
    passes = pd.DataFrame(
        columns=[
            "pass_id",
            "start_ms",
            "end_ms",
            "passer_gt_id",
            "receiver_gt_id",
            "team_id",
            "successful",
        ]
    )
    identity = pd.DataFrame(
        columns=[
            "gt_player_id",
            "team_id",
            "jersey_number",
            "local_track_ids_expected",
            "notes",
        ]
    )
    players.to_csv(out_dir / spec.players_csv, index=False)
    ball.to_csv(out_dir / spec.ball_csv, index=False)
    touches.to_csv(out_dir / spec.touches_csv, index=False)
    passes.to_csv(out_dir / spec.passes_csv, index=False)
    identity.to_csv(out_dir / spec.identity_csv, index=False)
    readme = out_dir / "README.md"
    readme.write_text(
        f"""# Opta short-clip ground truth (schema {GT_SCHEMA_VERSION})

Label a 15–30s clip manually. Unit-test green ≠ real accuracy.

## Files
- `{spec.players_csv}` — per-frame player boxes + team + stable gt_player_id
- `{spec.ball_csv}` — ball location when visible
- `{spec.touches_csv}` — contact times
- `{spec.passes_csv}` — pass start/end + success
- `{spec.identity_csv}` — real player matching notes

## Metrics produced by `evaluate_against_ground_truth`
- player ID precision/recall
- ID switches
- ball detection recall
- ball tracking coverage
- touch precision/recall
- pass precision/recall
""",
        encoding="utf-8",
    )
    write_json(out_dir / "schema.json", {"version": GT_SCHEMA_VERSION, **asdict(spec)})
    return {
        "players": str(out_dir / spec.players_csv),
        "ball": str(out_dir / spec.ball_csv),
        "touches": str(out_dir / spec.touches_csv),
        "passes": str(out_dir / spec.passes_csv),
        "identity": str(out_dir / spec.identity_csv),
        "readme": str(readme),
    }


def _pr(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_against_ground_truth(
    run_dir: Path,
    gt_dir: Path,
    *,
    touch_tolerance_ms: float = 200.0,
    pass_tolerance_ms: float = 400.0,
) -> dict[str, Any]:
    """Compare pipeline outputs to manual GT. Returns null metrics when GT empty."""
    run_dir = Path(run_dir)
    gt_dir = Path(gt_dir)
    spec = GroundTruthSpec()
    result: dict[str, Any] = {
        "schema_version": GT_SCHEMA_VERSION,
        "gt_dir": str(gt_dir),
        "run_dir": str(run_dir),
        "note": "Unit tests are not a substitute for this evaluation.",
    }

    gt_players = (
        pd.read_csv(gt_dir / spec.players_csv)
        if (gt_dir / spec.players_csv).is_file()
        else pd.DataFrame()
    )
    gt_ball = (
        pd.read_csv(gt_dir / spec.ball_csv) if (gt_dir / spec.ball_csv).is_file() else pd.DataFrame()
    )
    gt_touches = (
        pd.read_csv(gt_dir / spec.touches_csv)
        if (gt_dir / spec.touches_csv).is_file()
        else pd.DataFrame()
    )
    gt_passes = (
        pd.read_csv(gt_dir / spec.passes_csv)
        if (gt_dir / spec.passes_csv).is_file()
        else pd.DataFrame()
    )

    # Identity / ID switches from identity report if present
    id_report = (
        pd.read_parquet(run_dir / "global_identity_report.parquet")
        if (run_dir / "global_identity_report.parquet").is_file()
        else pd.DataFrame()
    )
    gmap = (
        pd.read_parquet(run_dir / "global_identity_map.parquet")
        if (run_dir / "global_identity_map.parquet").is_file()
        else pd.DataFrame()
    )
    if not id_report.empty:
        result["id_switches"] = int(
            id_report["track_fragment_count"].astype(int).map(lambda n: max(0, n - 1)).sum()
        )
        result["validated_players"] = int(id_report["validated"].astype(bool).sum())
    else:
        result["id_switches"] = None
        result["validated_players"] = None

    if not gt_players.empty and not gmap.empty and "gt_player_id" in gt_players.columns:
        # Crude ID precision: predicted validated globals vs unique GT players
        pred_n = int(gmap.loc[gmap.get("validated", True) == True, "global_id"].nunique())  # noqa: E712
        gt_n = int(gt_players["gt_player_id"].nunique())
        tp = min(pred_n, gt_n)
        result["player_id"] = _pr(tp, max(0, pred_n - tp), max(0, gt_n - tp))
    else:
        result["player_id"] = {
            "precision": None,
            "recall": None,
            "reason": "gt_or_predictions_missing",
        }

    ball = (
        pd.read_parquet(run_dir / "ball_trajectory.parquet")
        if (run_dir / "ball_trajectory.parquet").is_file()
        else pd.DataFrame()
    )
    if not gt_ball.empty and not ball.empty:
        gt_vis = gt_ball[gt_ball.get("visible", True) == True]  # noqa: E712
        matched = 0
        for row in gt_vis.itertuples(index=False):
            fid = int(row.frame_id)
            hit = ball[ball["frame_id"] == fid]
            if hit.empty:
                continue
            if bool(hit.iloc[0].get("visible", False)):
                matched += 1
        result["ball_detection_recall"] = (
            None if len(gt_vis) == 0 else round(matched / len(gt_vis), 4)
        )
        result["ball_tracking_coverage"] = round(float(ball["visible"].mean()), 4)
    else:
        result["ball_detection_recall"] = None
        result["ball_tracking_coverage"] = None

    touches = (
        pd.read_parquet(run_dir / "touch_events.parquet")
        if (run_dir / "touch_events.parquet").is_file()
        else pd.DataFrame()
    )
    if not gt_touches.empty and not touches.empty:
        tp = fp = 0
        used = set()
        for row in gt_touches.itertuples(index=False):
            t = float(row.timestamp_ms)
            candidates = touches[
                (touches["timestamp_ms"] - t).abs() <= touch_tolerance_ms
            ]
            if candidates.empty:
                continue
            tp += 1
            used.add(int(candidates.index[0]))
        fp = max(0, len(touches) - len(used))
        fn = max(0, len(gt_touches) - tp)
        result["touch"] = _pr(tp, fp, fn)
    else:
        result["touch"] = {"precision": None, "recall": None, "reason": "gt_or_pred_missing"}

    passes = (
        pd.read_parquet(run_dir / "pass_events.parquet")
        if (run_dir / "pass_events.parquet").is_file()
        else pd.DataFrame()
    )
    if not gt_passes.empty and not passes.empty:
        tp = 0
        for row in gt_passes.itertuples(index=False):
            start = float(row.start_ms)
            hit = passes[(passes["start_time_ms"] - start).abs() <= pass_tolerance_ms]
            if not hit.empty:
                tp += 1
        fp = max(0, len(passes) - tp)
        fn = max(0, len(gt_passes) - tp)
        result["pass"] = _pr(tp, fp, fn)
    else:
        result["pass"] = {"precision": None, "recall": None, "reason": "gt_or_pred_missing"}

    result["unit_tests_are_not_accuracy"] = True
    return result
