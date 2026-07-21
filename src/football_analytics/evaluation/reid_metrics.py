"""Offline ReID / identity health metrics for a pipeline run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_analytics.analytics.reid_matching import (
    calibrate_hard_negatives,
    cosine_similarity,
)


def evaluate_reid_run(run_dir: Path | str) -> dict[str, Any]:
    """Compute coverage, hard-negative separation, and identity stitch health."""
    run_dir = Path(run_dir)
    report: dict[str, Any] = {"run_dir": str(run_dir), "status": "ok"}

    tracks_path = run_dir / "tracks.parquet"
    proto_path = run_dir / "track_reid_prototypes.parquet"
    if not tracks_path.is_file():
        return {"status": "MISSING_TRACKS", "run_dir": str(run_dir)}

    tracks = pd.read_parquet(tracks_path)
    person = tracks[tracks["object_type"].eq("person")] if "object_type" in tracks.columns else tracks
    n_person_tracks = int(person["track_id"].nunique()) if not person.empty else 0

    prototypes = pd.read_parquet(proto_path) if proto_path.is_file() else pd.DataFrame()
    n_proto = int(len(prototypes))
    coverage = (n_proto / n_person_tracks) if n_person_tracks else 0.0
    report["person_tracks"] = n_person_tracks
    report["prototype_tracks"] = n_proto
    report["embedding_coverage"] = round(coverage, 4)

    emb: dict[int, np.ndarray] = {}
    intervals: dict[int, tuple[float, float]] = {}
    teams: dict[int, str | None] = {}
    if not prototypes.empty:
        for row in prototypes.itertuples(index=False):
            emb[int(row.track_id)] = np.asarray(list(row.embedding), dtype=np.float64)
    for tid, g in person.groupby("track_id"):
        intervals[int(tid)] = (float(g["timestamp_ms"].min()), float(g["timestamp_ms"].max()))

    ident_path = run_dir / "track_identities.parquet"
    if ident_path.is_file():
        ident = pd.read_parquet(ident_path)
        for tid, g in ident.groupby("track_id"):
            assigned = g[g["team_id"].notna()] if "team_id" in g.columns else g.iloc[0:0]
            if not assigned.empty:
                teams[int(tid)] = str(assigned.iloc[-1]["team_id"])

    calibration = calibrate_hard_negatives(emb, intervals, teams)
    report["hard_negative_calibration"] = {
        "pair_count": calibration.pair_count,
        "mean": calibration.mean,
        "p50": calibration.p50,
        "p75": calibration.p75,
        "p90": calibration.p90,
        "merge_threshold": calibration.merge_threshold,
        "strong_threshold": calibration.strong_threshold,
    }

    quality_path = run_dir / "identity_quality.json"
    if quality_path.is_file():
        report["identity_quality"] = json.loads(quality_path.read_text(encoding="utf-8"))

    # Simultaneous separation diagnostic
    same_team_sims: list[float] = []
    for i, a in enumerate(sorted(emb)):
        for b in sorted(emb)[i + 1 :]:
            if a not in intervals or b not in intervals:
                continue
            ia, ib = intervals[a], intervals[b]
            if ia[1] + 40 < ib[0] or ib[1] + 40 < ia[0]:
                continue
            if teams.get(a) and teams.get(b) and teams[a] == teams[b]:
                sim = cosine_similarity(emb[a], emb[b])
                if sim is not None:
                    same_team_sims.append(sim)
    report["simultaneous_same_team_hard_negatives"] = {
        "count": len(same_team_sims),
        "mean": float(np.mean(same_team_sims)) if same_team_sims else None,
        "p90": float(np.percentile(same_team_sims, 90)) if same_team_sims else None,
    }

    solved = (
        coverage >= 0.35
        and calibration.pair_count >= 5
        and bool(report.get("identity_quality"))
        and int(report["identity_quality"].get("merged_fragments", 0)) >= 0
    )
    flags = []
    iq = report.get("identity_quality") or {}
    if iq.get("identity_flags"):
        flags.extend(iq["identity_flags"])
    if coverage < 0.35:
        flags.append("LOW_REID_COVERAGE")
    report["flags"] = flags
    report["reid_solved"] = bool(
        coverage >= 0.35
        and calibration.pair_count >= 5
        and iq.get("reid_status") == "SOLVED"
        and not any("INVALID_PLAYER_IDENTITY_COUNT" in f for f in flags)
    )
    report["note"] = (
        "reid_solved requires broad embedding coverage, hard-negative calibration, "
        "and ≤11 validated identities per team after stitching."
    )

    out = run_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "reid_evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
