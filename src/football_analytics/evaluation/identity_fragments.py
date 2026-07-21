"""Classify football team_1 identity fragments without hard-capping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QUALITY_CLASSES = (
    "likely_real_player",
    "short_fragment",
    "duplicate_identity",
    "low_quality_crop",
    "team_assignment_instability",
    "role_instability",
    "camera_cut_fragment",
    "unresolved",
)


def classify_team_identities(
    run_dir: Path,
    *,
    team_id: str = "team_1",
    out_dir: Path | None = None,
    short_seconds: float = 0.5,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    report = pd.read_parquet(run_dir / "global_identity_report.parquet")
    decisions = (
        pd.read_parquet(run_dir / "global_identity_decisions.parquet")
        if (run_dir / "global_identity_decisions.parquet").is_file()
        else pd.DataFrame()
    )
    tracks = pd.read_parquet(run_dir / "tracks.parquet")
    identities = (
        pd.read_parquet(run_dir / "track_identities.parquet")
        if (run_dir / "track_identities.parquet").is_file()
        else pd.DataFrame()
    )

    team = report[report["team_id"].astype(str) == str(team_id)].copy()
    rows: list[dict[str, Any]] = []

    # Precompute bbox heights
    person = tracks[tracks["object_type"].eq("person")] if "object_type" in tracks.columns else tracks

    for _, r in team.iterrows():
        gid = int(r["global_player_id"])
        local_ids = [
            int(x)
            for x in str(r.get("local_track_ids") or "").split(",")
            if str(x).strip().isdigit()
        ]
        visible = float(r.get("visible_seconds") or 0.0)
        frag_count = int(r.get("track_fragment_count") or len(local_ids) or 1)
        reid = r.get("reid_cosine_mean")
        team_conf = r.get("team_colour_similarity_mean")
        role = str(r.get("role") or "unknown")
        simult = int(r.get("simultaneous_conflicts") or 0)

        track_rows = person[person["track_id"].isin(local_ids)] if local_ids else person.iloc[0:0]
        if not track_rows.empty and {"y1", "y2"}.issubset(track_rows.columns):
            heights = (track_rows["y2"] - track_rows["y1"]).astype(float)
            med_h = float(heights.median())
            first_f = int(track_rows["frame_id"].min())
            last_f = int(track_rows["frame_id"].max())
        else:
            med_h = float("nan")
            first_f = -1
            last_f = -1

        # Team instability across local tracks
        team_instability = False
        role_instability = False
        if not identities.empty and local_ids:
            sub = identities[identities["track_id"].isin(local_ids)]
            if "team_id" in sub.columns and sub["team_id"].nunique(dropna=True) > 1:
                team_instability = True
            if "role" in sub.columns and sub["role"].nunique(dropna=True) > 1:
                role_instability = True

        # Nearest merge candidate from decisions
        nearest = None
        reject_reason = None
        if not decisions.empty and local_ids:
            dsub = decisions[decisions["local_track_id"].isin(local_ids)]
            rejects = dsub[dsub.get("decision", pd.Series(dtype=str)).astype(str).eq("reject")]
            if not rejects.empty:
                reject_reason = str(rejects.iloc[0].get("reason"))
                nearest = rejects.iloc[0].get("candidate_global_id")

        # Classification (no auto-count as real players for short fragments).
        # Successfully merged fragments (frag_count>=2 + high reid) are real players,
        # not duplicates. Duplicates are separate global IDs that still look mergeable.
        if visible < short_seconds:
            qclass = "short_fragment"
        elif team_instability:
            qclass = "team_assignment_instability"
        elif role_instability:
            qclass = "role_instability"
        elif med_h == med_h and med_h < 40:
            qclass = "low_quality_crop"
        elif frag_count >= 2 and (reid is None or float(reid) >= 0.45):
            qclass = "likely_real_player"
        elif visible >= 2.0 and (med_h != med_h or med_h >= 60):
            qclass = "likely_real_player"
        elif reject_reason == "gap_too_long":
            qclass = "camera_cut_fragment"
        else:
            qclass = "unresolved"

        rows.append(
            {
                "global_player_id": gid,
                "local_track_ids": ",".join(str(x) for x in local_ids),
                "visible_seconds": visible,
                "first_frame": first_f,
                "last_frame": last_f,
                "fragment_count": frag_count,
                "median_bbox_height": med_h,
                "reid_confidence": reid,
                "team_confidence": team_conf,
                "role": role,
                "entry_pitch_location": None,
                "exit_pitch_location": None,
                "simultaneous_conflicts": simult,
                "nearest_merge_candidate": nearest,
                "merge_rejection_reason": reject_reason,
                "quality_status": qclass,
                "short_under_0_5s": bool(visible < short_seconds),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / f"identity_{team_id}_fragment_classes.csv", index=False)
    summary = {
        "team_id": team_id,
        "total_identities": int(len(df)),
        "class_counts": df["quality_status"].value_counts().to_dict() if not df.empty else {},
        "short_fragments_under_0_5s": int(df["short_under_0_5s"].sum()) if not df.empty else 0,
        "likely_real_player": int((df["quality_status"] == "likely_real_player").sum())
        if not df.empty
        else 0,
        "note": (
            "short_fragment and low_quality_crop are NOT counted as real players automatically; "
            "no hard-cap demotion applied."
        ),
    }
    (out / f"identity_{team_id}_fragment_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
