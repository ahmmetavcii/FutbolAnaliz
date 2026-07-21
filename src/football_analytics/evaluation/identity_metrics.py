"""Global identity evaluation (IDF1 etc.) — blocked until GT complete."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_analytics.evaluation.identity_gt import identity_gt_complete


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def evaluate_global_identity(
    *,
    gt_csv: Path,
    run_dir: Path,
    out_dir: Path | None = None,
    iou_threshold: float = 0.3,
) -> dict[str, Any]:
    gt = pd.read_csv(gt_csv) if Path(gt_csv).is_file() else pd.DataFrame()
    complete, reason = identity_gt_complete(gt)
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "status": "GT_INCOMPLETE" if not complete else "OK",
        "gt_incomplete_reason": None if complete else reason,
        "gt_frame_count": int(gt["frame_index"].nunique()) if not gt.empty else 0,
        "gt_player_count": int(gt["gt_player_id"].nunique()) if not gt.empty else 0,
        "note": "IDF1/ID precision/recall require completed identity ground truth.",
    }
    if not complete:
        (out / "identity_evaluation_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        # Empty placeholder artifacts
        pd.DataFrame().to_parquet(out / "identity_mapping.parquet", index=False)
        for name in (
            "identity_false_merges.csv",
            "identity_false_splits.csv",
            "identity_switches.csv",
        ):
            pd.DataFrame().to_csv(out / name, index=False)
        return report

    tracks = pd.read_parquet(run_dir / "tracks.parquet")
    gmap = (
        pd.read_parquet(run_dir / "global_identity_map.parquet")
        if (run_dir / "global_identity_map.parquet").is_file()
        else pd.DataFrame()
    )
    tid_to_gid = {}
    if not gmap.empty:
        # expect columns local_track_id / global_player_id variants
        lt = "local_track_id" if "local_track_id" in gmap.columns else "track_id"
        gt_col = "global_player_id" if "global_player_id" in gmap.columns else "global_id"
        for _, r in gmap.iterrows():
            tid_to_gid[int(r[lt])] = int(r[gt_col])

    # Match GT boxes to local tracks per frame, then map to global IDs
    matches: list[dict[str, Any]] = []
    for _, grow in gt.iterrows():
        if not bool(grow.get("visible", True)):
            continue
        fid = int(grow["frame_index"])
        gbox = (
            float(grow["x1"]),
            float(grow["y1"]),
            float(grow["x2"]),
            float(grow["y2"]),
        )
        frm = tracks[tracks["frame_id"] == fid]
        if frm.empty:
            continue
        best_tid, best_iou = None, 0.0
        for _, t in frm.iterrows():
            if not {"x1", "y1", "x2", "y2"}.issubset(t.index):
                continue
            tbox = (float(t["x1"]), float(t["y1"]), float(t["x2"]), float(t["y2"]))
            iou = _iou(gbox, tbox)
            if iou > best_iou:
                best_iou = iou
                best_tid = int(t["track_id"])
        if best_tid is None or best_iou < iou_threshold:
            continue
        matches.append(
            {
                "frame_index": fid,
                "gt_player_id": grow["gt_player_id"],
                "gt_team_id": grow.get("team_id"),
                "gt_role": grow.get("role"),
                "local_track_id": best_tid,
                "global_player_id": tid_to_gid.get(best_tid),
                "iou": best_iou,
            }
        )

    map_df = pd.DataFrame(matches)
    map_df.to_parquet(out / "identity_mapping.parquet", index=False)

    # Hungarian-style clear matching via majority vote: global ↔ gt
    false_merges = []
    false_splits = []
    switches = []
    if not map_df.empty:
        # For each global ID, which GT players it covers
        for gid, g in map_df.dropna(subset=["global_player_id"]).groupby("global_player_id"):
            gt_ids = g["gt_player_id"].value_counts()
            if len(gt_ids) > 1:
                false_merges.append(
                    {
                        "global_player_id": int(gid),
                        "gt_player_ids": ",".join(str(x) for x in gt_ids.index.tolist()),
                        "counts": ",".join(str(int(x)) for x in gt_ids.values.tolist()),
                    }
                )
        for gt_id, g in map_df.groupby("gt_player_id"):
            gids = g["global_player_id"].dropna().value_counts()
            if len(gids) > 1:
                false_splits.append(
                    {
                        "gt_player_id": gt_id,
                        "global_player_ids": ",".join(str(int(x)) for x in gids.index.tolist()),
                        "counts": ",".join(str(int(x)) for x in gids.values.tolist()),
                    }
                )
            # ID switches: changes of global id along time for same GT
            seq = g.sort_values("frame_index")["global_player_id"].tolist()
            prev = None
            for fid, gid in zip(g.sort_values("frame_index")["frame_index"], seq):
                if prev is not None and gid is not None and prev is not None and gid != prev:
                    switches.append(
                        {"gt_player_id": gt_id, "frame_index": int(fid), "from": prev, "to": gid}
                    )
                if gid is not None:
                    prev = gid

    # IDF1 approximation from matched pairs using bipartite majority mapping
    # IDTP: frames where predicted global matches the dominant GT mapping
    idtp = idfp = idfn = 0
    if not map_df.empty:
        # dominant gt for each global
        dom_gt = (
            map_df.dropna(subset=["global_player_id"])
            .groupby("global_player_id")["gt_player_id"]
            .agg(lambda s: s.value_counts().index[0])
            .to_dict()
        )
        dom_gid = (
            map_df.dropna(subset=["global_player_id"])
            .groupby("gt_player_id")["global_player_id"]
            .agg(lambda s: s.value_counts().index[0])
            .to_dict()
        )
        for _, r in map_df.iterrows():
            gid = r.get("global_player_id")
            gt_id = r["gt_player_id"]
            if gid is None or (isinstance(gid, float) and np.isnan(gid)):
                idfn += 1
                continue
            if dom_gt.get(gid) == gt_id and dom_gid.get(gt_id) == gid:
                idtp += 1
            else:
                idfp += 1
                idfn += 1

    id_precision = idtp / max(idtp + idfp, 1)
    id_recall = idtp / max(idtp + idfn, 1)
    idf1 = 2 * id_precision * id_recall / max(id_precision + id_recall, 1e-9)

    # Fragmentation / purity
    frag = (
        float(map_df.groupby("gt_player_id")["global_player_id"].nunique().mean())
        if not map_df.empty
        else None
    )

    report.update(
        {
            "idf1": round(idf1, 4),
            "id_precision": round(id_precision, 4),
            "id_recall": round(id_recall, 4),
            "id_switches": len(switches),
            "false_merges": len(false_merges),
            "false_splits": len(false_splits),
            "fragmentation_count": len(false_splits),
            "mean_globals_per_gt_player": round(frag, 4) if frag is not None else None,
            "matched_observations": int(len(map_df)),
        }
    )
    pd.DataFrame(false_merges).to_csv(out / "identity_false_merges.csv", index=False)
    pd.DataFrame(false_splits).to_csv(out / "identity_false_splits.csv", index=False)
    pd.DataFrame(switches).to_csv(out / "identity_switches.csv", index=False)
    (out / "identity_evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
