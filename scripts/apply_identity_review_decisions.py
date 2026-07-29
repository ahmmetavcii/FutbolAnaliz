#!/usr/bin/env python3
"""Apply human identity review decisions and re-render verified videos."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("/home/ahmet/projects/football-analytics")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path("/home/ahmet/workspace/global_identity_finalization/scripts")))

from football_analytics.evaluation.identity_review_store import IdentityReviewStore  # noqa: E402
from football_analytics.tracking.global_tracklet_association import (  # noqa: E402
    AssocConfig,
    associate_global_constrained,
    pairwise_veto,
)
import run_identity_finalization as idf  # noqa: E402

IDENTITY = Path("/mnt/c/football_data/results/tracking_identity_final")
OUT = Path("/mnt/c/football_data/results/tracking_human_verified")
REVIEW = ROOT / "configs/evaluation/identity_review/football"
VIDEO = Path("/mnt/c/football_data/videos/test_clips/football.mp4")


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log("Kararlar okunuyor")
    store = IdentityReviewStore(REVIEW)
    queue = pd.read_parquet(REVIEW / "review_queue.parquet")
    decisions = store.all_decisions()
    if len(decisions) < len(queue):
        log(f"WARNING: only {len(decisions)}/{len(queue)} decisions present")

    must_rows, cannot_rows, role_rows, conflict_rows = [], [], [], []

    log("Must-link kuralları uygulanıyor")
    log("Cannot-link kuralları uygulanıyor")
    for _, item in queue.iterrows():
        rid = str(item.review_id)
        d = decisions.get(rid)
        if not d:
            continue
        ta, tb = int(item.tracklet_a), int(item.tracklet_b)
        decision = d["human_decision"]
        if decision == "SAME":
            # physical safety check using tracklet table
            tracklets = pd.read_parquet(IDENTITY / "tracklets.parquet")
            # build minimal TrackletIdentity for veto
            from football_analytics.tracking.global_tracklet_association import TrackletIdentity

            def mini(tid: int):
                r = tracklets[tracklets.tracklet_id == tid].iloc[0]
                return TrackletIdentity(
                    tracklet_id=tid,
                    local_track_id=int(r.local_track_id),
                    class_name=str(r.class_name),
                    role=str(r.role),
                    team_id=int(r.team_id) if int(r.team_id) >= 0 else None,
                    team_confidence=0.8 if int(r.team_id) >= 0 else 0.0,
                    start_frame=int(r.start_frame),
                    end_frame=int(r.end_frame),
                    embedding=None,
                    embedding_variance=0.0,
                    valid_reid_crop_count=0,
                    jersey_number=item.jersey_a if tid == ta else item.jersey_b,
                    jersey_status="UNREADABLE",
                    jersey_confidence=0.0,
                    duration_s=float(r.duration),
                    quality="MEDIUM",
                    mean_conf=float(r.mean_conf),
                    start_pitch=None,
                    end_pitch=None,
                )

            a, b = mini(ta), mini(tb)
            # enrich jersey status if numbers present
            if a.jersey_number and b.jersey_number and str(a.jersey_number) != str(b.jersey_number):
                a.jersey_status = b.jersey_status = "CONFIRMED"
                a.jersey_confidence = b.jersey_confidence = 0.9
            veto = pairwise_veto(a, b, config=AssocConfig())
            if veto:
                conflict_rows.append(
                    {
                        "review_id": rid,
                        "tracklet_a": ta,
                        "tracklet_b": tb,
                        "human_decision": "SAME",
                        "conflict_status": "HUMAN_DECISION_CONFLICT",
                        "veto_reason": veto,
                    }
                )
            else:
                must_rows.append({"review_id": rid, "tracklet_a": ta, "tracklet_b": tb, "source": "human"})
        elif decision == "DIFFERENT":
            cannot_rows.append({"review_id": rid, "tracklet_a": ta, "tracklet_b": tb, "source": "human"})
        # UNSURE: no forced rule

        if int(d.get("role_flag") or 0):
            if d.get("role_a_override"):
                role_rows.append({"tracklet_id": ta, "role_override": d["role_a_override"], "review_id": rid, "auto_role": item.role_a})
            if d.get("role_b_override"):
                role_rows.append({"tracklet_id": tb, "role_override": d["role_b_override"], "review_id": rid, "auto_role": item.role_b})

    must_df = pd.DataFrame(must_rows)
    cannot_df = pd.DataFrame(cannot_rows)
    role_df = pd.DataFrame(role_rows)
    conflict_df = pd.DataFrame(conflict_rows)
    must_df.to_parquet(OUT / "human_must_links.parquet", index=False)
    cannot_df.to_parquet(OUT / "human_cannot_links.parquet", index=False)
    role_df.to_parquet(OUT / "human_role_overrides.parquet", index=False)
    conflict_df.to_parquet(OUT / "human_decision_conflicts.parquet", index=False)

    # Preserve old identity results (do not delete IDENTITY)
    assert (IDENTITY / "football_identity_final.mp4").exists()

    log("Global identity association yeniden çalışıyor")
    # Rebuild identities from cached features in identity final / workspace
    tracks = pd.read_parquet(IDENTITY / "local_tracks.parquet")
    tracklets = pd.read_parquet(IDENTITY / "tracklets.parquet")
    feat_dir = Path("/home/ahmet/workspace/global_identity_finalization/iterations/features_cache")
    if not (feat_dir / "tracklet_reid_prototypes.parquet").exists():
        feat_dir = IDENTITY
    feat = {
        "prototypes": {
            int(r.tracklet_id): r.to_dict()
            for _, r in pd.read_parquet(feat_dir / "tracklet_reid_prototypes.parquet").iterrows()
        },
        "shoes": {
            int(r.tracklet_id): r.to_dict()
            for _, r in pd.read_parquet(feat_dir / "shoe_sock_features.parquet").iterrows()
        },
        "jerseys": {
            int(r.tracklet_id): r.to_dict()
            for _, r in pd.read_parquet(feat_dir / "jersey_tracklet_evidence.parquet").iterrows()
        },
    }
    identities = idf.make_tracklet_identities(tracks, tracklets, feat, 25.0)

    # role overrides
    role_map = {int(r.tracklet_id): str(r.role_override) for _, r in role_df.iterrows()} if len(role_df) else {}
    for t in identities:
        if t.tracklet_id in role_map:
            ov = role_map[t.tracklet_id]
            t.role = ov
            if ov == "REFEREE":
                t.class_name = "referee"
            elif ov == "PLAYER":
                t.class_name = "player"
            elif ov in {"STAFF", "SPECTATOR"}:
                t.class_name = "staff"

    must_links = [(int(r.tracklet_a), int(r.tracklet_b)) for _, r in must_df.iterrows()] if len(must_df) else []
    cannot_links = [(int(r.tracklet_a), int(r.tracklet_b)) for _, r in cannot_df.iterrows()] if len(cannot_df) else []

    # mild threshold calibration from SAME/DIFFERENT reid sims
    sims_same, sims_diff = [], []
    for _, item in queue.iterrows():
        d = decisions.get(str(item.review_id))
        if not d or pd.isna(item.reid_similarity):
            continue
        if d["human_decision"] == "SAME":
            sims_same.append(float(item.reid_similarity))
        elif d["human_decision"] == "DIFFERENT":
            sims_diff.append(float(item.reid_similarity))
    cfg = AssocConfig(reid_merge=0.50, reid_strong=0.60, max_gap_s=8.0, max_gap_strong_s=16.0, min_merge_score=0.55)
    if sims_same and sims_diff:
        # keep between distributions without overfitting
        lo = float(np.percentile(sims_diff, 75)) if sims_diff else 0.48
        hi = float(np.percentile(sims_same, 25)) if sims_same else 0.55
        new_thr = float(np.clip(0.5 * (lo + hi), 0.45, 0.60))
        cfg.reid_merge = new_thr
        cfg.reid_strong = min(0.75, new_thr + 0.10)
        log(f"ReID merge threshold calibrated to {new_thr:.3f} (bounded)")

    mapping, merge_audit, players, reject_audit = associate_global_constrained(
        identities, config=cfg, must_links=must_links, cannot_links=cannot_links
    )

    log("Veto kontrolü yapılıyor")
    # ensure cannot-links not co-clustered
    tid_gid = {t.tracklet_id: mapping.get(t.tracklet_id) for t in identities}
    for a, b in cannot_links:
        if tid_gid.get(a) is not None and tid_gid.get(a) == tid_gid.get(b):
            raise RuntimeError(f"cannot-link violated: {a},{b}")

    # build global tracks
    t_index = idf._map_frame_to_tracklet(tracklets)
    tracklet_to_player = {}
    for p in players:
        for tid in p.tracklet_ids:
            tracklet_to_player[tid] = p

    human_verified_tids = set()
    for a, b in must_links:
        human_verified_tids.add(a)
        human_verified_tids.add(b)

    gtracks = []
    for _, r in tracks.iterrows():
        row = r.to_dict()
        lid = int(r.local_track_id)
        fi = int(r.frame_idx)
        tid = idf.lookup_tracklet(t_index, fi, lid) if lid >= 0 else None
        row["tracklet_id"] = tid
        if tid in role_map:
            row["role"] = role_map[tid]
            row["role_source"] = "human_override"
        else:
            row["role_source"] = "auto"
        p = tracklet_to_player.get(tid) if tid is not None else None
        if p is None or p.identity_status == "UNRESOLVED":
            row["global_player_id"] = None
            row["display_player_id"] = None
            row["identity_status"] = "UNRESOLVED"
            row["identity_confidence"] = 0.0
            row["human_verified"] = False
            row["jersey_number"] = None
        else:
            row["global_player_id"] = int(p.global_id)
            row["display_player_id"] = int(p.global_id)
            row["identity_status"] = p.identity_status
            row["identity_confidence"] = 0.9 if tid in human_verified_tids else (0.85 if p.identity_status == "CONFIRMED" else 0.55)
            row["human_verified"] = bool(tid in human_verified_tids)
            row["jersey_number"] = p.jersey_number
        gtracks.append(row)
    gdf = pd.DataFrame(gtracks)
    idf.assert_identity_invariants(gdf)
    gdf.to_parquet(OUT / "global_tracks_human_verified.parquet", index=False)

    summary = []
    for p in players:
        summary.append(
            {
                "global_player_id": p.global_id,
                "tracklet_ids": ",".join(map(str, p.tracklet_ids)),
                "team_id": p.team_id,
                "role": p.role,
                "identity_status": p.identity_status,
                "jersey_number": p.jersey_number,
                "human_verified": any(t in human_verified_tids for t in p.tracklet_ids),
                "n_tracklets": len(p.tracklet_ids),
            }
        )
    pd.DataFrame(summary).to_parquet(OUT / "identity_summary_human_verified.parquet", index=False)
    pd.DataFrame(merge_audit).to_parquet(OUT / "identity_merge_audit_human_verified.parquet", index=False)
    pd.DataFrame(reject_audit).to_parquet(OUT / "identity_reject_audit_human_verified.parquet", index=False)

    # switch candidates after
    switches = idf.mine_id_switches(gdf.assign(local_track_id=gdf["display_player_id"].fillna(-1).astype(int)))
    switches.to_parquet(OUT / "id_switch_candidates_after_review.parquet", index=False)

    log("Video render ediliyor")
    # custom render with checkmark
    _render_verified(gdf, OUT)

    log("2D video render ediliyor")
    # already included in _render_verified

    log("Testler çalışıyor")
    import subprocess

    proc = subprocess.run(
        [
            "/home/ahmet/miniconda3/envs/ai-dev/bin/python",
            "-m",
            "pytest",
            "-q",
            "--tb=line",
            "tests/evaluation/test_identity_manual_review.py",
            "tests/tracking/test_global_identity_finalization.py",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    (OUT / "apply_pytest.txt").write_text(proc.stdout + "\n" + proc.stderr)
    log(proc.stdout[-500:])

    report = f"""# Human Review Application Report

Decisions applied: {len(decisions)}
Must-links: {len(must_df)}
Cannot-links: {len(cannot_df)}
Role overrides: {len(role_df)}
Human SAME conflicts (not merged): {len(conflict_df)}
Global identities: {len(players)}
Trajectory break proxy after: {len(switches)}
reid_merge={cfg.reid_merge}
pytest_rc={proc.returncode}

Old identity results preserved at: {IDENTITY}
"""
    (OUT / "human_review_application_report.md").write_text(report)
    (OUT / "human_review_manifest.json").write_text(
        json.dumps(
            {
                "decisions": len(decisions),
                "must_links": len(must_df),
                "cannot_links": len(cannot_df),
                "role_overrides": len(role_df),
                "conflicts": len(conflict_df),
                "global_identities": len(players),
                "reid_merge": cfg.reid_merge,
                "pytest_rc": proc.returncode,
                "production_defaults_changed": 0,
            },
            indent=2,
        )
    )
    log("DONE human verified outputs written")


def _render_verified(tracks: pd.DataFrame, out_dir: Path) -> None:
    cap = cv2.VideoCapture(str(VIDEO))
    fps = float(cap.get(5) or 25)
    w, h = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(str(out_dir / "football_identity_human_verified.mp4"), fourcc, fps, (w, h))
    wr_dbg = cv2.VideoWriter(str(out_dir / "football_identity_human_verified_debug.mp4"), fourcc, fps, (w, h))
    tw = cv2.VideoWriter(str(out_dir / "football_tactical_2d_human_verified.mp4"), fourcc, fps, (1050, 720))

    def draw_pitch():
        img = np.zeros((720, 1050, 3), np.uint8)
        img[:] = (34, 110, 34)
        cv2.rectangle(img, (25, 25), (1025, 695), (255, 255, 255), 2)
        cv2.line(img, (525, 25), (525, 695), (255, 255, 255), 2)
        return img

    def pitch_px(x, y):
        return int(25 + (x / 105.0) * 1000), int(25 + (y / 68.0) * 670)

    pitch_bg = draw_pitch()
    by_frame = tracks.groupby("frame_idx")
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = frame.copy()
        dbg = frame.copy()
        pitch = pitch_bg.copy()
        if fi in by_frame.groups:
            for _, r in by_frame.get_group(fi).iterrows():
                if r.class_name == "ball":
                    continue
                x1, y1, x2, y2 = map(int, [r.smoothed_x1, r.smoothed_y1, r.smoothed_x2, r.smoothed_y2])
                role = str(r.role)
                team = int(r.team_id) if pd.notna(r.team_id) else -1
                gid = r.get("display_player_id")
                status = str(r.get("identity_status", "UNRESOLVED"))
                hv = bool(r.get("human_verified"))
                if "REF" in role.upper():
                    col, prefix = (0, 255, 255), "REF"
                elif team == 0:
                    col, prefix = (40, 40, 220), "T0"
                elif team == 1:
                    col, prefix = (220, 40, 40), "T1"
                else:
                    col, prefix = (180, 180, 180), "T?"
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                if pd.isna(gid) or status == "UNRESOLVED":
                    label = f"{prefix} | GID ?"
                else:
                    mark = " ✓" if hv else ""
                    if "REF" in role.upper():
                        label = f"REF | GID R{int(gid)}{mark}"
                    else:
                        jn = r.get("jersey_number")
                        jtxt = f"#{jn}" if pd.notna(jn) and str(jn) not in {"", "None"} else "#?"
                        label = f"{prefix} | GID {int(gid)} | {jtxt}{mark}"
                assert "LID" not in label
                cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                lid = int(r.local_track_id)
                tid = int(r.tracklet_id) if pd.notna(r.get("tracklet_id")) else -1
                gshow = "?" if pd.isna(gid) else str(int(gid))
                cv2.rectangle(dbg, (x1, y1), (x2, y2), col, 2)
                cv2.putText(
                    dbg,
                    f"LID {lid} | TID {tid} | GID {gshow}",
                    (x1, max(20, y1 - 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    dbg,
                    f"status={status} role_src={r.get('role_source')} hv={hv}",
                    (x1, max(40, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (200, 200, 0),
                    1,
                )
                if r.get("projection_valid") and pd.notna(r.get("filtered_pitch_x")) and not pd.isna(gid):
                    px, py = pitch_px(float(r.filtered_pitch_x), float(r.filtered_pitch_y))
                    cv2.circle(pitch, (px, py), 8, col, -1)
                    cv2.putText(pitch, str(int(gid)), (px + 6, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        wr.write(vis)
        wr_dbg.write(dbg)
        tw.write(pitch)
        fi += 1
    cap.release()
    wr.release()
    wr_dbg.release()
    tw.release()
    # side by side
    idf.side_by_side(
        out_dir / "football_identity_human_verified.mp4",
        out_dir / "football_tactical_2d_human_verified.mp4",
        out_dir / "football_identity_human_verified_side_by_side.mp4",
    )


if __name__ == "__main__":
    main()
