#!/usr/bin/env python3
"""Build a balanced 20-item identity review queue and pair/context clips."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("/home/ahmet/projects/football-analytics")
sys.path.insert(0, str(ROOT / "src"))

from football_analytics.tracking.global_tracklet_association import cosine  # noqa: E402

IDENTITY = Path("/mnt/c/football_data/results/tracking_identity_final")
VIDEO = Path("/mnt/c/football_data/videos/test_clips/football.mp4")
OUT = ROOT / "configs/evaluation/identity_review/football"
MEDIA = OUT / "media"


def _bytes_emb(b, dim: int) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).reshape(int(dim))


def load_prototypes() -> dict[int, np.ndarray]:
    df = pd.read_parquet(IDENTITY / "tracklet_reid_prototypes.parquet")
    out = {}
    for _, r in df.iterrows():
        out[int(r.tracklet_id)] = _bytes_emb(r.embedding_mean, int(r.dim))
    return out


def tracklet_lookup(tracklets: pd.DataFrame) -> dict[int, pd.Series]:
    return {int(r.tracklet_id): r for _, r in tracklets.iterrows()}


def local_to_tracklets_near(
    tracklets: pd.DataFrame, local_id: int, frame: int, prefer: str = "end"
) -> int | None:
    cands = tracklets[tracklets.local_track_id == int(local_id)]
    if cands.empty:
        return None
    if prefer == "end":
        cands = cands.assign(dist=(cands.end_frame - frame).abs())
    else:
        cands = cands.assign(dist=(cands.start_frame - frame).abs())
    return int(cands.sort_values("dist").iloc[0].tracklet_id)


def build_candidates() -> pd.DataFrame:
    tracklets = pd.read_parquet(IDENTITY / "tracklets.parquet")
    switches = pd.read_parquet(IDENTITY / "id_switch_candidates.parquet")
    merges = pd.read_parquet(IDENTITY / "identity_merge_audit.parquet")
    rejects = pd.read_parquet(IDENTITY / "identity_reject_audit.parquet")
    clusters = pd.read_parquet(IDENTITY / "global_identity_clusters.parquet")
    jersey = pd.read_parquet(IDENTITY / "jersey_tracklet_evidence.parquet")
    jmap = {int(r.tracklet_id): r for _, r in jersey.iterrows()}
    proto = load_prototypes()
    tmap = tracklet_lookup(tracklets)

    # map tracklet -> gid
    tid_gid = {}
    for _, c in clusters.iterrows():
        gid = int(c.global_player_id)
        for tid in str(c.tracklet_ids).split(","):
            if tid.strip():
                tid_gid[int(tid)] = gid

    rows = []

    def add_pair(
        ta: int,
        tb: int,
        review_type: str,
        selection_reason: str,
        impact: float,
        model_decision: str,
        model_confidence: float,
        association_score: float | None = None,
    ) -> None:
        if ta == tb or ta not in tmap or tb not in tmap:
            return
        a, b = tmap[ta], tmap[tb]
        # order by time
        if int(a.start_frame) > int(b.start_frame):
            a, b, ta, tb = b, a, tb, ta
        gap = max(0.0, (int(b.start_frame) - int(a.end_frame)) / 25.0)
        # pitch distance if available from switches later
        sim = cosine(proto.get(ta), proto.get(tb))
        ja = jmap.get(ta)
        jb = jmap.get(tb)
        rows.append(
            {
                "tracklet_a": ta,
                "tracklet_b": tb,
                "global_id_a": tid_gid.get(ta),
                "global_id_b": tid_gid.get(tb),
                "team_a": int(a.team_id) if int(a.team_id) >= 0 else None,
                "team_b": int(b.team_id) if int(b.team_id) >= 0 else None,
                "role_a": str(a.role),
                "role_b": str(b.role),
                "start_frame_a": int(a.start_frame),
                "end_frame_a": int(a.end_frame),
                "start_frame_b": int(b.start_frame),
                "end_frame_b": int(b.end_frame),
                "time_gap_seconds": float(gap),
                "pitch_distance_m": np.nan,
                "reid_similarity": float(sim) if sim is not None else np.nan,
                "jersey_a": None if ja is None else ja.get("jersey_number"),
                "jersey_b": None if jb is None else jb.get("jersey_number"),
                "association_score": association_score,
                "model_decision": model_decision,
                "model_confidence": float(model_confidence),
                "selection_reason": selection_reason,
                "impact_score": float(impact),
                "review_type": review_type,
                "duration_a": float(a.duration),
                "duration_b": float(b.duration),
            }
        )

    # 1) trajectory breaks / switches -> map to tracklets
    for _, s in switches.sort_values("confidence", ascending=False).head(80).iterrows():
        ta = local_to_tracklets_near(tracklets, int(s.old_local_id), int(s.start_frame), "end")
        tb = local_to_tracklets_near(tracklets, int(s.new_local_id), int(s.end_frame), "start")
        if ta is None or tb is None:
            continue
        ctype = str(s.candidate_type)
        if ctype == "short_gap_relink":
            rtype = "occlusion"
        elif ctype == "reentry_fragment":
            rtype = "reentry"
        else:
            rtype = "trajectory_break"
        impact = float(s.confidence) * (1.0 + min(float(s.get("time_gap_seconds") or 0), 5) / 5.0)
        # longer tracklets matter more
        da = float(tmap[ta].duration) if ta in tmap else 0
        db = float(tmap[tb].duration) if tb in tmap else 0
        impact *= 1.0 + min(da + db, 20) / 20.0
        add_pair(
            ta,
            tb,
            rtype,
            f"id_switch:{ctype}",
            impact,
            model_decision="DIFFERENT",  # currently separate local/global fragments
            model_confidence=float(s.confidence),
            association_score=None,
        )
        # fill pitch
        if rows:
            rows[-1]["pitch_distance_m"] = s.get("pitch_distance_m")

    # 2) accepted merges (esp low-ish confidence) — model says SAME
    for _, m in merges.iterrows():
        if m.decision != "merge":
            continue
        score = float(m.score) if pd.notna(m.score) else 0.5
        impact = 1.2 - min(score, 1.2) / 2.0  # lower score = higher review impact
        add_pair(
            int(m.tracklet_a),
            int(m.tracklet_b),
            "merged_low_conf" if score < 0.7 else "merged",
            f"merge:{m.reason}",
            impact + 0.5,
            "SAME",
            score,
            association_score=score,
        )

    # 3) high reid but rejected / not merged (near threshold)
    # approximate: pairs with high reid from prototypes that have different GIDs
    tids = list(proto.keys())
    for i, ta in enumerate(tids):
        if ta not in tmap:
            continue
        for tb in tids[i + 1 :]:
            if tb not in tmap:
                continue
            if tid_gid.get(ta) == tid_gid.get(tb) and tid_gid.get(ta) is not None:
                continue
            sim = cosine(proto[ta], proto[tb])
            if sim is None or sim < 0.72:
                continue
            a, b = tmap[ta], tmap[tb]
            if int(a.team_id) >= 0 and int(b.team_id) >= 0 and int(a.team_id) != int(b.team_id):
                continue
            # temporal non-overlap preferred
            if not (int(a.end_frame) < int(b.start_frame) or int(b.end_frame) < int(a.start_frame)):
                continue
            gap = abs(int(b.start_frame) - int(a.end_frame)) / 25.0
            if gap > 20:
                continue
            add_pair(
                ta,
                tb,
                "high_reid_split",
                "high_reid_different_gid",
                float(sim) * 1.5,
                "DIFFERENT",
                float(sim),
                association_score=float(sim),
            )

    # 4) role confusion candidates
    for _, s in switches[switches.role.str.contains("REF", case=False, na=False)].head(20).iterrows():
        ta = local_to_tracklets_near(tracklets, int(s.old_local_id), int(s.start_frame), "end")
        tb = local_to_tracklets_near(tracklets, int(s.new_local_id), int(s.end_frame), "start")
        if ta and tb:
            add_pair(ta, tb, "role_confusion", "referee_related_switch", 1.3, "DIFFERENT", 0.6)

    if not rows:
        raise RuntimeError("no review candidates built")

    df = pd.DataFrame(rows)
    # unique pairs
    df["pair_key"] = df.apply(lambda r: tuple(sorted((int(r.tracklet_a), int(r.tracklet_b)))), axis=1)
    df = df.sort_values("impact_score", ascending=False).drop_duplicates("pair_key")

    selected = []

    def take(mask, n, label_hint=None):
        sub = df[mask] if mask is not None else df
        for _, r in sub.iterrows():
            key = r.pair_key
            if any(key == s.pair_key for s in selected):
                continue
            selected.append(r)
            if len([s for s in selected if (label_hint is None) or True]) and len(
                [x for x in selected if True]
            ):
                pass
            # count by bucket externally
            if sum(1 for _ in selected if True) and False:
                break
            # stop when enough of this call
            # handled by caller checking len added
            yield r
            if sum(1 for __ in []) >= n:
                break

    buckets: list[pd.Series] = []

    def pick(cond, n: int) -> None:
        nonlocal buckets
        got = 0
        for _, r in df[cond].iterrows():
            if any(r.pair_key == b.pair_key for b in buckets):
                continue
            buckets.append(r)
            got += 1
            if got >= n:
                break

    # Balanced:
    # 8 likely SAME (merges / high impact same model)
    pick(df.model_decision == "SAME", 8)
    # 6 likely DIFFERENT (split high reid / trajectory)
    pick((df.model_decision == "DIFFERENT") & (df.review_type.isin(["trajectory_break", "high_reid_split", "gap_fragment", "merged_low_conf"]) == False), 6)
    # fill different from trajectory_break
    pick(df.review_type.isin(["trajectory_break", "high_reid_split"]), 6)
    # 3 occlusion
    pick(df.review_type == "occlusion", 3)
    # 3 reentry
    pick(df.review_type.isin(["reentry", "reentry_fragment"]), 3)

    # fill to 20 with highest impact remaining
    for _, r in df.iterrows():
        if len(buckets) >= 20:
            break
        if any(r.pair_key == b.pair_key for b in buckets):
            continue
        buckets.append(r)

    buckets = buckets[:20]
    if len(buckets) < 20:
        # emergency fill
        for _, r in df.iterrows():
            if len(buckets) >= 20:
                break
            if any(r.pair_key == b.pair_key for b in buckets):
                continue
            buckets.append(r)

    out_rows = []
    for i, r in enumerate(buckets, 1):
        rid = f"review_{i:03d}"
        d = r.to_dict()
        d["review_id"] = rid
        d["clip_a_path"] = str(MEDIA / f"{rid}_a.mp4")
        d["clip_b_path"] = str(MEDIA / f"{rid}_b.mp4")
        d["full_context_clip_path"] = str(MEDIA / f"{rid}_context.mp4")
        d["summary_jpg_path"] = str(MEDIA / f"{rid}_summary.jpg")
        d["reviewed"] = False
        d["human_decision"] = None
        d["review_timestamp"] = None
        out_rows.append(d)

    queue = pd.DataFrame(out_rows)
    # drop helper cols
    for c in ["pair_key", "duration_a", "duration_b"]:
        if c in queue.columns:
            queue = queue.drop(columns=[c])
    return queue


def _bbox_at(tracks: pd.DataFrame, local_id: int, frame: int) -> tuple[float, float, float, float] | None:
    g = tracks[(tracks.local_track_id == local_id) & (tracks.frame_idx == frame)]
    if g.empty:
        # nearest
        g = tracks[tracks.local_track_id == local_id]
        if g.empty:
            return None
        g = g.assign(d=(g.frame_idx - frame).abs()).sort_values("d")
        r = g.iloc[0]
    else:
        r = g.iloc[0]
    return float(r.smoothed_x1), float(r.smoothed_y1), float(r.smoothed_x2), float(r.smoothed_y2)


def render_tracklet_crop(
    video: Path,
    tracks: pd.DataFrame,
    local_id: int,
    start_f: int,
    end_f: int,
    out_mp4: Path,
    label: str,
    *,
    out_size: tuple[int, int] = (320, 480),
    pad: float = 0.25,
    max_seconds: float = 2.5,
    fps: float = 25.0,
) -> bool:
    cap = cv2.VideoCapture(str(video))
    vfps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    max_frames = int(max_seconds * vfps)
    span = list(range(int(start_f), int(end_f) + 1))
    if len(span) > max_frames:
        # center window
        mid = len(span) // 2
        half = max_frames // 2
        span = span[max(0, mid - half) : mid - half + max_frames]
    if not span:
        cap.release()
        return False

    # estimate mean box for stable crop window
    boxes = []
    for fi in span:
        b = _bbox_at(tracks, local_id, fi)
        if b:
            boxes.append(b)
    if not boxes:
        cap.release()
        return False
    x1 = float(np.median([b[0] for b in boxes]))
    y1 = float(np.median([b[1] for b in boxes]))
    x2 = float(np.median([b[2] for b in boxes]))
    y2 = float(np.median([b[3] for b in boxes]))
    bw, bh = x2 - x1, y2 - y1
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    side = max(bw, bh * 0.55) * (1.0 + pad)
    # keep aspect for out_size
    ow, oh = out_size
    aspect = ow / oh
    crop_h = side / aspect if side / aspect > side else side
    crop_w = crop_h * aspect
    # actually: height-driven
    crop_h = max(bh * (1 + pad), 120)
    crop_w = crop_h * aspect
    if crop_w < bw * (1 + pad):
        crop_w = bw * (1 + pad)
        crop_h = crop_w / aspect

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(str(out_mp4), fourcc, vfps, out_size)
    ok_any = False
    for fi in span:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        # follow box lightly
        b = _bbox_at(tracks, local_id, fi) or (x1, y1, x2, y2)
        bc = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
        cx = 0.7 * cx + 0.3 * bc[0]
        cy = 0.7 * cy + 0.3 * bc[1]
        xa = int(max(0, cx - crop_w / 2))
        ya = int(max(0, cy - crop_h / 2))
        xb = int(min(w, xa + crop_w))
        yb = int(min(h, ya + crop_h))
        xa = max(0, xb - int(crop_w))
        ya = max(0, yb - int(crop_h))
        crop = frame[ya:yb, xa:xb]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, out_size)
        cv2.putText(crop, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        wr.write(crop)
        ok_any = True
    wr.release()
    cap.release()
    return ok_any


def render_context(
    video: Path,
    tracks: pd.DataFrame,
    local_a: int,
    local_b: int,
    start_f: int,
    end_f: int,
    out_mp4: Path,
    *,
    max_seconds: float = 3.0,
) -> bool:
    cap = cv2.VideoCapture(str(video))
    vfps = float(cap.get(5) or 25)
    w, h = int(cap.get(3)), int(cap.get(4))
    max_frames = int(max_seconds * vfps)
    span = list(range(int(start_f), int(end_f) + 1))
    if len(span) > max_frames:
        mid = len(span) // 2
        half = max_frames // 2
        span = span[max(0, mid - half) : mid - half + max_frames]
    wr = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), vfps, (w, h))
    ok_any = False
    for fi in span:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        for lid, col, tag in [(local_a, (0, 255, 0), "A"), (local_b, (0, 165, 255), "B")]:
            b = _bbox_at(tracks, lid, fi)
            if not b:
                continue
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        wr.write(frame)
        ok_any = True
    wr.release()
    cap.release()
    return ok_any


def summary_jpg(clip_a: Path, clip_b: Path, out_jpg: Path) -> None:
    def mid(path: Path):
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
        ok, fr = cap.read()
        cap.release()
        if not ok:
            return np.zeros((480, 320, 3), np.uint8)
        return fr

    a, b = mid(clip_a), mid(clip_b)
    a, b = cv2.resize(a, (320, 480)), cv2.resize(b, (320, 480))
    cv2.imwrite(str(out_jpg), np.hstack([a, b]))


def generate_media(queue: pd.DataFrame) -> pd.DataFrame:
    MEDIA.mkdir(parents=True, exist_ok=True)
    tracks = pd.read_parquet(IDENTITY / "global_tracks.parquet")
    tracklets = pd.read_parquet(IDENTITY / "tracklets.parquet")
    tmap = tracklet_lookup(tracklets)
    rows = []
    for _, r in queue.iterrows():
        ta, tb = int(r.tracklet_a), int(r.tracklet_b)
        la, lb = int(tmap[ta].local_track_id), int(tmap[tb].local_track_id)
        sa, ea = int(r.start_frame_a), int(r.end_frame_a)
        sb, eb = int(r.start_frame_b), int(r.end_frame_b)
        path_a, path_b = Path(r.clip_a_path), Path(r.clip_b_path)
        path_c = Path(r.full_context_clip_path)
        path_j = Path(r.summary_jpg_path)
        ok_a = render_tracklet_crop(VIDEO, tracks, la, sa, ea, path_a, "A")
        ok_b = render_tracklet_crop(VIDEO, tracks, lb, sb, eb, path_b, "B")
        ctx_start = min(sa, sb)
        ctx_end = max(ea, eb)
        # keep context around junction
        mid = (int(r.end_frame_a) + int(r.start_frame_b)) // 2 if ea <= sb else (sa + eb) // 2
        ok_c = render_context(VIDEO, tracks, la, lb, max(0, mid - 30), mid + 30, path_c)
        if ok_a and ok_b:
            summary_jpg(path_a, path_b, path_j)
        d = r.to_dict()
        d["media_ok"] = bool(ok_a and ok_b and ok_c and path_j.exists())
        rows.append(d)
        print(f"media {r.review_id} a={ok_a} b={ok_b} c={ok_c}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    print("building candidate queue...", flush=True)
    queue = build_candidates()
    assert len(queue) == 20, len(queue)
    assert queue.review_id.nunique() == 20
    print("generating media clips...", flush=True)
    queue = generate_media(queue)
    if int(queue.media_ok.sum()) < 20:
        # still keep queue; mark missing
        print(f"WARNING media_ok={queue.media_ok.sum()}/20", flush=True)
    queue.to_parquet(OUT / "review_queue.parquet", index=False)
    queue.drop(columns=[c for c in queue.columns if c.startswith("model_")], errors="ignore")  # keep model in parquet but UI hides
    # write a UI-safe copy without exposing? Keep full parquet; UI must not display model_* before answer.
    meta = {
        "n_items": int(len(queue)),
        "with_pair_clips": int(((queue.apply(lambda r: Path(r.clip_a_path).exists() and Path(r.clip_b_path).exists(), axis=1))).sum()),
        "with_context_clips": int(queue.apply(lambda r: Path(r.full_context_clip_path).exists(), axis=1).sum()),
        "review_types": queue.review_type.value_counts().to_dict(),
    }
    (OUT / "queue_manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(f"wrote {OUT / 'review_queue.parquet'}", flush=True)


if __name__ == "__main__":
    main()
