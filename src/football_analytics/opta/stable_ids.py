"""Stable display IDs: map every local track to a persistent player number.

ByteTrack local IDs fragment when players move/occlude. Global identity merge
fixes many fragments; this module also attaches remaining tracks via ReID,
pixel-space proximity, and orphan-to-orphan chaining so overlay IDs stay
constant. Cross-team and simultaneous-overlap merges are forbidden.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from football_analytics.analytics.reid_matching import cosine_similarity


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float], *, tol_ms: float = 40.0) -> bool:
    return not (a[1] + tol_ms < b[0] or b[1] + tol_ms < a[0])


def _foot(row: Any) -> tuple[float, float]:
    if hasattr(row, "bbox_x1"):
        x1, x2, y2 = float(row.bbox_x1), float(row.bbox_x2), float(row.bbox_y2)
    else:
        x1, x2, y2 = float(row["bbox_x1"]), float(row["bbox_x2"]), float(row["bbox_y2"])
    return (0.5 * (x1 + x2), y2)


def _team_compatible(a: str | None, b: str | None) -> bool:
    if a and b and str(a) != str(b):
        return False
    return True


def _gap_and_dist(
    earlier_end: tuple[float, float],
    later_start: tuple[float, float],
    earlier_xy: tuple[float, float],
    later_xy: tuple[float, float],
) -> tuple[float, float] | None:
    """Return (gap_ms, dist_px) if later starts after earlier ends; else None."""
    gap = later_start[0] - earlier_end[1]
    if gap < 0:
        return None
    dist = float(
        np.hypot(later_xy[0] - earlier_xy[0], later_xy[1] - earlier_xy[1])
    )
    return gap, dist


def _proximity_ok(
    gap_ms: float,
    dist_px: float,
    *,
    proximity_gap_ms: float,
    proximity_dist_px: float,
    max_speed_px_s: float,
) -> bool:
    if gap_ms > proximity_gap_ms:
        return False
    if dist_px > proximity_dist_px:
        return False
    if gap_ms > 1.0:
        speed = dist_px / (gap_ms / 1000.0)
        if speed > max_speed_px_s:
            return False
    return True


def build_stable_display_map(
    tracks: pd.DataFrame,
    global_identity_map: pd.DataFrame | None,
    global_identity_report: pd.DataFrame | None,
    reid_prototypes: pd.DataFrame | None,
    identities: pd.DataFrame | None = None,
    *,
    reid_attach_threshold: float = 0.78,
    reid_relative_margin: float = 0.025,
    proximity_gap_ms: float = 6000.0,
    proximity_dist_px: float = 200.0,
    max_speed_px_s: float = 420.0,
    camera_id: str = "camera_1",
) -> pd.DataFrame:
    """Return rows: local_track_id, display_id, global_id, source, team_id, unresolved."""
    person = tracks[tracks["object_type"].eq("person")] if "object_type" in tracks.columns else tracks
    if person.empty:
        return pd.DataFrame(
            columns=[
                "camera_id",
                "local_track_id",
                "display_id",
                "global_id",
                "source",
                "team_id",
                "unresolved",
            ]
        )

    intervals: dict[int, tuple[float, float]] = {}
    start_xy: dict[int, tuple[float, float]] = {}
    end_xy: dict[int, tuple[float, float]] = {}
    for tid, g in person.groupby("track_id"):
        g = g.sort_values("timestamp_ms")
        tid_i = int(tid)
        intervals[tid_i] = (float(g["timestamp_ms"].min()), float(g["timestamp_ms"].max()))
        first = g.iloc[0]
        last = g.iloc[-1]
        start_xy[tid_i] = _foot(first)
        end_xy[tid_i] = _foot(last)

    team_by: dict[int, str | None] = {}
    if identities is not None and not identities.empty:
        for tid, g in identities.groupby("track_id"):
            assigned = g[g["team_id"].notna()] if "team_id" in g.columns else g.iloc[0:0]
            if assigned.empty:
                team_by[int(tid)] = None
            else:
                team_by[int(tid)] = str(assigned["team_id"].astype(str).mode().iloc[0])

    emb_by: dict[int, np.ndarray] = {}
    if reid_prototypes is not None and not reid_prototypes.empty:
        for row in reid_prototypes.itertuples(index=False):
            if hasattr(row, "valid") and not bool(row.valid):
                continue
            emb = getattr(row, "embedding", None)
            if emb is None:
                continue
            arr = np.asarray(list(emb), dtype=np.float64)
            if arr.size:
                emb_by[int(row.track_id)] = arr

    local_to_global: dict[int, int] = {}
    global_meta: dict[int, dict[str, Any]] = {}
    unresolved_globals: set[int] = set()
    source_by: dict[int, str] = {}

    if global_identity_map is not None and not global_identity_map.empty:
        for row in global_identity_map.itertuples(index=False):
            lid = int(row.local_track_id)
            gid = int(row.global_id)
            map_team = getattr(row, "team_id", None)
            local_team = team_by.get(lid)
            if map_team and local_team and str(map_team) != str(local_team):
                continue
            local_to_global[lid] = gid
            source_by[lid] = "global_identity"
            unresolved = bool(getattr(row, "unresolved", False))
            if unresolved:
                unresolved_globals.add(gid)
            meta = global_meta.setdefault(
                gid,
                {
                    "team_id": map_team or local_team,
                    "locals": [],
                    "intervals": [],
                    "embedding": None,
                },
            )
            meta["locals"].append(lid)
            if lid in intervals:
                meta["intervals"].append(intervals[lid])
            if local_team and not meta["team_id"]:
                meta["team_id"] = local_team

    if global_identity_report is not None and not global_identity_report.empty:
        for row in global_identity_report.itertuples(index=False):
            gid = int(row.global_player_id)
            if gid not in global_meta:
                continue
            if getattr(row, "team_id", None):
                global_meta[gid]["team_id"] = row.team_id

    def _refresh_embedding(gid: int) -> None:
        meta = global_meta[gid]
        vectors = [emb_by[lid] for lid in meta["locals"] if lid in emb_by]
        if vectors:
            mean = np.mean(np.stack(vectors), axis=0)
            norm = float(np.linalg.norm(mean))
            meta["embedding"] = mean / norm if norm > 1e-12 else mean
        else:
            meta["embedding"] = None

    for gid in list(global_meta):
        _refresh_embedding(gid)

    next_id = (max(global_meta.keys()) + 1) if global_meta else 1

    def _attach(lid: int, gid: int, source: str) -> None:
        local_to_global[lid] = gid
        source_by[lid] = source
        meta = global_meta[gid]
        meta["locals"].append(lid)
        meta["intervals"].append(intervals[lid])
        if team_by.get(lid) and not meta.get("team_id"):
            meta["team_id"] = team_by[lid]
        _refresh_embedding(gid)

    def _best_reid_candidate(lid: int) -> tuple[int | None, float, float]:
        """Return (gid, best_sim, second_sim)."""
        emb = emb_by.get(lid)
        if emb is None:
            return None, -1.0, -1.0
        team = team_by.get(lid)
        iv = intervals[lid]
        scored: list[tuple[float, int]] = []
        for gid, meta in global_meta.items():
            if meta.get("embedding") is None:
                continue
            if not _team_compatible(team, meta.get("team_id")):
                continue
            if any(_intervals_overlap(iv, other) for other in meta["intervals"]):
                continue
            sim = cosine_similarity(emb, meta["embedding"])
            if sim is None:
                continue
            scored.append((float(sim), gid))
        if not scored:
            return None, -1.0, -1.0
        scored.sort(reverse=True)
        best_sim, best_gid = scored[0]
        second = scored[1][0] if len(scored) > 1 else -1.0
        return best_gid, best_sim, second

    def _reid_accept(best: float, second: float, *, allow_unknown: bool) -> bool:
        if best < reid_attach_threshold:
            return False
        if second < 0:
            return True
        if best - second >= reid_relative_margin:
            return True
        # Unique-ish: large absolute score still ok when margin is tiny but best is strong.
        if allow_unknown and best >= reid_attach_threshold + 0.08 and best - second >= 0.01:
            return True
        return False

    # --- Pass 1: ReID attach onto existing globals ---
    for lid in sorted(set(intervals) - set(local_to_global)):
        team = team_by.get(lid)
        best_gid, best_sim, second = _best_reid_candidate(lid)
        allow_unknown = team is None
        if best_gid is not None and _reid_accept(best_sim, second, allow_unknown=allow_unknown):
            # Unknown may attach only when relatively unique.
            if team is None and second >= 0 and best_sim - second < reid_relative_margin:
                continue
            _attach(lid, best_gid, f"reid_attach:{best_sim:.3f}")

    def _best_proximity_candidate(lid: int) -> tuple[int | None, float]:
        team = team_by.get(lid)
        iv = intervals[lid]
        best_gid = None
        best_dist = 1e18
        for gid, meta in global_meta.items():
            if not _team_compatible(team, meta.get("team_id")):
                continue
            if any(_intervals_overlap(iv, other) for other in meta["intervals"]):
                continue
            for member in meta["locals"]:
                if member not in intervals:
                    continue
                # orphan after member
                fwd = _gap_and_dist(intervals[member], iv, end_xy[member], start_xy[lid])
                if fwd is not None:
                    gap, dist = fwd
                    if _proximity_ok(
                        gap,
                        dist,
                        proximity_gap_ms=proximity_gap_ms,
                        proximity_dist_px=proximity_dist_px,
                        max_speed_px_s=max_speed_px_s,
                    ) and dist < best_dist:
                        best_dist = dist
                        best_gid = gid
                # orphan before member
                back = _gap_and_dist(iv, intervals[member], end_xy[lid], start_xy[member])
                if back is not None:
                    gap, dist = back
                    if _proximity_ok(
                        gap,
                        dist,
                        proximity_gap_ms=proximity_gap_ms,
                        proximity_dist_px=proximity_dist_px,
                        max_speed_px_s=max_speed_px_s,
                    ) and dist < best_dist:
                        best_dist = dist
                        best_gid = gid
        return best_gid, best_dist

    # --- Pass 2: multi-round proximity onto existing globals ---
    for _ in range(4):
        remaining = sorted(
            set(intervals) - set(local_to_global),
            key=lambda lid: intervals[lid][0],
        )
        attached = 0
        for lid in remaining:
            best_gid, best_dist = _best_proximity_candidate(lid)
            if best_gid is not None:
                _attach(lid, best_gid, f"proximity:{best_dist:.1f}")
                attached += 1
        if attached == 0:
            break

    # --- Pass 3: orphan–orphan chaining (fixes the main multi-ID hole) ---
    orphans = sorted(set(intervals) - set(local_to_global), key=lambda lid: intervals[lid][0])
    if orphans:
        parent = {lid: lid for lid in orphans}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            # Prefer earlier-starting root for stable numbering.
            if intervals[ra][0] <= intervals[rb][0]:
                parent[rb] = ra
            else:
                parent[ra] = rb

        def chain_compatible(a: int, b: int) -> tuple[bool, str]:
            if not _team_compatible(team_by.get(a), team_by.get(b)):
                return False, ""
            if _intervals_overlap(intervals[a], intervals[b]):
                return False, ""
            # Order temporally
            if intervals[a][0] <= intervals[b][0]:
                earlier, later = a, b
            else:
                earlier, later = b, a
            gap = intervals[later][0] - intervals[earlier][1]
            dist = float(
                np.hypot(
                    start_xy[later][0] - end_xy[earlier][0],
                    start_xy[later][1] - end_xy[earlier][1],
                )
            )
            prox = _proximity_ok(
                gap,
                dist,
                proximity_gap_ms=proximity_gap_ms,
                proximity_dist_px=proximity_dist_px,
                max_speed_px_s=max_speed_px_s,
            )
            sim = None
            if a in emb_by and b in emb_by:
                sim = cosine_similarity(emb_by[a], emb_by[b])
            if prox and (sim is None or sim >= reid_attach_threshold - 0.08):
                return True, f"orphan_proximity:{dist:.1f}"
            if sim is not None and sim >= reid_attach_threshold:
                # Longer gap allowed when ReID is strong.
                if gap <= proximity_gap_ms * 1.5 and dist <= proximity_dist_px * 1.6:
                    return True, f"orphan_reid:{sim:.3f}"
                if gap <= proximity_gap_ms and sim >= reid_attach_threshold + 0.05:
                    return True, f"orphan_reid:{sim:.3f}"
            return False, ""

        # Greedy chronological chaining: each orphan links to best previous orphan.
        for i, lid in enumerate(orphans):
            best_prev = None
            best_score = 1e18
            best_reason = ""
            for prev in orphans[:i]:
                ok, reason = chain_compatible(prev, lid)
                if not ok:
                    continue
                # Prefer small distance; break ties with ReID.
                gap = abs(intervals[lid][0] - intervals[prev][1])
                dist = float(
                    np.hypot(
                        start_xy[lid][0] - end_xy[prev][0],
                        start_xy[lid][1] - end_xy[prev][1],
                    )
                )
                sim = 0.0
                if lid in emb_by and prev in emb_by:
                    s = cosine_similarity(emb_by[lid], emb_by[prev])
                    sim = float(s) if s is not None else 0.0
                score = dist - 40.0 * sim + 0.01 * gap
                if score < best_score:
                    best_score = score
                    best_prev = prev
                    best_reason = reason
            if best_prev is not None:
                # Ensure whole components stay temporally non-overlapping.
                ra, rb = find(best_prev), find(lid)
                members_a = [x for x in orphans if find(x) == ra]
                members_b = [x for x in orphans if find(x) == rb]
                conflict = False
                for x in members_a:
                    for y in members_b:
                        if x != y and _intervals_overlap(intervals[x], intervals[y]):
                            conflict = True
                            break
                    if conflict:
                        break
                if not conflict:
                    union(best_prev, lid)
                    source_by.setdefault(lid, best_reason)
                    source_by.setdefault(best_prev, best_reason)

        # Materialize chains as new globals (or attach into existing if ReID matches).
        roots: dict[int, list[int]] = {}
        for lid in orphans:
            roots.setdefault(find(lid), []).append(lid)

        for _root, members in sorted(roots.items(), key=lambda kv: intervals[kv[1][0]][0]):
            members = sorted(members, key=lambda lid: intervals[lid][0])
            # Try attach whole chain onto an existing global via ReID of any member.
            attached_gid = None
            attach_sim = -1.0
            for lid in members:
                if lid in local_to_global:
                    continue
                gid, sim, second = _best_reid_candidate(lid)
                if gid is not None and _reid_accept(sim, second, allow_unknown=True):
                    # Verify no overlap with that global for ALL members.
                    meta = global_meta[gid]
                    if any(
                        _intervals_overlap(intervals[m], other)
                        for m in members
                        for other in meta["intervals"]
                    ):
                        continue
                    if not _team_compatible(team_by.get(lid), meta.get("team_id")):
                        continue
                    if sim > attach_sim:
                        attach_sim = sim
                        attached_gid = gid
            if attached_gid is not None:
                for lid in members:
                    if lid not in local_to_global:
                        _attach(lid, attached_gid, f"chain_reid:{attach_sim:.3f}")
                continue

            # New global for the chain
            gid = next_id
            next_id += 1
            team = next((team_by.get(m) for m in members if team_by.get(m)), None)
            global_meta[gid] = {
                "team_id": team,
                "locals": [],
                "intervals": [],
                "embedding": None,
            }
            unresolved_globals.add(gid)
            for lid in members:
                reason = source_by.get(lid, "orphan_chain")
                if reason == "orphan_chain" and len(members) == 1:
                    reason = "singleton_local"
                elif reason.startswith("orphan_"):
                    reason = f"orphan_chain:{reason}"
                else:
                    reason = f"orphan_chain:{len(members)}"
                _attach(lid, gid, reason)

    # --- Pass 4: soft collapse unresolved globals into each other / preferred ---
    def _global_span(gid: int) -> tuple[float, float]:
        ivs = global_meta[gid]["intervals"]
        return min(i[0] for i in ivs), max(i[1] for i in ivs)

    def _can_merge_globals(a: int, b: int) -> tuple[bool, float, str]:
        ma, mb = global_meta[a], global_meta[b]
        if not _team_compatible(ma.get("team_id"), mb.get("team_id")):
            return False, 1e18, ""
        for ia in ma["intervals"]:
            for ib in mb["intervals"]:
                if _intervals_overlap(ia, ib):
                    return False, 1e18, ""
        # End of earlier → start of later using extreme members
        a_end_lid = max(ma["locals"], key=lambda lid: intervals[lid][1])
        b_start_lid = min(mb["locals"], key=lambda lid: intervals[lid][0])
        b_end_lid = max(mb["locals"], key=lambda lid: intervals[lid][1])
        a_start_lid = min(ma["locals"], key=lambda lid: intervals[lid][0])
        candidates = []
        for earlier_lid, later_lid in (
            (a_end_lid, b_start_lid),
            (b_end_lid, a_start_lid),
        ):
            gap = intervals[later_lid][0] - intervals[earlier_lid][1]
            if gap < 0:
                continue
            dist = float(
                np.hypot(
                    start_xy[later_lid][0] - end_xy[earlier_lid][0],
                    start_xy[later_lid][1] - end_xy[earlier_lid][1],
                )
            )
            candidates.append((gap, dist))
        if not candidates:
            return False, 1e18, ""
        gap, dist = min(candidates, key=lambda x: x[1])
        sim = None
        if ma.get("embedding") is not None and mb.get("embedding") is not None:
            sim = cosine_similarity(ma["embedding"], mb["embedding"])
        prox = _proximity_ok(
            gap,
            dist,
            proximity_gap_ms=proximity_gap_ms * 1.25,
            proximity_dist_px=proximity_dist_px * 1.25,
            max_speed_px_s=max_speed_px_s,
        )
        if prox and (sim is None or float(sim) >= reid_attach_threshold - 0.10):
            return True, dist, f"collapse_proximity:{dist:.1f}"
        if sim is not None and float(sim) >= reid_attach_threshold:
            if gap <= proximity_gap_ms * 1.75:
                return True, dist, f"collapse_reid:{float(sim):.3f}"
        return False, 1e18, ""

    def _absorb(src: int, dst: int, reason: str) -> None:
        """Merge global src into dst."""
        if src == dst or src not in global_meta or dst not in global_meta:
            return
        for lid in list(global_meta[src]["locals"]):
            local_to_global[lid] = dst
            source_by[lid] = reason
            global_meta[dst]["locals"].append(lid)
            if lid in intervals:
                global_meta[dst]["intervals"].append(intervals[lid])
        if global_meta[src].get("team_id") and not global_meta[dst].get("team_id"):
            global_meta[dst]["team_id"] = global_meta[src]["team_id"]
        _refresh_embedding(dst)
        unresolved_globals.discard(src)
        del global_meta[src]

    # Prefer collapsing unresolved → resolved first, then unresolved↔unresolved.
    changed = True
    rounds = 0
    while changed and rounds < 6:
        changed = False
        rounds += 1
        unresolved = sorted(g for g in global_meta if g in unresolved_globals)
        resolved = sorted(g for g in global_meta if g not in unresolved_globals)
        # unresolved into resolved
        for src in list(unresolved):
            if src not in global_meta:
                continue
            best_dst = None
            best_dist = 1e18
            best_reason = ""
            for dst in resolved:
                ok, dist, reason = _can_merge_globals(src, dst)
                if ok and dist < best_dist:
                    best_dist = dist
                    best_dst = dst
                    best_reason = reason
            if best_dst is not None:
                _absorb(src, best_dst, best_reason)
                changed = True
        # unresolved into unresolved
        unresolved = sorted(g for g in global_meta if g in unresolved_globals)
        for i, src in enumerate(unresolved):
            if src not in global_meta:
                continue
            best_dst = None
            best_dist = 1e18
            best_reason = ""
            for dst in unresolved:
                if dst <= src or dst not in global_meta:
                    continue
                ok, dist, reason = _can_merge_globals(src, dst)
                if ok and dist < best_dist:
                    best_dist = dist
                    best_dst = dst
                    best_reason = reason
            if best_dst is not None:
                # Absorb later into earlier span for stable ids
                sa, sb = _global_span(src), _global_span(best_dst)
                if sa[0] <= sb[0]:
                    _absorb(best_dst, src, best_reason)
                else:
                    _absorb(src, best_dst, best_reason)
                changed = True

    # Compact display ids: multi-fragment / resolved players first
    preferred = sorted(
        (gid for gid in global_meta if gid not in unresolved_globals),
        key=lambda g: (-len(global_meta[g]["locals"]), g),
    )
    others = sorted(
        (gid for gid in global_meta if gid in unresolved_globals),
        key=lambda g: (-len(global_meta[g]["locals"]), g),
    )
    display_of_global: dict[int, int] = {}
    display_n = 1
    for gid in preferred + others:
        display_of_global[gid] = display_n
        display_n += 1

    rows = []
    for lid, gid in sorted(local_to_global.items()):
        if gid not in global_meta:
            continue
        rows.append(
            {
                "camera_id": camera_id,
                "local_track_id": int(lid),
                "display_id": int(display_of_global[gid]),
                "global_id": int(gid),
                "source": source_by.get(lid, "unknown"),
                "team_id": global_meta.get(gid, {}).get("team_id") or team_by.get(lid),
                "unresolved": bool(gid in unresolved_globals),
            }
        )
    return pd.DataFrame(rows)


def display_id_lookup(stable_map: pd.DataFrame) -> dict[int, int]:
    if stable_map is None or stable_map.empty:
        return {}
    return {
        int(row.local_track_id): int(row.display_id)
        for row in stable_map.itertuples(index=False)
    }


def team_lookup(stable_map: pd.DataFrame) -> dict[int, str]:
    if stable_map is None or stable_map.empty:
        return {}
    out: dict[int, str] = {}
    for row in stable_map.itertuples(index=False):
        if row.team_id is None or (isinstance(row.team_id, float) and np.isnan(row.team_id)):
            continue
        out[int(row.local_track_id)] = str(row.team_id)
    return out
