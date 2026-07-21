"""Online overlay display slots — one persistent number per on-field person.

ByteTrack + offline ReID still fragment. For the annotated video we maintain
per-team slots matched frame-to-frame by foot position so the same person
keeps ``P#`` even when local track_id changes. Inactive slots are reclaimed
before minting new IDs, then a final non-overlap collapse merges fragments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class _Slot:
    slot_id: int
    team_id: str
    cx: float
    cy: float
    age: int = 0
    hits: int = 1
    last_track_id: int | None = None
    members: set[int] = field(default_factory=set)


def _foot_xy(row: Any) -> tuple[float, float]:
    return (
        0.5 * (float(row.bbox_x1) + float(row.bbox_x2)),
        float(row.bbox_y2),
    )


def _collapse_display_ids(
    frame_map: pd.DataFrame,
    *,
    max_gap_frames: int = 180,
    max_bridge_px: float = 220.0,
) -> pd.DataFrame:
    """Merge display IDs that never co-occur and are temporally/spatially continuous."""
    if frame_map.empty:
        return frame_map
    out = frame_map.copy()
    parent: dict[int, int] = {int(d): int(d) for d in out["display_id"].unique()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Co-visibility: cannot merge if same frame.
    for _fid, g in out.groupby("frame_id"):
        ids = [int(x) for x in g["display_id"].unique()]
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                # mark conflict via sentinel later — we just skip pairs that co-occur
                pass

    co_visible: set[tuple[int, int]] = set()
    for _fid, g in out.groupby("frame_id"):
        ids = sorted({int(x) for x in g["display_id"].tolist()})
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                co_visible.add((a, b))

    # Per-display timeline endpoints
    meta: dict[int, dict[str, Any]] = {}
    for did, g in out.groupby("display_id"):
        g = g.sort_values("frame_id")
        first = g.iloc[0]
        last = g.iloc[-1]
        team = str(g["team_id"].astype(str).mode().iloc[0])
        meta[int(did)] = {
            "team": team,
            "f0": int(g["frame_id"].min()),
            "f1": int(g["frame_id"].max()),
            "start": (
                0.5 * (float(first["bbox_x1"]) + float(first["bbox_x2"]))
                if "bbox_x1" in g.columns
                else None
            ),
            "end": None,
        }
    # Need positions: join from original if present
    if not {"bbox_x1", "bbox_x2", "bbox_y2"}.issubset(out.columns):
        # Only frame_id/track_id/display/team — skip spatial collapse of endpoints
        # Fall back to team-wise temporal chain without distance using track continuity only.
        for team, ids in out.groupby("team_id")["display_id"].apply(lambda s: sorted(set(int(x) for x in s))).items():
            spans = sorted(
                ((meta[i]["f0"], meta[i]["f1"], i) for i in ids if i in meta),
                key=lambda x: x[0],
            )
            for i in range(1, len(spans)):
                f0a, f1a, a = spans[i - 1]
                f0b, f1b, b = spans[i]
                if (min(a, b), max(a, b)) in co_visible:
                    continue
                gap = f0b - f1a
                if 0 <= gap <= max_gap_frames:
                    # Only merge if no other id overlaps the bridge awkwardly — keep simple
                    union(a, b)
    else:
        for did, g in out.groupby("display_id"):
            g = g.sort_values("frame_id")
            first, last = g.iloc[0], g.iloc[-1]
            meta[int(did)]["start_xy"] = (
                0.5 * (float(first.bbox_x1) + float(first.bbox_x2)),
                float(first.bbox_y2),
            )
            meta[int(did)]["end_xy"] = (
                0.5 * (float(last.bbox_x1) + float(last.bbox_x2)),
                float(last.bbox_y2),
            )

        for team in out["team_id"].unique():
            ids = [int(d) for d in out.loc[out["team_id"] == team, "display_id"].unique()]
            spans = sorted(((meta[i]["f0"], meta[i]["f1"], i) for i in ids), key=lambda x: x[0])
            for i, (f0a, f1a, a) in enumerate(spans):
                best = None
                best_score = 1e18
                for f0b, f1b, b in spans[i + 1 :]:
                    if f0b - f1a > max_gap_frames:
                        break
                    if f0b < f1a:
                        continue
                    key = (min(a, b), max(a, b))
                    if key in co_visible:
                        continue
                    gap = f0b - f1a
                    dist = float(
                        np.hypot(
                            meta[b]["start_xy"][0] - meta[a]["end_xy"][0],
                            meta[b]["start_xy"][1] - meta[a]["end_xy"][1],
                        )
                    )
                    if dist <= max_bridge_px and gap <= max_gap_frames:
                        score = dist + 0.05 * gap
                        if score < best_score:
                            best_score = score
                            best = b
                if best is not None:
                    union(a, best)

    out["display_id"] = out["display_id"].map(lambda d: find(int(d)))

    # Soft roster: if a team still has too many IDs, merge shortest into nearest
    # non-co-visible same-team ID.
    for team in list(out["team_id"].unique()):
        while True:
            ids = [int(d) for d in out.loc[out["team_id"] == team, "display_id"].unique()]
            if len(ids) <= 13:
                break
            spans = []
            for did in ids:
                g = out[out["display_id"] == did]
                spans.append((len(g), int(g["frame_id"].min()), int(g["frame_id"].max()), did))
            spans.sort()  # shortest first
            short_len, f0, f1, short_id = spans[0]
            # rebuild co-visibility for current ids
            co: set[tuple[int, int]] = set()
            for _fid, g in out.groupby("frame_id"):
                present = sorted({int(x) for x in g["display_id"].tolist() if int(x) in ids})
                for i, a in enumerate(present):
                    for b in present[i + 1 :]:
                        co.add((a, b))
            # endpoint of short
            g_short = out[out["display_id"] == short_id].sort_values("frame_id")
            if "bbox_x1" not in out.columns:
                break
            s_first, s_last = g_short.iloc[0], g_short.iloc[-1]
            s_start = (0.5 * (float(s_first.bbox_x1) + float(s_first.bbox_x2)), float(s_first.bbox_y2))
            s_end = (0.5 * (float(s_last.bbox_x1) + float(s_last.bbox_x2)), float(s_last.bbox_y2))
            best = None
            best_score = 1e18
            for _ln, a0, a1, other in spans[1:]:
                key = (min(short_id, other), max(short_id, other))
                if key in co:
                    continue
                g_o = out[out["display_id"] == other].sort_values("frame_id")
                o_first, o_last = g_o.iloc[0], g_o.iloc[-1]
                o_start = (0.5 * (float(o_first.bbox_x1) + float(o_first.bbox_x2)), float(o_first.bbox_y2))
                o_end = (0.5 * (float(o_last.bbox_x1) + float(o_last.bbox_x2)), float(o_last.bbox_y2))
                # try both orders
                for gap, dist in (
                    (a0 - f1, float(np.hypot(o_start[0] - s_end[0], o_start[1] - s_end[1]))),
                    (f0 - a1, float(np.hypot(s_start[0] - o_end[0], s_start[1] - o_end[1]))),
                ):
                    if gap < 0:
                        continue
                    score = dist + 0.02 * gap
                    if score < best_score:
                        best_score = score
                        best = other
            if best is None:
                # force merge into longest even if weak
                best = spans[-1][3]
                if best == short_id:
                    break
            out.loc[out["display_id"] == short_id, "display_id"] = best

    return out


def build_overlay_slot_assignments(
    tracks: pd.DataFrame,
    identities: pd.DataFrame | None,
    *,
    max_slots_per_team: int = 12,
    match_dist_px: float = 200.0,
    hold_frames: int = 240,
    camera_id: str = "camera_1",
    stable_map: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign stable overlay display IDs via online per-team slots."""
    person = (
        tracks[tracks["object_type"].eq("person")].copy()
        if "object_type" in tracks.columns
        else tracks.copy()
    )
    if person.empty:
        empty_f = pd.DataFrame(columns=["frame_id", "track_id", "display_id", "team_id"])
        empty_t = pd.DataFrame(
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
        return empty_f, empty_t

    team_by: dict[int, str] = {}
    if identities is not None and not identities.empty and "team_id" in identities.columns:
        for tid, g in identities.groupby("track_id"):
            valid = g[g["team_id"].notna()]
            if valid.empty:
                continue
            team_by[int(tid)] = str(valid["team_id"].astype(str).mode().iloc[0])

    # Optional seed: prefer previously resolved stable display ids as soft labels.
    seed_display: dict[int, int] = {}
    if stable_map is not None and not stable_map.empty:
        for row in stable_map.itertuples(index=False):
            seed_display[int(row.local_track_id)] = int(row.display_id)

    def team_of(tid: int) -> str:
        return team_by.get(int(tid), "unknown")

    next_slot = 1
    slots_by_team: dict[str, list[_Slot]] = {"team_0": [], "team_1": [], "unknown": []}
    # Map seed display → slot when first seen
    seed_to_slot: dict[int, int] = {}
    frame_rows: list[dict[str, Any]] = []
    track_votes: dict[int, list[int]] = {}
    track_team_seen: dict[int, str] = {}

    for frame_id, group in person.sort_values(["frame_id", "track_id"]).groupby("frame_id", sort=True):
        detections: list[dict[str, Any]] = []
        for row in group.itertuples(index=False):
            tid = int(row.track_id)
            cx, cy = _foot_xy(row)
            detections.append(
                {
                    "track_id": tid,
                    "team_id": team_of(tid),
                    "cx": cx,
                    "cy": cy,
                    "seed": seed_display.get(tid),
                    "bbox_x1": float(row.bbox_x1),
                    "bbox_y1": float(row.bbox_y1),
                    "bbox_x2": float(row.bbox_x2),
                    "bbox_y2": float(row.bbox_y2),
                }
            )

        for slots in slots_by_team.values():
            for slot in slots:
                slot.age += 1

        assigned_slot_ids: set[int] = set()
        for team in ("team_0", "team_1", "unknown"):
            team_dets = [d for d in detections if d["team_id"] == team]
            # Keep slots within hold window
            slots = [s for s in slots_by_team[team] if s.age <= hold_frames]
            slots_by_team[team] = slots

            if team_dets and slots:
                costs = np.full((len(team_dets), len(slots)), 1e9, dtype=np.float64)
                for i, det in enumerate(team_dets):
                    for j, slot in enumerate(slots):
                        dist = float(np.hypot(det["cx"] - slot.cx, det["cy"] - slot.cy))
                        if slot.last_track_id == det["track_id"]:
                            costs[i, j] = max(0.0, dist * 0.2 - 100.0)
                            continue
                        # Soft preference for same seeded stable id
                        seed_bonus = 0.0
                        if det["seed"] is not None and seed_to_slot.get(int(det["seed"])) == slot.slot_id:
                            seed_bonus = -40.0
                        gate = match_dist_px * (1.7 if slot.age <= 2 else 1.15)
                        cost = dist + 0.12 * slot.age + seed_bonus
                        if dist <= gate:
                            costs[i, j] = cost

                matched_d: set[int] = set()
                matched_s: set[int] = set()
                pairs = [
                    (float(costs[i, j]), i, j)
                    for i in range(len(team_dets))
                    for j in range(len(slots))
                    if costs[i, j] < 1e8
                ]
                pairs.sort()
                for _cost, i, j in pairs:
                    if i in matched_d or j in matched_s:
                        continue
                    slot = slots[j]
                    if slot.slot_id in assigned_slot_ids:
                        continue
                    matched_d.add(i)
                    matched_s.add(j)
                    det = team_dets[i]
                    slot.cx = 0.6 * det["cx"] + 0.4 * slot.cx
                    slot.cy = 0.6 * det["cy"] + 0.4 * slot.cy
                    slot.age = 0
                    slot.hits += 1
                    slot.last_track_id = det["track_id"]
                    slot.members.add(det["track_id"])
                    assigned_slot_ids.add(slot.slot_id)
                    det["display_id"] = slot.slot_id
                    if det["seed"] is not None:
                        seed_to_slot.setdefault(int(det["seed"]), slot.slot_id)

                unmatched = [team_dets[i] for i in range(len(team_dets)) if i not in matched_d]
            else:
                unmatched = team_dets

            for det in unmatched:
                # 1) Reclaim seeded slot if it exists
                if det["seed"] is not None and int(det["seed"]) in seed_to_slot:
                    sid = seed_to_slot[int(det["seed"])]
                    existing = next((s for s in slots_by_team[team] if s.slot_id == sid), None)
                    if existing is not None and existing.slot_id not in assigned_slot_ids:
                        # Ensure not used by another detection this frame
                        existing.cx = det["cx"]
                        existing.cy = det["cy"]
                        existing.age = 0
                        existing.hits += 1
                        existing.last_track_id = det["track_id"]
                        existing.members.add(det["track_id"])
                        assigned_slot_ids.add(existing.slot_id)
                        det["display_id"] = existing.slot_id
                        continue

                # 2) Reclaim nearest inactive/active unused slot
                candidates = [s for s in slots_by_team[team] if s.slot_id not in assigned_slot_ids]
                if candidates:
                    best = min(
                        candidates,
                        key=lambda s: float(np.hypot(det["cx"] - s.cx, det["cy"] - s.cy))
                        + (0.0 if s.age > 0 else 15.0),
                    )
                    dist = float(np.hypot(det["cx"] - best.cx, det["cy"] - best.cy))
                    reclaim_gate = match_dist_px * (2.4 if best.age > 0 else 1.3)
                    active_now = sum(1 for s in slots_by_team[team] if s.age == 0 or s.slot_id in assigned_slot_ids)
                    if dist <= reclaim_gate or active_now >= max_slots_per_team:
                        best.cx = 0.65 * det["cx"] + 0.35 * best.cx
                        best.cy = 0.65 * det["cy"] + 0.35 * best.cy
                        best.age = 0
                        best.hits += 1
                        best.last_track_id = det["track_id"]
                        best.members.add(det["track_id"])
                        assigned_slot_ids.add(best.slot_id)
                        det["display_id"] = best.slot_id
                        if det["seed"] is not None:
                            seed_to_slot.setdefault(int(det["seed"]), best.slot_id)
                        continue

                # 3) Mint new slot
                slot = _Slot(
                    slot_id=next_slot,
                    team_id=team,
                    cx=det["cx"],
                    cy=det["cy"],
                    age=0,
                    last_track_id=det["track_id"],
                    members={det["track_id"]},
                )
                next_slot += 1
                slots_by_team[team].append(slot)
                assigned_slot_ids.add(slot.slot_id)
                det["display_id"] = slot.slot_id
                if det["seed"] is not None:
                    seed_to_slot.setdefault(int(det["seed"]), slot.slot_id)

        for det in detections:
            display = int(det["display_id"])
            tid = int(det["track_id"])
            frame_rows.append(
                {
                    "frame_id": int(frame_id),
                    "track_id": tid,
                    "display_id": display,
                    "team_id": det["team_id"],
                    "bbox_x1": det["bbox_x1"],
                    "bbox_y1": det["bbox_y1"],
                    "bbox_x2": det["bbox_x2"],
                    "bbox_y2": det["bbox_y2"],
                }
            )
            track_votes.setdefault(tid, []).append(display)
            track_team_seen[tid] = det["team_id"]

    frame_map = pd.DataFrame(frame_rows)
    # Collapse fragmented slot ids.
    frame_map = _collapse_display_ids(frame_map, max_gap_frames=hold_frames, max_bridge_px=match_dist_px * 1.2)

    track_rows: list[dict[str, Any]] = []
    for tid, votes in sorted(track_votes.items()):
        # Recompute votes from collapsed frame_map
        sub = frame_map.loc[frame_map["track_id"] == tid, "display_id"]
        if sub.empty:
            continue
        vals, counts = np.unique(sub.to_numpy(), return_counts=True)
        display = int(vals[int(np.argmax(counts))])
        track_rows.append(
            {
                "camera_id": camera_id,
                "local_track_id": int(tid),
                "display_id": display,
                "global_id": display,
                "source": "overlay_slot",
                "team_id": track_team_seen.get(tid),
                "unresolved": False,
            }
        )
    track_map = pd.DataFrame(track_rows)

    if not frame_map.empty:
        order = (
            frame_map.sort_values(["frame_id", "display_id"])
            .drop_duplicates("display_id")["display_id"]
            .tolist()
        )
        remap = {old: i + 1 for i, old in enumerate(order)}
        frame_map["display_id"] = frame_map["display_id"].map(remap)
        track_map["display_id"] = track_map["display_id"].map(remap)
        track_map["global_id"] = track_map["display_id"]
        # Drop bbox helpers from exported frame map
        frame_map = frame_map[["frame_id", "track_id", "display_id", "team_id"]]

    return frame_map, track_map


def frame_display_lookup(frame_map: pd.DataFrame) -> dict[tuple[int, int], int]:
    if frame_map is None or frame_map.empty:
        return {}
    return {
        (int(r.frame_id), int(r.track_id)): int(r.display_id)
        for r in frame_map.itertuples(index=False)
    }
