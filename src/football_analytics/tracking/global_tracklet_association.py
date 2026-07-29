"""Offline constrained global tracklet association for football identity.

No team-size hard-caps. Prefer UNRESOLVED / separate IDs over unsafe merges.
Generic pedestrian ReID (OSNet Market1501) is treated as weak evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return None
    return float(np.dot(a, b) / (na * nb))


@dataclass
class TrackletIdentity:
    tracklet_id: int
    local_track_id: int
    class_name: str  # player | referee | goalkeeper | staff | ...
    role: str
    team_id: int | None
    team_confidence: float
    start_frame: int
    end_frame: int
    embedding: np.ndarray | None
    embedding_variance: float
    valid_reid_crop_count: int
    jersey_number: str | None
    jersey_status: str  # CONFIRMED | PROVISIONAL | UNREADABLE | CONFLICTING
    jersey_confidence: float
    jersey_support_count: int = 0
    jersey_conflict_count: int = 0
    shoe_hsv: tuple[float, float, float] | None = None
    sock_hsv: tuple[float, float, float] | None = None
    shoe_status: str = "UNAVAILABLE"
    shoe_confidence: float = 0.0
    mean_pitch: tuple[float, float] | None = None
    start_pitch: tuple[float, float] | None = None
    end_pitch: tuple[float, float] | None = None
    mean_velocity: tuple[float, float] | None = None
    duration_s: float = 0.0
    quality_score: float = 0.0
    quality: str = "MEDIUM"
    mean_conf: float = 0.0
    shot_ids: list[str] = field(default_factory=list)
    scene_ids: list[int] = field(default_factory=list)


@dataclass
class GlobalPlayerState:
    global_id: int
    tracklet_ids: list[int] = field(default_factory=list)
    local_track_ids: list[int] = field(default_factory=list)
    team_id: int | None = None
    class_name: str = "player"
    role: str = "PLAYER"
    intervals: list[tuple[int, int]] = field(default_factory=list)
    embedding: np.ndarray | None = None
    jersey_number: str | None = None
    jersey_status: str = "UNREADABLE"
    jersey_confidence: float = 0.0
    shoe_hsv: tuple[float, float, float] | None = None
    identity_status: str = "PROVISIONAL"  # PENDING | PROVISIONAL | CONFIRMED | UNRESOLVED
    evidence: list[str] = field(default_factory=list)
    end_pitch: tuple[float, float] | None = None


@dataclass
class AssocConfig:
    reid_merge: float = 0.52
    reid_strong: float = 0.62
    reid_min_crops: int = 2
    max_gap_s: float = 8.0
    max_gap_strong_s: float = 18.0
    max_gap_short_s: float = 2.0
    max_speed_mps: float = 11.0
    short_gap_pitch_m: float = 8.0
    min_merge_score: float = 0.55
    shoe_aux_bonus: float = 0.03  # never sufficient alone
    jersey_match_bonus: float = 0.15
    require_reid_or_jersey: bool = True
    fps: float = 25.0


def _overlap(a: tuple[int, int], b: tuple[int, int], tol: int = 0) -> bool:
    return not (a[1] + tol < b[0] or b[1] + tol < a[0])


def _any_overlap(intervals: list[tuple[int, int]], other: tuple[int, int], tol: int = 0) -> bool:
    return any(_overlap(iv, other, tol=tol) for iv in intervals)


def _role_family(role: str, class_name: str) -> str:
    r = (role or class_name or "").upper()
    if "REF" in r:
        return "REFEREE"
    if "STAFF" in r or "BENCH" in r or "SPECTATOR" in r:
        return "STAFF"
    if "GK" in r or "GOALKEEPER" in r:
        return "GOALKEEPER"
    if "PLAYER" in r or class_name == "player":
        return "PLAYER"
    return "OTHER"


def _hsv_close(a, b, thr=(18.0, 55.0, 55.0)) -> bool:
    if a is None or b is None:
        return False
    dh = min(abs(a[0] - b[0]), 180 - abs(a[0] - b[0]))
    return dh <= thr[0] and abs(a[1] - b[1]) <= thr[1] and abs(a[2] - b[2]) <= thr[2]


def pairwise_veto(
    a: TrackletIdentity,
    b: TrackletIdentity,
    *,
    config: AssocConfig,
) -> str | None:
    """Return veto reason or None if merge is allowed to be scored."""
    if a.tracklet_id == b.tracklet_id:
        return "same_tracklet"
    if _overlap((a.start_frame, a.end_frame), (b.start_frame, b.end_frame), tol=0):
        return "simultaneous_overlap"
    fa, fb = _role_family(a.role, a.class_name), _role_family(b.role, b.class_name)
    if fa != fb and {fa, fb} <= {"PLAYER", "REFEREE", "STAFF", "GOALKEEPER"}:
        if {fa, fb} == {"PLAYER", "GOALKEEPER"}:
            pass  # allow GK/player only with strong evidence later — still soft veto for merge default
        elif "REFEREE" in {fa, fb} or "STAFF" in {fa, fb}:
            return "role_conflict"
    if (
        a.team_id is not None
        and b.team_id is not None
        and a.team_id >= 0
        and b.team_id >= 0
        and a.team_id != b.team_id
        and a.team_confidence >= 0.4
        and b.team_confidence >= 0.4
        and fa == "PLAYER"
        and fb == "PLAYER"
    ):
        return "cross_team"
    if (
        a.jersey_status in {"CONFIRMED", "PROVISIONAL"}
        and b.jersey_status in {"CONFIRMED", "PROVISIONAL"}
        and a.jersey_number
        and b.jersey_number
        and a.jersey_number != b.jersey_number
        and a.jersey_confidence >= 0.55
        and b.jersey_confidence >= 0.55
        and a.jersey_status != "CONFLICTING"
        and b.jersey_status != "CONFLICTING"
    ):
        return "jersey_conflict"
    # temporal gap + motion
    if a.end_frame < b.start_frame:
        gap_s = (b.start_frame - a.end_frame) / config.fps
        if a.end_pitch and b.start_pitch and gap_s > 0.05:
            dist = float(np.hypot(b.start_pitch[0] - a.end_pitch[0], b.start_pitch[1] - a.end_pitch[1]))
            if dist / gap_s > config.max_speed_mps:
                return "impossible_motion"
    elif b.end_frame < a.start_frame:
        gap_s = (a.start_frame - b.end_frame) / config.fps
        if b.end_pitch and a.start_pitch and gap_s > 0.05:
            dist = float(np.hypot(a.start_pitch[0] - b.end_pitch[0], a.start_pitch[1] - b.end_pitch[1]))
            if dist / gap_s > config.max_speed_mps:
                return "impossible_motion"
    return None


def pair_score(a: TrackletIdentity, b: TrackletIdentity, *, config: AssocConfig) -> tuple[float, list[str]] | None:
    veto = pairwise_veto(a, b, config=config)
    if veto:
        return None
    # order temporally: earlier then later
    if a.end_frame <= b.start_frame:
        first, second = a, b
    elif b.end_frame <= a.start_frame:
        first, second = b, a
    else:
        return None

    gap_s = max(0.0, (second.start_frame - first.end_frame) / config.fps)
    sim = cosine(first.embedding, second.embedding)
    max_gap = config.max_gap_strong_s if (sim is not None and sim >= config.reid_strong) else config.max_gap_s
    if gap_s > max_gap:
        return None

    reasons: list[str] = []
    score = 0.0
    has_strong = False

    if sim is not None and first.valid_reid_crop_count >= 1 and second.valid_reid_crop_count >= 1:
        if sim < config.reid_merge:
            return None
        score += sim
        reasons.append(f"reid={sim:.3f}")
        if sim >= config.reid_strong:
            has_strong = True
    else:
        # no ReID: only confirmed jersey match + same team
        if not (
            first.jersey_number
            and second.jersey_number
            and first.jersey_number == second.jersey_number
            and first.jersey_status == "CONFIRMED"
            and second.jersey_status == "CONFIRMED"
            and first.team_id is not None
            and first.team_id == second.team_id
        ):
            return None
        score += 0.45
        reasons.append("jersey_confirmed_match")
        has_strong = True

    if first.jersey_number and second.jersey_number and first.jersey_number == second.jersey_number:
        if first.jersey_confidence >= 0.5 and second.jersey_confidence >= 0.5:
            score += config.jersey_match_bonus
            reasons.append("jersey_match")
            has_strong = True

    # short-gap motion support
    if gap_s <= config.max_gap_short_s and first.end_pitch and second.start_pitch:
        dist = float(np.hypot(second.start_pitch[0] - first.end_pitch[0], second.start_pitch[1] - first.end_pitch[1]))
        if dist <= config.short_gap_pitch_m:
            score += 0.08
            reasons.append(f"motion_gap={gap_s:.2f}")
        elif sim is None or sim < config.reid_strong:
            return None

    # shoe auxiliary only
    if (
        first.shoe_status == "VALID"
        and second.shoe_status == "VALID"
        and _hsv_close(first.shoe_hsv, second.shoe_hsv)
        and first.shoe_confidence >= 0.25
        and second.shoe_confidence >= 0.25
    ):
        score += config.shoe_aux_bonus
        reasons.append("shoe_aux")

    # scene-cut alone + color: already blocked (need reid/jersey)
    if score < config.min_merge_score and not has_strong:
        return None
    if score < config.min_merge_score:
        return None
    return score, reasons


class _DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


def _cluster_safe(tracklets: list[TrackletIdentity], members: list[int], newbie: int, config: AssocConfig) -> bool:
    """Transitive safety: newbie must not veto with any existing cluster member."""
    t_new = tracklets[newbie]
    for mi in members:
        if pairwise_veto(tracklets[mi], t_new, config=config) is not None:
            return False
    return True


def associate_global_constrained(
    tracklets: list[TrackletIdentity],
    *,
    config: AssocConfig | None = None,
    must_links: list[tuple[int, int]] | None = None,
    cannot_links: list[tuple[int, int]] | None = None,
) -> tuple[dict[int, int], list[dict[str, Any]], list[GlobalPlayerState], list[dict[str, Any]]]:
    """Constrained hierarchical linking with pairwise + transitive vetoes.

    must_links / cannot_links are tracklet_id pairs (human review constraints).
    Physical vetoes always win over must-links (caller should record conflicts).
    """
    cfg = config or AssocConfig()
    n = len(tracklets)
    if n == 0:
        return {}, [], [], []

    id_to_idx = {t.tracklet_id: i for i, t in enumerate(tracklets)}
    cannot: set[tuple[int, int]] = set()
    for a, b in cannot_links or []:
        if a in id_to_idx and b in id_to_idx:
            i, j = id_to_idx[a], id_to_idx[b]
            cannot.add((min(i, j), max(i, j)))

    # score all pairs
    edges: list[tuple[float, int, int, list[str]]] = []
    reject_audit: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in cannot:
                reject_audit.append(
                    {
                        "tracklet_a": tracklets[i].tracklet_id,
                        "tracklet_b": tracklets[j].tracklet_id,
                        "decision": "reject",
                        "reason": "human_cannot_link",
                    }
                )
                continue
            veto = pairwise_veto(tracklets[i], tracklets[j], config=cfg)
            if veto:
                reject_audit.append(
                    {
                        "tracklet_a": tracklets[i].tracklet_id,
                        "tracklet_b": tracklets[j].tracklet_id,
                        "decision": "reject",
                        "reason": veto,
                    }
                )
                continue
            scored = pair_score(tracklets[i], tracklets[j], config=cfg)
            if scored is None:
                reject_audit.append(
                    {
                        "tracklet_a": tracklets[i].tracklet_id,
                        "tracklet_b": tracklets[j].tracklet_id,
                        "decision": "reject",
                        "reason": "score_below_threshold",
                    }
                )
                continue
            score, reasons = scored
            edges.append((score, i, j, reasons))

    # Force must-link edges with high priority if physically safe
    must_edges: list[tuple[float, int, int, list[str]]] = []
    for a, b in must_links or []:
        if a not in id_to_idx or b not in id_to_idx:
            continue
        i, j = id_to_idx[a], id_to_idx[b]
        ii, jj = min(i, j), max(i, j)
        if (ii, jj) in cannot:
            reject_audit.append(
                {
                    "tracklet_a": a,
                    "tracklet_b": b,
                    "decision": "reject",
                    "reason": "must_vs_cannot",
                }
            )
            continue
        veto = pairwise_veto(tracklets[i], tracklets[j], config=cfg)
        if veto:
            reject_audit.append(
                {
                    "tracklet_a": a,
                    "tracklet_b": b,
                    "decision": "reject",
                    "reason": f"human_must_link_blocked:{veto}",
                }
            )
            continue
        must_edges.append((9.0, i, j, ["human_must_link"]))

    edges = must_edges + edges
    edges.sort(key=lambda x: -x[0])
    dsu = _DSU(n)
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    merge_audit: list[dict[str, Any]] = []

    def _cannot_between(members_i: list[int], members_j: list[int]) -> bool:
        for a in members_i:
            for b in members_j:
                aa, bb = min(a, b), max(a, b)
                if (aa, bb) in cannot:
                    return True
        return False

    for score, i, j, reasons in edges:
        ri, rj = dsu.find(i), dsu.find(j)
        if ri == rj:
            continue
        members_i = clusters[ri]
        members_j = clusters[rj]
        if _cannot_between(members_i, members_j):
            reject_audit.append(
                {
                    "tracklet_a": tracklets[i].tracklet_id,
                    "tracklet_b": tracklets[j].tracklet_id,
                    "decision": "reject",
                    "reason": "human_cannot_link_transitive",
                    "score": score,
                }
            )
            continue
        # transitive veto check between all pairs across clusters
        safe = True
        for a in members_i:
            for b in members_j:
                if pairwise_veto(tracklets[a], tracklets[b], config=cfg) is not None:
                    safe = False
                    break
            if not safe:
                break
        if not safe:
            reject_audit.append(
                {
                    "tracklet_a": tracklets[i].tracklet_id,
                    "tracklet_b": tracklets[j].tracklet_id,
                    "decision": "reject",
                    "reason": "unsafe_transitive",
                    "score": score,
                }
            )
            continue
        dsu.union(i, j)
        root = dsu.find(i)
        merged = members_i + members_j
        clusters[root] = merged
        if root != ri:
            clusters.pop(ri, None)
        if root != rj and rj in clusters and rj != root:
            clusters.pop(rj, None)
        merge_audit.append(
            {
                "tracklet_a": tracklets[i].tracklet_id,
                "tracklet_b": tracklets[j].tracklet_id,
                "decision": "merge",
                "reason": ",".join(reasons),
                "score": score,
            }
        )

    # allocate monotonic GIDs by earliest start among cluster (stable display)
    roots = {}
    for i in range(n):
        roots.setdefault(dsu.find(i), []).append(i)

    ordered_roots = sorted(
        roots.items(),
        key=lambda kv: (min(tracklets[i].start_frame for i in kv[1]), min(tracklets[i].tracklet_id for i in kv[1])),
    )

    mapping: dict[int, int] = {}
    players: list[GlobalPlayerState] = []
    next_gid = 1
    for _, members in ordered_roots:
        members = sorted(members, key=lambda i: tracklets[i].start_frame)
        # skip pure ball
        if all(tracklets[i].class_name == "ball" for i in members):
            continue
        # quality gate: very short low-quality single tracklets → UNRESOLVED (still get GID for display as ?)
        t0 = tracklets[members[0]]
        status = "PROVISIONAL"
        if len(members) >= 2 or (t0.valid_reid_crop_count >= 2 and t0.duration_s >= 1.0):
            status = "CONFIRMED"
        if t0.quality == "REJECT" or t0.duration_s < 0.2:
            status = "UNRESOLVED"

        emb = None
        for i in members:
            e = tracklets[i].embedding
            if e is None:
                continue
            emb = e.copy() if emb is None else 0.7 * emb + 0.3 * e
        if emb is not None:
            nrm = float(np.linalg.norm(emb))
            if nrm > 1e-9:
                emb = emb / nrm

        jersey_number = None
        jersey_status = "UNREADABLE"
        jersey_conf = 0.0
        for i in members:
            ti = tracklets[i]
            if ti.jersey_status == "CONFIRMED" and ti.jersey_number:
                jersey_number = ti.jersey_number
                jersey_status = "CONFIRMED"
                jersey_conf = max(jersey_conf, ti.jersey_confidence)
                break
            if ti.jersey_status == "PROVISIONAL" and ti.jersey_number and jersey_number is None:
                jersey_number = ti.jersey_number
                jersey_status = "PROVISIONAL"
                jersey_conf = ti.jersey_confidence

        team_id = None
        for i in members:
            if tracklets[i].team_id is not None and tracklets[i].team_id >= 0:
                team_id = tracklets[i].team_id
                break

        role = tracklets[members[0]].role
        class_name = tracklets[members[0]].class_name
        intervals = [(tracklets[i].start_frame, tracklets[i].end_frame) for i in members]
        end_pitch = tracklets[members[-1]].end_pitch

        # referee GIDs use R-prefixed display later; numeric allocator still monotonic
        gid = next_gid
        next_gid += 1
        gp = GlobalPlayerState(
            global_id=gid,
            tracklet_ids=[tracklets[i].tracklet_id for i in members],
            local_track_ids=[tracklets[i].local_track_id for i in members],
            team_id=team_id,
            class_name=class_name,
            role=role,
            intervals=intervals,
            embedding=emb,
            jersey_number=jersey_number,
            jersey_status=jersey_status,
            jersey_confidence=jersey_conf,
            identity_status=status,
            evidence=["cluster_size=%d" % len(members)],
            end_pitch=end_pitch,
        )
        players.append(gp)
        for i in members:
            mapping[tracklets[i].tracklet_id] = gid

    return mapping, merge_audit, players, reject_audit
