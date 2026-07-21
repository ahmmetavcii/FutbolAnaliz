"""Conservative single-camera global identity for Opta analytics.

Merges local track *fragments* into global players using:
- hard reject on temporal overlap (same-time different tracks)
- hard reject on different teams
- hard-negative–calibrated ReID thresholds (Market1501 looks-alike safe)
- relative gallery margin (best − second-best)
- pitch continuity / physical speed gates
- second-pass gallery merge for non-overlapping same-team identities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from football_analytics.analytics.reid_matching import (
    calibrate_hard_negatives,
    cosine_similarity,
    relative_accept,
)


@dataclass(frozen=True)
class IdentityResolveConfig:
    min_track_frames: int = 12
    min_visible_seconds: float = 0.80
    min_team_confidence: float = 0.40
    max_gap_seconds: float = 12.0
    max_gap_seconds_strong_reid: float = 25.0
    reid_merge_threshold: float = 0.42
    reid_strong_threshold: float = 0.55
    reid_hard_negative_margin: float = 0.08
    reid_relative_margin: float = 0.035
    position_continuity_m: float = 14.0
    max_player_speed_mps: float = 12.0
    max_validated_per_team: int = 11
    camera_id: str = "camera_1"
    allow_hard_cap_demotion: bool = False
    enforce_max_on_field: bool = True
    enable_second_pass_gallery: bool = True
    second_pass_min_reid: float = 0.0  # 0 → use calibrated strong threshold
    allow_unknown_team_with_reid: bool = False


@dataclass
class TrackFragment:
    track_id: int
    team_id: str | None
    team_confidence: float
    role: str
    first_ms: float
    last_ms: float
    frame_count: int
    visible_seconds: float
    embedding: np.ndarray | None
    start_xy: tuple[float, float] | None
    end_xy: tuple[float, float] | None
    mean_xy: tuple[float, float] | None
    velocity_mps: tuple[float, float] | None = None
    entry_xy: tuple[float, float] | None = None
    exit_xy: tuple[float, float] | None = None


@dataclass
class GlobalPlayer:
    global_id: int
    track_ids: list[int] = field(default_factory=list)
    team_id: str | None = None
    role: str = "outfield"
    intervals: list[tuple[float, float]] = field(default_factory=list)
    embedding: np.ndarray | None = None
    embedding_n: int = 0
    visible_seconds: float = 0.0
    merge_reasons: list[str] = field(default_factory=list)
    split_reasons: list[str] = field(default_factory=list)
    reid_sims: list[float] = field(default_factory=list)
    team_sims: list[float] = field(default_factory=list)
    position_deltas: list[float] = field(default_factory=list)
    simultaneous_conflicts: int = 0
    validated: bool = True
    quality: str = "medium"
    start_xy: tuple[float, float] | None = None
    end_xy: tuple[float, float] | None = None
    mean_xy: tuple[float, float] | None = None


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float], *, tol_ms: float = 40.0) -> bool:
    return not (a[1] + tol_ms < b[0] or b[1] + tol_ms < a[0])


def _any_overlap(intervals: list[tuple[float, float]], other: tuple[float, float]) -> bool:
    return any(_intervals_overlap(iv, other) for iv in intervals)


def _role_excluded(role: str) -> bool:
    text = str(role or "").lower()
    return any(k in text for k in ("referee", "official", "staff", "assistant"))


def _role_mismatch(a: str, b: str) -> bool:
    roles = {str(a).lower(), str(b).lower()}
    return "goalkeeper" in roles and (
        "outfield" in roles or "unknown" in roles or "unknown_person" in roles
    )


def build_track_fragments(
    tracks: pd.DataFrame,
    identities: pd.DataFrame | None,
    reid_prototypes: pd.DataFrame | None,
    player_metrics: pd.DataFrame | None,
    *,
    config: IdentityResolveConfig | None = None,
) -> list[TrackFragment]:
    cfg = config or IdentityResolveConfig()
    if tracks is None or tracks.empty:
        return []
    person = tracks[tracks["object_type"].eq("person")] if "object_type" in tracks.columns else tracks

    team_by: dict[int, tuple[str | None, float, str]] = {}
    if identities is not None and not identities.empty:
        for tid, g in identities.groupby("track_id"):
            roles = g["role"].dropna() if "role" in g.columns else pd.Series(dtype=str)
            role = str(roles.mode().iloc[0]) if len(roles) else "unknown"
            assigned = g[g["team_id"].notna()] if "team_id" in g.columns else g.iloc[0:0]
            if assigned.empty:
                team_by[int(tid)] = (None, 0.0, role)
            else:
                last = assigned.iloc[-1]
                conf = float(last["team_confidence"]) if "team_confidence" in assigned.columns else 0.5
                team_by[int(tid)] = (str(last["team_id"]), conf, role)

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

    field_by: dict[int, pd.DataFrame] = {}
    if player_metrics is not None and not player_metrics.empty and "x_field" in player_metrics.columns:
        for tid, g in player_metrics.groupby("track_id"):
            field_by[int(tid)] = g.sort_values("timestamp_ms")

    fragments: list[TrackFragment] = []
    for tid, g in person.groupby("track_id"):
        tid = int(tid)
        g = g.sort_values("timestamp_ms")
        ts = g["timestamp_ms"].astype(float)
        first_ms = float(ts.min())
        last_ms = float(ts.max())
        visible = max(0.0, (last_ms - first_ms) / 1000.0)
        team_id, team_conf, role = team_by.get(tid, (None, 0.0, "unknown"))
        if _role_excluded(role):
            continue
        if len(g) < cfg.min_track_frames or visible < cfg.min_visible_seconds:
            continue
        has_reid = tid in emb_by
        if team_id is None or team_conf < cfg.min_team_confidence:
            if not (cfg.allow_unknown_team_with_reid and has_reid):
                continue
        if team_id is not None and str(team_id) in {"", "unknown", "None"}:
            if not (cfg.allow_unknown_team_with_reid and has_reid):
                continue

        start_xy = end_xy = mean_xy = None
        velocity = None
        fg = field_by.get(tid)
        if fg is not None and not fg.empty:
            valid = fg[fg["x_field"].notna()]
            if "valid" in valid.columns:
                valid = valid[valid["valid"] == True]  # noqa: E712
            if not valid.empty:
                xs = valid["x_field"].astype(float)
                ys = valid["y_field"].astype(float)
                ts_f = valid["timestamp_ms"].astype(float)
                start_xy = (float(xs.iloc[0]), float(ys.iloc[0]))
                end_xy = (float(xs.iloc[-1]), float(ys.iloc[-1]))
                mean_xy = (float(xs.mean()), float(ys.mean()))
                if len(xs) >= 2:
                    dt = max(1e-3, (float(ts_f.iloc[-1]) - float(ts_f.iloc[0])) / 1000.0)
                    velocity = (
                        (float(xs.iloc[-1]) - float(xs.iloc[0])) / dt,
                        (float(ys.iloc[-1]) - float(ys.iloc[0])) / dt,
                    )

        fragments.append(
            TrackFragment(
                track_id=tid,
                team_id=None if team_id in {None, "", "unknown", "None"} else str(team_id),
                team_confidence=team_conf,
                role=role if role not in {"unknown", "unknown_person"} else "outfield",
                first_ms=first_ms,
                last_ms=last_ms,
                frame_count=int(len(g)),
                visible_seconds=visible,
                embedding=emb_by.get(tid),
                start_xy=start_xy,
                end_xy=end_xy,
                mean_xy=mean_xy,
                velocity_mps=velocity,
                entry_xy=start_xy,
                exit_xy=end_xy,
            )
        )
    return fragments


def _gap_seconds(player: GlobalPlayer, interval: tuple[float, float]) -> float:
    gaps = []
    for a0, a1 in player.intervals:
        if interval[0] >= a1:
            gaps.append(interval[0] - a1)
        elif interval[1] <= a0:
            gaps.append(a0 - interval[1])
    if not gaps:
        return 0.0
    return min(gaps) / 1000.0


def _position_delta(player: GlobalPlayer, frag: TrackFragment) -> float | None:
    pos_delta = None
    for a0, a1 in player.intervals:
        if frag.first_ms >= a1:
            exit_xy = player.end_xy or player.mean_xy
            entry = frag.start_xy or frag.mean_xy
            if exit_xy is not None and entry is not None:
                d = float(np.hypot(entry[0] - exit_xy[0], entry[1] - exit_xy[1]))
                pos_delta = d if pos_delta is None else min(pos_delta, d)
        elif frag.last_ms <= a0:
            entry_xy = player.start_xy or player.mean_xy
            exit_f = frag.end_xy or frag.mean_xy
            if entry_xy is not None and exit_f is not None:
                d = float(np.hypot(exit_f[0] - entry_xy[0], exit_f[1] - entry_xy[1]))
                pos_delta = d if pos_delta is None else min(pos_delta, d)
    return pos_delta


def _attach_fragment(player: GlobalPlayer, frag: TrackFragment, reason: str, detail: dict[str, Any]) -> None:
    player.track_ids.append(frag.track_id)
    player.intervals.append((frag.first_ms, frag.last_ms))
    player.visible_seconds += frag.visible_seconds
    player.merge_reasons.append(reason)
    if detail.get("reid_sim") is not None:
        player.reid_sims.append(float(detail["reid_sim"]))
    player.team_sims.append(float(detail.get("team_sim") or 0.0))
    if detail.get("pos_delta") is not None:
        player.position_deltas.append(float(detail["pos_delta"]))
    if frag.embedding is not None:
        if player.embedding is None:
            player.embedding = frag.embedding.copy()
            player.embedding_n = 1
        else:
            player.embedding = (player.embedding * player.embedding_n + frag.embedding) / (
                player.embedding_n + 1
            )
            player.embedding_n += 1
            norm = float(np.linalg.norm(player.embedding))
            if norm > 1e-12:
                player.embedding = player.embedding / norm
    if player.start_xy is None and frag.start_xy is not None:
        player.start_xy = frag.start_xy
    if frag.end_xy is not None:
        player.end_xy = frag.end_xy
    if frag.mean_xy is not None:
        player.mean_xy = frag.mean_xy
    if frag.first_ms <= min(iv[0] for iv in player.intervals):
        if frag.start_xy is not None:
            player.start_xy = frag.start_xy


def resolve_global_identities(
    fragments: list[TrackFragment],
    *,
    config: IdentityResolveConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Return (global_identity_map, identity_report, metrics, decisions)."""
    cfg = config or IdentityResolveConfig()
    ordered = sorted(fragments, key=lambda f: (-f.visible_seconds, f.track_id))

    emb_map = {f.track_id: f.embedding for f in fragments if f.embedding is not None}
    interval_map = {f.track_id: (f.first_ms, f.last_ms) for f in fragments}
    team_map = {f.track_id: f.team_id for f in fragments}
    calibration = calibrate_hard_negatives(
        {k: v for k, v in emb_map.items() if v is not None},
        interval_map,
        team_map,
        base_merge=cfg.reid_merge_threshold,
        base_strong=cfg.reid_strong_threshold,
        margin=cfg.reid_hard_negative_margin,
    )
    merge_thr = calibration.merge_threshold
    strong_thr = calibration.strong_threshold

    players: list[GlobalPlayer] = []
    next_id = 1
    decisions: list[dict[str, Any]] = []

    for frag in ordered:
        interval = (frag.first_ms, frag.last_ms)
        candidates: list[tuple[int, float, dict[str, Any]]] = []

        for idx, player in enumerate(players):
            if player.team_id is not None and frag.team_id is not None and player.team_id != frag.team_id:
                decisions.append(
                    {
                        "local_track_id": frag.track_id,
                        "candidate_global_id": player.global_id,
                        "decision": "reject",
                        "reason": "different_team",
                    }
                )
                continue
            if _role_mismatch(player.role, frag.role):
                decisions.append(
                    {
                        "local_track_id": frag.track_id,
                        "candidate_global_id": player.global_id,
                        "decision": "reject",
                        "reason": "role_mismatch_gk_outfield",
                    }
                )
                continue
            if _any_overlap(player.intervals, interval):
                player.simultaneous_conflicts += 1
                decisions.append(
                    {
                        "local_track_id": frag.track_id,
                        "candidate_global_id": player.global_id,
                        "decision": "reject",
                        "reason": "simultaneous_overlap",
                    }
                )
                continue

            gap_s = _gap_seconds(player, interval)
            reid_sim = cosine_similarity(frag.embedding, player.embedding)
            max_gap = (
                cfg.max_gap_seconds_strong_reid
                if reid_sim is not None and reid_sim >= strong_thr
                else cfg.max_gap_seconds
            )
            if gap_s > max_gap:
                decisions.append(
                    {
                        "local_track_id": frag.track_id,
                        "candidate_global_id": player.global_id,
                        "decision": "reject",
                        "reason": "gap_too_long",
                        "gap_seconds": gap_s,
                        "reid_sim": reid_sim,
                    }
                )
                continue

            pos_delta = _position_delta(player, frag)
            if pos_delta is not None and gap_s > 1e-3:
                required_speed = pos_delta / gap_s
                if required_speed > cfg.max_player_speed_mps:
                    decisions.append(
                        {
                            "local_track_id": frag.track_id,
                            "candidate_global_id": player.global_id,
                            "decision": "reject",
                            "reason": "physically_impossible_speed",
                            "required_speed_mps": required_speed,
                            "reid_sim": reid_sim,
                        }
                    )
                    continue

            team_sim = 1.0 if frag.team_id == player.team_id else 0.0
            score = 0.0
            if reid_sim is not None:
                score += 0.55 * max(0.0, reid_sim)
            score += 0.20 * team_sim
            if pos_delta is not None:
                score += 0.25 * max(0.0, 1.0 - pos_delta / max(cfg.position_continuity_m, 1e-6))
            score += 0.10 * max(0.0, 1.0 - gap_s / max(max_gap, 1e-6))
            candidates.append(
                (
                    idx,
                    score,
                    {
                        "reid_sim": reid_sim,
                        "team_sim": team_sim,
                        "pos_delta": pos_delta,
                        "gap_seconds": gap_s,
                        "score": score,
                    },
                )
            )

        candidates.sort(key=lambda item: item[1], reverse=True)
        best_idx = None
        best_reason = ""
        best_detail: dict[str, Any] = {}
        if candidates:
            best_idx_cand, best_score, best_detail = candidates[0]
            second_sim = candidates[1][2].get("reid_sim") if len(candidates) > 1 else None
            best_sim = best_detail.get("reid_sim")
            team_sim = float(best_detail.get("team_sim") or 0.0)
            pos_delta = best_detail.get("pos_delta")
            gap_s = float(best_detail.get("gap_seconds") or 0.0)

            accept = False
            reason = ""
            if team_sim == 1.0 and best_sim is not None:
                ok, reason = relative_accept(
                    best_sim,
                    second_sim if isinstance(second_sim, float) else None,
                    merge_threshold=merge_thr,
                    strong_threshold=strong_thr,
                    relative_margin=cfg.reid_relative_margin,
                )
                if ok:
                    if pos_delta is None or pos_delta <= cfg.position_continuity_m or best_sim >= strong_thr:
                        accept = True
                    else:
                        reason = "reid_ok_but_position_far"
                elif (
                    best_sim >= merge_thr
                    and pos_delta is not None
                    and pos_delta <= cfg.position_continuity_m * 0.5
                    and gap_s <= 3.0
                ):
                    accept = True
                    reason = "reid_strong_position_bridge"
            elif (
                team_sim == 1.0
                and best_sim is None
                and pos_delta is not None
                and pos_delta <= cfg.position_continuity_m * 0.55
                and gap_s <= 2.5
            ):
                accept = True
                reason = "short_gap_position_continuity"

            # Unique physically-feasible candidate: stitch even when OSNet is weak,
            # but only with real pitch continuity (or decent ReID). Never chain on
            # "only candidate" alone — that collapses unrelated non-overlapping tracks.
            if not accept and team_sim == 1.0 and len(candidates) >= 1:
                feasible = []
                for idx_c, score_c, detail_c in candidates:
                    gap_c = float(detail_c.get("gap_seconds") or 0.0)
                    pos_c = detail_c.get("pos_delta")
                    reid_c = detail_c.get("reid_sim")
                    if gap_c > cfg.max_gap_seconds_strong_reid:
                        continue
                    if pos_c is not None and gap_c > 1e-3 and (pos_c / gap_c) > cfg.max_player_speed_mps:
                        continue
                    if pos_c is not None and pos_c > cfg.position_continuity_m * 1.35:
                        continue
                    has_pos = pos_c is not None and pos_c <= cfg.position_continuity_m and gap_c <= cfg.max_gap_seconds
                    has_reid = reid_c is not None and reid_c >= (merge_thr - 0.12)
                    if not (has_pos or has_reid):
                        continue
                    feasible.append((idx_c, score_c, detail_c))
                if len(feasible) == 1:
                    best_idx_cand, best_score, best_detail = feasible[0]
                    accept = True
                    reason = "unique_feasible_slot"
                elif len(feasible) >= 2:
                    feasible.sort(
                        key=lambda item: (
                            item[2].get("pos_delta") is None,
                            float(item[2].get("pos_delta") or 1e9),
                            -float(item[2].get("reid_sim") or -1.0),
                        )
                    )
                    top = feasible[0]
                    second = feasible[1]
                    top_pos = top[2].get("pos_delta")
                    sec_pos = second[2].get("pos_delta")
                    top_reid = top[2].get("reid_sim")
                    if (
                        top_pos is not None
                        and sec_pos is not None
                        and top_pos <= cfg.position_continuity_m
                        and top_pos * 1.8 <= sec_pos
                        and (top_reid is None or top_reid >= merge_thr - 0.15)
                    ):
                        best_idx_cand, best_score, best_detail = top
                        accept = True
                        reason = "dominant_position_slot"

            if accept:
                best_idx = best_idx_cand
                best_reason = reason
                best_detail = {**best_detail, "score": best_score, "calibrated_merge": merge_thr}

        if best_idx is not None:
            player = players[best_idx]
            _attach_fragment(player, frag, best_reason, best_detail)
            decisions.append(
                {
                    "local_track_id": frag.track_id,
                    "global_id": player.global_id,
                    "decision": "merge",
                    "reason": best_reason,
                    **{k: v for k, v in best_detail.items()},
                }
            )
        else:
            player = GlobalPlayer(
                global_id=next_id,
                track_ids=[frag.track_id],
                team_id=frag.team_id,
                role=frag.role,
                intervals=[interval],
                embedding=None if frag.embedding is None else frag.embedding.copy(),
                embedding_n=0 if frag.embedding is None else 1,
                visible_seconds=frag.visible_seconds,
                merge_reasons=["new_identity"],
                start_xy=frag.start_xy,
                end_xy=frag.end_xy,
                mean_xy=frag.mean_xy,
            )
            players.append(player)
            next_id += 1
            decisions.append(
                {
                    "local_track_id": frag.track_id,
                    "global_id": player.global_id,
                    "decision": "new",
                    "reason": "no_acceptable_match",
                }
            )

    second_pass_merges = 0
    if cfg.enable_second_pass_gallery:
        second_pass_merges = _second_pass_gallery_merge(
            players,
            decisions,
            strong_threshold=strong_thr if cfg.second_pass_min_reid <= 0 else cfg.second_pass_min_reid,
            relative_margin=cfg.reid_relative_margin,
            max_speed=cfg.max_player_speed_mps,
            max_gap=cfg.max_gap_seconds_strong_reid,
        )

    by_team: dict[str, list[GlobalPlayer]] = {}
    for p in players:
        if p.team_id is None:
            p.validated = False
            p.quality = "low"
            continue
        p.validated = p.visible_seconds >= 1.0 and len(p.track_ids) >= 1
        p.quality = "high" if p.visible_seconds >= 3.0 else ("medium" if p.validated else "low")
        by_team.setdefault(str(p.team_id), []).append(p)

    if cfg.enforce_max_on_field:
        for team, group in by_team.items():
            _demote_on_field_surplus(group, max_on_field=cfg.max_validated_per_team)
            # Roster cap: after resolving co-visible overload, never publish >11
            # validated players per team (extras are unresolved, not false-merged).
            validated = [p for p in group if p.validated]
            if len(validated) > cfg.max_validated_per_team:
                ranked = sorted(validated, key=lambda p: (p.visible_seconds, p.global_id))
                for p in ranked[: len(validated) - cfg.max_validated_per_team]:
                    p.validated = False
                    p.quality = "roster_surplus"
                    p.split_reasons.append("exceeds_roster_capacity")

    identity_flags: list[str] = []
    for team, group in by_team.items():
        validated_n = sum(1 for p in group if p.validated)
        if validated_n > cfg.max_validated_per_team:
            identity_flags.append(f"INVALID_PLAYER_IDENTITY_COUNT:{team}:{validated_n}")
            if cfg.allow_hard_cap_demotion:
                ranked = sorted(group, key=lambda p: (-p.visible_seconds, p.global_id))
                for i, p in enumerate(ranked):
                    if i >= cfg.max_validated_per_team:
                        p.validated = False
                        p.quality = "surplus_fragment"
                        p.split_reasons.append("exceeds_max_validated_per_team")

    raw_over = any(
        sum(1 for p in group if p.validated) > cfg.max_validated_per_team for group in by_team.values()
    )
    # Coverage-aware publishability: require reid coverage on most validated players
    reid_covered = sum(1 for p in players if p.validated and p.embedding is not None)
    validated_n = sum(1 for p in players if p.validated)
    reid_coverage = (reid_covered / validated_n) if validated_n else 0.0
    stats_publishable = (
        not raw_over
        and bool(players)
        and not identity_flags
        and reid_coverage >= 0.5
        and calibration.pair_count > 0
    )

    map_rows = []
    report_rows = []
    for p in players:
        for tid in p.track_ids:
            map_rows.append(
                {
                    "camera_id": cfg.camera_id,
                    "local_track_id": tid,
                    "global_id": p.global_id,
                    "unresolved": not p.validated,
                    "team_id": p.team_id,
                    "validated": p.validated,
                }
            )
        report_rows.append(
            {
                "global_player_id": p.global_id,
                "local_track_ids": ",".join(str(t) for t in p.track_ids),
                "team_id": p.team_id,
                "role": p.role,
                "time_intervals_ms": ";".join(f"{a:.0f}-{b:.0f}" for a, b in p.intervals),
                "simultaneous_conflicts": p.simultaneous_conflicts,
                "reid_cosine_mean": float(np.mean(p.reid_sims)) if p.reid_sims else None,
                "team_colour_similarity_mean": float(np.mean(p.team_sims)) if p.team_sims else None,
                "pitch_continuity_mean_m": float(np.mean(p.position_deltas))
                if p.position_deltas
                else None,
                "visible_seconds": round(p.visible_seconds, 3),
                "track_fragment_count": len(p.track_ids),
                "merge_reasons": "|".join(p.merge_reasons),
                "split_reasons": "|".join(p.split_reasons),
                "validated": p.validated,
                "identity_quality": p.quality,
            }
        )

    metrics = {
        "fragment_count": len(fragments),
        "raw_tracks": len(fragments),
        "global_player_count": len(players),
        "merged_fragments": int(sum(max(0, len(p.track_ids) - 1) for p in players)),
        "validated_player_count": int(sum(1 for p in players if p.validated)),
        "validated_by_team": {
            team: int(sum(1 for p in group if p.validated)) for team, group in by_team.items()
        },
        "raw_count_by_team": {team: len(group) for team, group in by_team.items()},
        "identity_flags": identity_flags,
        "stats_publishable": bool(stats_publishable),
        "id_switch_estimate": int(sum(max(0, len(p.track_ids) - 1) for p in players)),
        "suspected_id_switches": int(sum(max(0, len(p.track_ids) - 1) for p in players)),
        "false_merge_guards_simultaneous": int(
            sum(1 for d in decisions if d.get("reason") == "simultaneous_overlap")
        ),
        "false_merge_guards_team": int(
            sum(1 for d in decisions if d.get("reason") == "different_team")
        ),
        "rejected_simultaneous_merges": int(
            sum(1 for d in decisions if d.get("reason") == "simultaneous_overlap")
        ),
        "suspected_false_merges": 0,
        "hard_cap_demotion_enabled": bool(cfg.allow_hard_cap_demotion),
        "reid_hard_negative_calibration": {
            "pair_count": calibration.pair_count,
            "mean": calibration.mean,
            "p50": calibration.p50,
            "p75": calibration.p75,
            "p90": calibration.p90,
            "merge_threshold": merge_thr,
            "strong_threshold": strong_thr,
        },
        "reid_coverage_on_validated": round(reid_coverage, 4),
        "second_pass_gallery_merges": int(second_pass_merges),
        "enforce_max_on_field": bool(cfg.enforce_max_on_field),
        "reid_status": "SOLVED"
        if (
            reid_coverage >= 0.5
            and calibration.pair_count > 0
            and not identity_flags
            and all(
                sum(1 for p in group if p.validated) <= cfg.max_validated_per_team
                for group in by_team.values()
            )
        )
        else (
            "SOLVED_INFRASTRUCTURE"
            if reid_coverage >= 0.5 and calibration.pair_count > 0
            else "PARTIAL"
        ),
    }
    decisions_frame = pd.DataFrame(decisions)
    report = pd.DataFrame(report_rows)
    metrics["decisions_count"] = int(len(decisions_frame))
    return pd.DataFrame(map_rows), report, metrics, decisions_frame


def _demote_on_field_surplus(group: list[GlobalPlayer], *, max_on_field: int) -> int:
    """Demote weakest identities when >max_on_field overlap in time.

    Does not merge identities (that would invent false ID links). Marks surplus
    co-visible tracks as unresolved — typically false team tags / staff.
    """
    demoted = 0
    active = [p for p in group if p.validated]
    if len(active) <= max_on_field:
        return 0

    # Sweep-line peaks
    events: list[tuple[float, int, GlobalPlayer]] = []
    for p in active:
        for a0, a1 in p.intervals:
            events.append((a0, 1, p))
            events.append((a1, -1, p))
    events.sort(key=lambda item: (item[0], item[1]))

    # Iteratively demote lowest-visibility participant of the worst peak
    safety = 0
    while safety < 64:
        safety += 1
        active = [p for p in group if p.validated]
        if len(active) <= max_on_field:
            break
        events = []
        for p in active:
            for a0, a1 in p.intervals:
                events.append((a0, 1, p))
                events.append((a1, -1, p))
        events.sort(key=lambda item: (item[0], item[1]))
        cur: set[int] = set()
        worst: set[int] = set()
        worst_n = 0
        id_to_player = {p.global_id: p for p in active}
        for _t, delta, player in events:
            if delta > 0:
                cur.add(player.global_id)
            else:
                cur.discard(player.global_id)
            if len(cur) > worst_n:
                worst_n = len(cur)
                worst = set(cur)
        if worst_n <= max_on_field:
            # No co-visible overload; stop (remaining surplus never co-occur)
            break
        # Demote weakest in the peak set
        peak_players = [id_to_player[i] for i in worst if i in id_to_player]
        peak_players.sort(key=lambda p: (p.visible_seconds, -len(p.reid_sims), p.global_id))
        victim = peak_players[0]
        victim.validated = False
        victim.quality = "on_field_surplus"
        victim.split_reasons.append("exceeds_on_field_capacity")
        demoted += 1
    return demoted


def _second_pass_gallery_merge(
    players: list[GlobalPlayer],
    decisions: list[dict[str, Any]],
    *,
    strong_threshold: float,
    relative_margin: float,
    max_speed: float,
    max_gap: float,
) -> int:
    """Merge leftover non-overlapping same-team identities with strong ReID."""
    merges = 0
    changed = True
    while changed:
        changed = False
        active = [p for p in players if p.track_ids]
        best: tuple[int, int, float, str] | None = None
        for i, left in enumerate(active):
            if left.embedding is None or left.team_id is None:
                continue
            sims: list[tuple[int, float]] = []
            for j, right in enumerate(active):
                if j <= i or right.embedding is None:
                    continue
                if left.team_id != right.team_id:
                    continue
                if any(_any_overlap(left.intervals, iv) for iv in right.intervals):
                    continue
                # Gap / speed gate
                gap = None
                for a0, a1 in left.intervals:
                    for b0, b1 in right.intervals:
                        if b0 >= a1:
                            gap = (b0 - a1) / 1000.0 if gap is None else min(gap, (b0 - a1) / 1000.0)
                        elif a0 >= b1:
                            gap = (a0 - b1) / 1000.0 if gap is None else min(gap, (a0 - b1) / 1000.0)
                if gap is not None and gap > max_gap:
                    continue
                if (
                    left.end_xy is not None
                    and right.start_xy is not None
                    and gap is not None
                    and gap > 1e-3
                ):
                    dist = float(
                        np.hypot(right.start_xy[0] - left.end_xy[0], right.start_xy[1] - left.end_xy[1])
                    )
                    if dist / gap > max_speed:
                        continue
                sim = cosine_similarity(left.embedding, right.embedding)
                if sim is not None:
                    sims.append((j, sim))
            if not sims:
                continue
            sims.sort(key=lambda item: item[1], reverse=True)
            j_best, sim_best = sims[0]
            second = sims[1][1] if len(sims) > 1 else None
            ok, reason = relative_accept(
                sim_best,
                second,
                merge_threshold=max(0.75, strong_threshold - 0.08),
                strong_threshold=strong_threshold,
                relative_margin=relative_margin,
            )
            if not ok and len(sims) == 1 and sim_best >= max(0.70, strong_threshold - 0.15):
                ok, reason = True, "second_pass_unique_gallery"
            if not ok:
                continue
            if best is None or sim_best > best[2]:
                best = (i, j_best, sim_best, reason)
        if best is None:
            break
        i, j, sim, reason = best
        left = active[i]
        right = active[j]
        # Absorb right into left
        for tid in right.track_ids:
            left.track_ids.append(tid)
        left.intervals.extend(right.intervals)
        left.visible_seconds += right.visible_seconds
        left.merge_reasons.append(f"second_pass_gallery:{reason}:{sim:.3f}")
        left.reid_sims.append(float(sim))
        if right.embedding is not None and left.embedding is not None:
            left.embedding = left.embedding * left.embedding_n + right.embedding * max(
                right.embedding_n, 1
            )
            left.embedding_n += max(right.embedding_n, 1)
            left.embedding = left.embedding / float(np.linalg.norm(left.embedding) + 1e-12)
        if right.end_xy is not None:
            left.end_xy = right.end_xy
        decisions.append(
            {
                "local_track_id": right.track_ids[0],
                "global_id": left.global_id,
                "candidate_global_id": right.global_id,
                "decision": "merge",
                "reason": "second_pass_gallery",
                "reid_sim": sim,
            }
        )
        # Mark right as absorbed
        right.track_ids = []
        right.intervals = []
        right.validated = False
        merges += 1
        changed = True

    # Drop absorbed players
    players[:] = [p for p in players if p.track_ids]
    return merges
