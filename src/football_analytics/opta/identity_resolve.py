"""Conservative single-camera global identity for Opta analytics.

Merges local track *fragments* into global players using:
- hard reject on temporal overlap (same-time different tracks)
- hard reject on different teams
- ReID cosine similarity + team colour + pitch continuity for short gaps

Short / low-confidence fragments are discarded from validated counts.
Validated players per team must be ≤ 11 or stats are not publishable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IdentityResolveConfig:
    min_track_frames: int = 20
    min_visible_seconds: float = 1.20
    min_team_confidence: float = 0.45
    max_gap_seconds: float = 8.0
    reid_merge_threshold: float = 0.42
    reid_strong_threshold: float = 0.55
    position_continuity_m: float = 12.0
    max_player_speed_mps: float = 12.0
    max_validated_per_team: int = 11
    camera_id: str = "camera_1"
    # Do NOT hard-cap validated to 11 by demoting surplus.
    # If count > 11 after merge, stats_publishable=false.
    allow_hard_cap_demotion: bool = False


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


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return None
    return float(np.dot(a, b) / (na * nb))


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float], *, tol_ms: float = 40.0) -> bool:
    return not (a[1] + tol_ms < b[0] or b[1] + tol_ms < a[0])


def _any_overlap(intervals: list[tuple[float, float]], other: tuple[float, float]) -> bool:
    return any(_intervals_overlap(iv, other) for iv in intervals)


def _role_excluded(role: str) -> bool:
    text = str(role or "").lower()
    return any(k in text for k in ("referee", "official", "staff", "assistant"))


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
        if team_id is None or team_conf < cfg.min_team_confidence:
            # Keep as non-team fragment only if role is clearly outfield later; skip unknowns for validated pool
            continue
        if str(team_id) in {"", "unknown", "None"}:
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
                team_id=team_id,
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


def resolve_global_identities(
    fragments: list[TrackFragment],
    *,
    config: IdentityResolveConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Return (global_identity_map, identity_report, metrics, decisions)."""
    cfg = config or IdentityResolveConfig()
    # Longest-first stabilises merges
    ordered = sorted(fragments, key=lambda f: (-f.visible_seconds, f.track_id))
    players: list[GlobalPlayer] = []
    next_id = 1
    decisions: list[dict[str, Any]] = []

    for frag in ordered:
        interval = (frag.first_ms, frag.last_ms)
        best_idx: int | None = None
        best_score = -1.0
        best_reason = ""
        best_detail: dict[str, Any] = {}

        for idx, player in enumerate(players):
            # Different team → never merge
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
            # Goalkeeper ↔ outfield hard reject
            roles = {str(player.role).lower(), str(frag.role).lower()}
            if "goalkeeper" in roles and ("outfield" in roles or "unknown" in roles):
                if "goalkeeper" in {str(player.role).lower()} and "goalkeeper" not in {
                    str(frag.role).lower()
                }:
                    decisions.append(
                        {
                            "local_track_id": frag.track_id,
                            "candidate_global_id": player.global_id,
                            "decision": "reject",
                            "reason": "role_mismatch_gk_outfield",
                        }
                    )
                    continue
                if "goalkeeper" in {str(frag.role).lower()} and "goalkeeper" not in {
                    str(player.role).lower()
                }:
                    decisions.append(
                        {
                            "local_track_id": frag.track_id,
                            "candidate_global_id": player.global_id,
                            "decision": "reject",
                            "reason": "role_mismatch_gk_outfield",
                        }
                    )
                    continue
            # Simultaneous visibility → never merge
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

            # Gap between fragment and nearest player interval
            gaps = []
            for a0, a1 in player.intervals:
                if frag.first_ms >= a1:
                    gaps.append(frag.first_ms - a1)
                elif frag.last_ms <= a0:
                    gaps.append(a0 - frag.last_ms)
            gap_ms = min(gaps) if gaps else 0.0
            gap_s = gap_ms / 1000.0
            if gap_s > cfg.max_gap_seconds:
                decisions.append(
                    {
                        "local_track_id": frag.track_id,
                        "candidate_global_id": player.global_id,
                        "decision": "reject",
                        "reason": "gap_too_long",
                        "gap_seconds": gap_s,
                    }
                )
                continue

            reid_sim = _cosine(frag.embedding, player.embedding)
            team_sim = 1.0 if frag.team_id == player.team_id else 0.0
            # Position continuity: compare nearest gap endpoints (exit→entry)
            pos_delta = None
            if frag.start_xy is not None or frag.end_xy is not None:
                # Find closest interval on either side
                for a0, a1 in player.intervals:
                    if frag.first_ms >= a1:
                        # fragment after player interval: player exit → frag entry
                        exit_xy = getattr(player, "_end_xy", None) or getattr(player, "_mean_xy", None)
                        entry = frag.start_xy or frag.mean_xy
                        if exit_xy is not None and entry is not None:
                            d = float(np.hypot(entry[0] - exit_xy[0], entry[1] - exit_xy[1]))
                            pos_delta = d if pos_delta is None else min(pos_delta, d)
                    elif frag.last_ms <= a0:
                        # fragment before player interval: frag exit → player entry
                        entry_xy = getattr(player, "_start_xy", None) or getattr(player, "_mean_xy", None)
                        exit_f = frag.end_xy or frag.mean_xy
                        if entry_xy is not None and exit_f is not None:
                            d = float(np.hypot(exit_f[0] - entry_xy[0], exit_f[1] - entry_xy[1]))
                            pos_delta = d if pos_delta is None else min(pos_delta, d)
            # Physical feasibility: required speed across gap
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
                        }
                    )
                    continue

            score = 0.0
            reasons = []
            if reid_sim is not None:
                score += 0.55 * max(0.0, reid_sim)
                reasons.append(f"reid={reid_sim:.3f}")
            score += 0.20 * team_sim
            reasons.append(f"team={team_sim:.2f}")
            if pos_delta is not None:
                pos_score = max(0.0, 1.0 - pos_delta / max(cfg.position_continuity_m, 1e-6))
                score += 0.25 * pos_score
                reasons.append(f"pos_delta={pos_delta:.2f}")
            score += 0.10 * max(0.0, 1.0 - gap_s / cfg.max_gap_seconds)

            accept = False
            reason = ""
            if reid_sim is not None and reid_sim >= cfg.reid_strong_threshold and team_sim == 1.0:
                accept = True
                reason = "strong_reid_same_team"
            elif (
                reid_sim is not None
                and reid_sim >= cfg.reid_merge_threshold
                and team_sim == 1.0
                and (pos_delta is None or pos_delta <= cfg.position_continuity_m)
            ):
                accept = True
                reason = "reid_team_position"
            elif (
                reid_sim is None
                and team_sim == 1.0
                and pos_delta is not None
                and pos_delta <= cfg.position_continuity_m * 0.6
                and gap_s <= 2.0
            ):
                accept = True
                reason = "short_gap_position_continuity"
            elif (
                reid_sim is not None
                and reid_sim >= cfg.reid_merge_threshold - 0.05
                and team_sim == 1.0
                and pos_delta is not None
                and pos_delta <= cfg.position_continuity_m * 0.5
                and gap_s <= 3.0
            ):
                accept = True
                reason = "reid_strong_position_bridge"

            if accept and score > best_score:
                best_score = score
                best_idx = idx
                best_reason = reason
                best_detail = {
                    "reid_sim": reid_sim,
                    "team_sim": team_sim,
                    "pos_delta": pos_delta,
                    "gap_seconds": gap_s,
                    "score": score,
                    "cues": reasons,
                }

        if best_idx is not None:
            player = players[best_idx]
            player.track_ids.append(frag.track_id)
            player.intervals.append(interval)
            player.visible_seconds += frag.visible_seconds
            player.merge_reasons.append(best_reason)
            if best_detail.get("reid_sim") is not None:
                player.reid_sims.append(float(best_detail["reid_sim"]))
            player.team_sims.append(float(best_detail.get("team_sim") or 0.0))
            if best_detail.get("pos_delta") is not None:
                player.position_deltas.append(float(best_detail["pos_delta"]))
            if frag.embedding is not None:
                if player.embedding is None:
                    player.embedding = frag.embedding.copy()
                    player.embedding_n = 1
                else:
                    player.embedding = (player.embedding * player.embedding_n + frag.embedding) / (
                        player.embedding_n + 1
                    )
                    player.embedding_n += 1
            if frag.end_xy is not None:
                player._end_xy = frag.end_xy  # type: ignore[attr-defined]
            if frag.start_xy is not None:
                if not hasattr(player, "_start_xy") or player._start_xy is None:  # type: ignore[attr-defined]
                    player._start_xy = frag.start_xy  # type: ignore[attr-defined]
            if frag.mean_xy is not None:
                player._mean_xy = frag.mean_xy  # type: ignore[attr-defined]
            decisions.append(
                {
                    "local_track_id": frag.track_id,
                    "global_id": player.global_id,
                    "decision": "merge",
                    "reason": best_reason,
                    **{k: v for k, v in best_detail.items() if k != "cues"},
                    "cues": ";".join(best_detail.get("cues") or []),
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
            )
            player._end_xy = frag.end_xy  # type: ignore[attr-defined]
            player._start_xy = frag.start_xy  # type: ignore[attr-defined]
            player._mean_xy = frag.mean_xy  # type: ignore[attr-defined]
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

    # Quality-based validation — NO hard-cap demotion unless explicitly enabled.
    by_team: dict[str, list[GlobalPlayer]] = {}
    for p in players:
        if p.team_id is None:
            p.validated = False
            p.quality = "low"
            continue
        # Validated = enough visibility + not unresolved leftover
        p.validated = p.visible_seconds >= 1.0 and len(p.track_ids) >= 1
        p.quality = "high" if p.visible_seconds >= 3.0 else ("medium" if p.validated else "low")
        by_team.setdefault(str(p.team_id), []).append(p)

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
        sum(1 for p in group if p.validated) > cfg.max_validated_per_team
        for group in by_team.values()
    )
    stats_publishable = not raw_over and bool(players) and not identity_flags

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
        "suspected_false_merges": 0,  # reserved; no automatic claim without GT
        "hard_cap_demotion_enabled": bool(cfg.allow_hard_cap_demotion),
    }
    decisions_frame = pd.DataFrame(decisions)
    report = pd.DataFrame(report_rows)
    metrics["decisions_count"] = int(len(decisions_frame))
    return pd.DataFrame(map_rows), report, metrics, decisions_frame
