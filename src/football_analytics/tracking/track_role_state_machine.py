"""Track-level role state machine with hysteresis and zone vetoes."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class RoleLabel(str, Enum):
    PLAYER = "PLAYER"
    GOALKEEPER = "GOALKEEPER"
    REFEREE_CENTER = "REFEREE_CENTER"
    REFEREE_ASSISTANT = "REFEREE_ASSISTANT"
    STAFF = "STAFF"
    BENCH_PLAYER = "BENCH_PLAYER"
    SPECTATOR = "SPECTATOR"
    UNRESOLVED = "UNRESOLVED"


class RolePhase(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOCKED = "LOCKED"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class RoleSMConfig:
    min_frames_confirm: int = 12
    min_frames_lock: int = 25
    referee_enter: float = 0.62
    referee_stay: float = 0.48
    player_enter: float = 0.55
    bench_veto: float = 0.40
    stands_veto: float = 0.50
    on_pitch_ref_min: float = 0.20
    assistant_touchline_min: float = 0.35
    team_color_ref_veto: float = 0.18  # distance threshold in BGR/255 space (lower=closer)
    history: int = 120


@dataclass
class _Acc:
    det: deque = field(default_factory=lambda: deque(maxlen=120))
    zone: deque = field(default_factory=lambda: deque(maxlen=120))
    shot: deque = field(default_factory=lambda: deque(maxlen=120))
    conf: deque = field(default_factory=lambda: deque(maxlen=120))
    feet: deque = field(default_factory=lambda: deque(maxlen=120))
    colors: deque = field(default_factory=lambda: deque(maxlen=60))
    role: RoleLabel = RoleLabel.UNRESOLVED
    phase: RolePhase = RolePhase.TENTATIVE
    flips: int = 0
    stable_frames: int = 0


class TrackRoleStateMachine:
    def __init__(self, config: RoleSMConfig | None = None) -> None:
        self.config = config or RoleSMConfig()
        self._t: dict[int, _Acc] = defaultdict(lambda: _Acc())
        self.team_centers: list[np.ndarray] | None = None
        self.ref_color: np.ndarray | None = None

    def set_team_centers(self, centers: list[tuple[float, float, float]] | None) -> None:
        if centers and len(centers) >= 2:
            self.team_centers = [np.asarray(c, dtype=np.float64) for c in centers[:2]]

    def update(
        self,
        tid: int,
        *,
        det_class: str,
        zone: str,
        shot_type: str,
        conf: float,
        foot_xy: tuple[float, float],
        jersey_bgr: tuple[float, float, float] | None = None,
    ) -> None:
        if tid < 0:
            return
        a = self._t[tid]
        a.det.append(det_class.lower())
        a.zone.append(zone)
        a.shot.append(shot_type.lower())
        a.conf.append(float(conf))
        a.feet.append(foot_xy)
        if jersey_bgr is not None:
            a.colors.append(np.asarray(jersey_bgr, dtype=np.float64))

        n = len(a.det)
        if n < 3:
            return

        votes = Counter(a.det)
        zones = Counter(a.zone)
        ref_r = (votes.get("referee", 0)) / n
        player_r = (votes.get("player", 0)) / n
        on = sum(1 for z in a.zone if z in {"ON_PITCH", "PITCH_MARGIN", "TOUCHLINE_MARGIN"}) / n
        on_interior = sum(1 for z in a.zone if z == "ON_PITCH") / n
        touch = sum(1 for z in a.zone if z in {"PITCH_MARGIN", "TOUCHLINE_MARGIN"}) / n
        bench = sum(1 for z in a.zone if z in {"BENCH_TECHNICAL_AREA", "BENCH", "TECHNICAL_AREA"}) / n
        stands = sum(1 for z in a.zone if z in {"STANDS", "OFF_FIELD"}) / n
        motion = 0.0
        if len(a.feet) >= 2:
            xs = [p[0] for p in a.feet]
            ys = [p[1] for p in a.feet]
            motion = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))

        # team color closeness
        close_team = False
        mean_col = None
        if a.colors:
            mean_col = np.mean(np.stack(list(a.colors)), axis=0)
            if self.team_centers is not None:
                d0 = float(np.linalg.norm(mean_col - self.team_centers[0]) / 255.0)
                d1 = float(np.linalg.norm(mean_col - self.team_centers[1]) / 255.0)
                close_team = min(d0, d1) < self.config.team_color_ref_veto

        proposed = RoleLabel.UNRESOLVED
        reason = "default"

        # hard vetoes
        if stands >= self.config.stands_veto:
            proposed = RoleLabel.SPECTATOR
            reason = "stands_veto"
        elif bench >= self.config.bench_veto and on < 0.25:
            proposed = RoleLabel.STAFF if ref_r >= 0.25 or player_r < 0.5 else RoleLabel.BENCH_PLAYER
            reason = "bench_veto"
        elif (
            ref_r >= (self.config.referee_stay if a.role in {RoleLabel.REFEREE_CENTER, RoleLabel.REFEREE_ASSISTANT} else self.config.referee_enter)
            and on >= self.config.on_pitch_ref_min
            and bench < self.config.bench_veto
            and not close_team
        ):
            if touch >= self.config.assistant_touchline_min and on_interior < 0.55:
                proposed = RoleLabel.REFEREE_ASSISTANT
                reason = "touchline_assistant"
            else:
                proposed = RoleLabel.REFEREE_CENTER
                reason = "ref_votes+on_pitch"
            # update ref color prototype slowly
            if mean_col is not None and n >= self.config.min_frames_lock:
                if self.ref_color is None:
                    self.ref_color = mean_col
                else:
                    self.ref_color = 0.9 * self.ref_color + 0.1 * mean_col
        elif player_r >= self.config.player_enter and on >= 0.3 and stands < 0.35:
            proposed = RoleLabel.PLAYER
            reason = "player_votes+on_pitch"
        elif close_team and on >= 0.3 and ref_r < 0.55:
            proposed = RoleLabel.PLAYER
            reason = "team_color_over_weak_ref"
        else:
            proposed = RoleLabel.UNRESOLVED
            reason = "insufficient"

        # state machine / hysteresis
        if a.phase == RolePhase.LOCKED:
            # only change on strong contradictory evidence
            strong_contra = False
            if a.role in {RoleLabel.REFEREE_CENTER, RoleLabel.REFEREE_ASSISTANT}:
                if proposed in {RoleLabel.STAFF, RoleLabel.BENCH_PLAYER, RoleLabel.SPECTATOR}:
                    strong_contra = True
                if proposed == RoleLabel.PLAYER and player_r > 0.75 and ref_r < 0.25 and n >= 40:
                    strong_contra = True
            elif a.role == RoleLabel.PLAYER:
                if proposed in {RoleLabel.SPECTATOR, RoleLabel.STAFF} and (stands > 0.6 or bench > 0.6):
                    strong_contra = True
                # do NOT flip to referee on short bursts
                if proposed in {RoleLabel.REFEREE_CENTER, RoleLabel.REFEREE_ASSISTANT}:
                    if ref_r < 0.75 or on < 0.35 or close_team or n < 40:
                        strong_contra = False
                        proposed = a.role
                        reason = "locked_block_ref_flip"
                    else:
                        strong_contra = True
            if not strong_contra and proposed != a.role:
                proposed = a.role
                reason = "locked_hold"
            elif strong_contra and proposed != a.role:
                a.flips += 1
                a.phase = RolePhase.CONFIRMED
                a.stable_frames = 0
        else:
            if proposed != a.role and a.role != RoleLabel.UNRESOLVED:
                # require sustained evidence
                a.stable_frames = 0
                if n < self.config.min_frames_confirm:
                    proposed = a.role
                    reason = "tentative_hold"
                else:
                    a.flips += 1
            else:
                a.stable_frames += 1

        a.role = proposed
        if n >= self.config.min_frames_lock and a.stable_frames >= 10 and proposed != RoleLabel.UNRESOLVED:
            a.phase = RolePhase.LOCKED
        elif n >= self.config.min_frames_confirm and proposed != RoleLabel.UNRESOLVED:
            a.phase = RolePhase.CONFIRMED
        else:
            a.phase = RolePhase.TENTATIVE if proposed != RoleLabel.UNRESOLVED else RolePhase.UNRESOLVED

        a._last_reason = reason  # type: ignore[attr-defined]
        a._last_stats = {  # type: ignore[attr-defined]
            "ref_r": ref_r,
            "player_r": player_r,
            "on": on,
            "touch": touch,
            "bench": bench,
            "stands": stands,
            "motion": motion,
            "close_team": close_team,
        }

    def role_of(self, tid: int) -> tuple[RoleLabel, RolePhase, float, str]:
        a = self._t.get(tid)
        if a is None:
            return RoleLabel.UNRESOLVED, RolePhase.UNRESOLVED, 0.0, "missing"
        stats = getattr(a, "_last_stats", {})
        conf = 0.4
        if a.phase == RolePhase.LOCKED:
            conf = 0.85
        elif a.phase == RolePhase.CONFIRMED:
            conf = 0.7
        elif a.phase == RolePhase.TENTATIVE:
            conf = 0.5
        return a.role, a.phase, conf, getattr(a, "_last_reason", "")

    def finalize_rows(self) -> list[dict[str, Any]]:
        rows = []
        for tid, a in self._t.items():
            role, phase, conf, reason = self.role_of(tid)
            stats = getattr(a, "_last_stats", {})
            rows.append(
                {
                    "local_track_id": int(tid),
                    "role": role.value,
                    "phase": phase.value,
                    "role_confidence": conf,
                    "decision_reason": reason,
                    "flips": a.flips,
                    "on_pitch_ratio": stats.get("on", 0.0),
                    "bench_ratio": stats.get("bench", 0.0),
                    "stands_ratio": stats.get("stands", 0.0),
                    "touchline_ratio": stats.get("touch", 0.0),
                    "referee_vote_ratio": stats.get("ref_r", 0.0),
                    "player_vote_ratio": stats.get("player_r", 0.0),
                    "motion_area_px": stats.get("motion", 0.0),
                    "close_team_color": bool(stats.get("close_team", False)),
                    "n_frames": len(a.det),
                }
            )
        return rows
