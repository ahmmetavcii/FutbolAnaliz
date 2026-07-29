"""Track-level role classification: player / referee / staff / spectator."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RoleConfig:
    min_frames: int = 8
    min_referee_vote_ratio: float = 0.55
    min_referee_on_pitch_ratio: float = 0.25
    max_referee_bench_ratio: float = 0.55
    spectator_stands_ratio: float = 0.6
    staff_bench_ratio: float = 0.5
    max_spectator_motion: float = 8.0  # mean foot displacement px/frame


@dataclass
class _TrackAcc:
    det_votes: list[str] = field(default_factory=list)
    zones: list[str] = field(default_factory=list)
    feet: list[tuple[float, float]] = field(default_factory=list)
    confs: list[float] = field(default_factory=list)


class TrackRoleClassifier:
    def __init__(self, config: RoleConfig | None = None) -> None:
        self.config = config or RoleConfig()
        self._tracks: dict[int, _TrackAcc] = defaultdict(_TrackAcc)

    def update(
        self,
        track_id: int,
        det_class: str,
        zone: str,
        foot_xy: tuple[float, float],
        conf: float,
    ) -> None:
        if track_id < 0:
            return
        st = self._tracks[track_id]
        st.det_votes.append(det_class.lower())
        st.zones.append(zone)
        st.feet.append(foot_xy)
        st.confs.append(float(conf))

    def finalize(self) -> list[dict[str, Any]]:
        rows = []
        for tid, st in self._tracks.items():
            n = len(st.det_votes)
            if n == 0:
                continue
            votes = Counter(st.det_votes)
            zones = Counter(st.zones)
            player_r = votes.get("player", 0) / n
            ref_r = votes.get("referee", 0) / n
            on_pitch = sum(1 for z in st.zones if z in {"ON_PITCH", "PITCH_MARGIN"}) / n
            bench = sum(1 for z in st.zones if z == "BENCH_TECHNICAL_AREA") / n
            stands = sum(1 for z in st.zones if z in {"STANDS", "OFF_FIELD"}) / n
            motion = 0.0
            if len(st.feet) >= 2:
                diffs = [
                    np.hypot(st.feet[i][0] - st.feet[i - 1][0], st.feet[i][1] - st.feet[i - 1][1])
                    for i in range(1, len(st.feet))
                ]
                motion = float(np.mean(diffs))

            role = "UNRESOLVED_PERSON"
            reason = "default"
            conf = 0.4

            if n < self.config.min_frames:
                role = "UNRESOLVED_PERSON"
                reason = "too_short"
                conf = 0.3
            elif stands >= self.config.spectator_stands_ratio and motion <= self.config.max_spectator_motion:
                role = "SPECTATOR"
                reason = "stands_static"
                conf = 0.75
            elif bench >= self.config.staff_bench_ratio and on_pitch < 0.2:
                if ref_r >= 0.4:
                    role = "STAFF"
                    reason = "bench_dominant_ref_votes"
                    conf = 0.7
                else:
                    role = "BENCH_PLAYER" if player_r >= 0.4 else "STAFF"
                    reason = "bench_dominant"
                    conf = 0.65
            elif (
                ref_r >= self.config.min_referee_vote_ratio
                and on_pitch >= self.config.min_referee_on_pitch_ratio
                and bench <= self.config.max_referee_bench_ratio
                and n >= self.config.min_frames
            ):
                role = "REFEREE"
                reason = "track_level_ref_votes+on_pitch"
                conf = min(0.95, 0.5 + 0.5 * ref_r)
            elif ref_r >= 0.35 and (bench > 0.4 or on_pitch < 0.2):
                role = "UNRESOLVED_PERSON"
                reason = "weak_ref_likely_staff"
                conf = 0.45
            elif player_r >= 0.5 and on_pitch >= 0.35:
                role = "PLAYER"
                reason = "player_votes+on_pitch"
                conf = min(0.9, 0.45 + 0.5 * player_r)
            elif player_r >= 0.5 and stands < 0.4:
                role = "PLAYER"
                reason = "player_votes_margin"
                conf = 0.55
            else:
                role = "UNRESOLVED_PERSON"
                reason = "insufficient_evidence"
                conf = 0.4

            rows.append(
                {
                    "local_track_id": int(tid),
                    "role": role,
                    "role_confidence": float(conf),
                    "on_pitch_ratio": float(on_pitch),
                    "bench_ratio": float(bench),
                    "stands_ratio": float(stands),
                    "motion_score": float(motion),
                    "referee_vote_ratio": float(ref_r),
                    "player_vote_ratio": float(player_r),
                    "staff_vote_ratio": float(bench if role in {"STAFF", "BENCH_PLAYER"} else 0.0),
                    "decision_reason": reason,
                    "n_frames": n,
                }
            )
        return rows
