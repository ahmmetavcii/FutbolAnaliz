"""Pitch zone geometry for Opta-like analytics (attacking-direction aware)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PitchZones:
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    penalty_depth_m: float = 16.5
    penalty_width_m: float = 40.32
    #: Attack direction for team_0: +1 means increasing x is attacking.
    team0_attack_sign: int = 1

    def geometry(self) -> dict[str, Any]:
        """Static pitch region polygons/rects in pitch metres."""
        L, W = self.pitch_length_m, self.pitch_width_m
        third = L / 3.0
        cy = W / 2.0
        half_pw = self.penalty_width_m / 2.0
        return {
            "pitch_length_m": L,
            "pitch_width_m": W,
            "zone_1": {"x0": 0.0, "x1": third, "label": "defensive_third"},
            "zone_2": {"x0": third, "x1": 2 * third, "label": "middle_third"},
            "zone_3": {"x0": 2 * third, "x1": L, "label": "attacking_third"},
            "own_penalty_area_team0": {
                "x0": 0.0,
                "x1": self.penalty_depth_m,
                "y0": cy - half_pw,
                "y1": cy + half_pw,
            },
            "opponent_penalty_area_team0": {
                "x0": L - self.penalty_depth_m,
                "x1": L,
                "y0": cy - half_pw,
                "y1": cy + half_pw,
            },
            "left_channel": {"y0": 2 * W / 3.0, "y1": W},
            "central_channel": {"y0": W / 3.0, "y1": 2 * W / 3.0},
            "right_channel": {"y0": 0.0, "y1": W / 3.0},
            "final_third": {"x0": 2 * third, "x1": L},
            "half_spaces": {
                "left": {"y0": 2 * W / 3.0, "y1": W},
                "right": {"y0": 0.0, "y1": W / 3.0},
            },
            "team0_attack_sign": self.team0_attack_sign,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["geometry"] = self.geometry()
        return payload

    def zone_third(self, x: float, *, team_id: int | None) -> str:
        """Return zone_1/2/3 in the attacking direction of ``team_id``."""
        sign = self.team0_attack_sign if team_id in (0, None, "team_0") else -self.team0_attack_sign
        # Map to attacking-progress coordinate in [0, L]
        progress = x if sign > 0 else (self.pitch_length_m - x)
        third = self.pitch_length_m / 3.0
        if progress < third:
            return "zone_1"
        if progress < 2 * third:
            return "zone_2"
        return "zone_3"

    def in_own_penalty(self, x: float, y: float, *, team_id: int | None) -> bool:
        sign = self.team0_attack_sign if team_id in (0, None, "team_0") else -self.team0_attack_sign
        cy = self.pitch_width_m / 2.0
        half_w = self.penalty_width_m / 2.0
        if abs(y - cy) > half_w:
            return False
        if sign > 0:
            return 0.0 <= x <= self.penalty_depth_m
        return self.pitch_length_m - self.penalty_depth_m <= x <= self.pitch_length_m

    def in_opponent_penalty(self, x: float, y: float, *, team_id: int | None) -> bool:
        other = 1 if team_id in (0, None, "team_0") else 0
        return self.in_own_penalty(x, y, team_id=other)

    def channel(self, y: float) -> str:
        # Looking attack direction: left = high y in standard pitch coords? Use thirds of width.
        w = self.pitch_width_m
        if y < w / 3.0:
            return "right_channel"
        if y < 2 * w / 3.0:
            return "central_channel"
        return "left_channel"

    def labels_at(self, x: float, y: float, *, team_id: int | None) -> dict[str, Any]:
        zone = self.zone_third(x, team_id=team_id)
        return {
            "zone": zone,
            "final_third": zone == "zone_3",
            "own_penalty_area": self.in_own_penalty(x, y, team_id=team_id),
            "opponent_penalty_area": self.in_opponent_penalty(x, y, team_id=team_id),
            "channel": self.channel(y),
            "half_space": self.channel(y) != "central_channel" and zone != "zone_1",
        }


def infer_attack_sign_from_positions(
    team0_xs: list[float], team1_xs: list[float]
) -> int:
    """Heuristic: team that spends more time on the left defends left → attack +x."""
    if not team0_xs or not team1_xs:
        return 1
    m0 = sum(team0_xs) / len(team0_xs)
    m1 = sum(team1_xs) / len(team1_xs)
    # Lower mean x ≈ defending left goal → attacking toward +x
    return 1 if m0 <= m1 else -1
