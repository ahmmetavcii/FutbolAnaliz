"""Goalkeeper evidence scoring. Team membership is preserved for GK role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GoalkeeperFeatures:
    kit_distinct_from_teammates: float | None = None
    own_penalty_area_occupancy: float | None = None
    is_deepest_teammate: bool = False
    team_id: int | None = None
    on_pitch: bool = True


@dataclass(frozen=True)
class GoalkeeperClassifierConfig:
    min_kit_distinctiveness: float = 0.45
    min_penalty_occupancy: float = 0.35


class GoalkeeperClassifier:
    def __init__(self, config: GoalkeeperClassifierConfig | None = None) -> None:
        self.config = config or GoalkeeperClassifierConfig()

    def score(self, features: GoalkeeperFeatures) -> float:
        if not features.on_pitch:
            return 0.0
        kit = features.kit_distinct_from_teammates or 0.0
        area = features.own_penalty_area_occupancy or 0.0
        score = 0.55 * kit + 0.35 * area
        if features.is_deepest_teammate:
            score += 0.1
        if features.team_id is None:
            score *= 0.85
        return float(max(0.0, min(1.0, score)))


def score_goalkeeper(features: Mapping[str, Any]) -> dict[str, Any]:
    """Backward-compatible dict API used by lightweight tests/helpers."""
    clf = GoalkeeperClassifier()
    value = clf.score(
        GoalkeeperFeatures(
            kit_distinct_from_teammates=1.0
            if features.get("distinct_kit_from_outfield")
            else features.get("goalkeeper_kit_distinctiveness"),
            own_penalty_area_occupancy=1.0
            if features.get("near_penalty_area")
            else features.get("own_penalty_area_occupancy"),
            is_deepest_teammate=bool(features.get("is_deepest_teammate", False)),
            team_id=features.get("team_id")
            if isinstance(features.get("team_id"), int)
            else (0 if features.get("team_id") else None),
            on_pitch=bool(features.get("on_pitch", True)),
        )
    )
    if features.get("low_field_coverage"):
        value = min(1.0, value + 0.1)
    return {"score": value, "keep_team_id": True}
