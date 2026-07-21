"""Referee / assistant referee / fourth official evidence scoring.

Distinguishing the three official types is done positionally: the center
referee lives on the pitch, assistant referees patrol the touchlines, and the
fourth official stays inside the technical area. Kit similarity to the
official color cluster provides the base evidence; position modulates which
official role the evidence supports. Scores are evidence, not decisions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RefereeFeatures:
    """Per-frame features relevant to identifying match officials.

    ``kit_similarity_*`` values are in ``[0, 1]``; ``None`` means unmeasured.
    """

    kit_similarity_officials: float | None = None
    kit_similarity_team0: float | None = None
    kit_similarity_team1: float | None = None
    on_pitch: bool = True
    near_touchline: bool = False
    in_technical_area: bool = False


@dataclass(frozen=True)
class OfficialScores:
    referee: float = 0.0
    assistant_referee: float = 0.0
    fourth_official: float = 0.0

    def __post_init__(self) -> None:
        for value in (self.referee, self.assistant_referee, self.fourth_official):
            if not 0.0 <= value <= 1.0:
                raise ValueError("official scores must be in [0, 1]")

    @property
    def best(self) -> float:
        return max(self.referee, self.assistant_referee, self.fourth_official)


@dataclass(frozen=True)
class RefereeClassifierConfig:
    #: Official-kit similarity below this contributes no official evidence.
    min_kit_similarity: float = 0.35
    #: How strongly a team-kit match suppresses official evidence.
    team_kit_penalty: float = 0.8


class RefereeClassifier:
    """Score official-role evidence for one person in one frame."""

    def __init__(self, config: RefereeClassifierConfig | None = None) -> None:
        self.config = config or RefereeClassifierConfig()

    def score(self, features: RefereeFeatures) -> OfficialScores:
        kit = features.kit_similarity_officials
        if kit is None or kit < self.config.min_kit_similarity:
            return OfficialScores()

        team_kit = max(
            features.kit_similarity_team0 or 0.0,
            features.kit_similarity_team1 or 0.0,
        )
        base = kit * max(0.0, 1.0 - self.config.team_kit_penalty * team_kit)
        if base <= 0.0:
            return OfficialScores()

        if features.on_pitch and not features.near_touchline:
            return OfficialScores(referee=base)
        if features.near_touchline:
            # Touchline position: mostly assistant, some center-ref ambiguity.
            return OfficialScores(referee=base * 0.3, assistant_referee=base)
        if features.in_technical_area:
            return OfficialScores(fourth_official=base)
        # Off pitch, position unclear: weak evidence split across officials.
        return OfficialScores(
            referee=base * 0.2, assistant_referee=base * 0.3, fourth_official=base * 0.3
        )
