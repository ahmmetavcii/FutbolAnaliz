"""Whole-match summary: teams, players, officials, events, quality.

This is a pure aggregation of the component summaries — recomputing after
re-detection or new manual corrections is just calling
:func:`build_match_summary` again with the fresh inputs. Nothing is cached or
mutated. Officials are carried as their own section and are never folded into
team numbers; the score comes exclusively from confirmed goal events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from football_analytics.analytics.goalkeeper_summary import GoalkeeperSummary
from football_analytics.analytics.officials_summary import OfficialSummary
from football_analytics.analytics.player_summary import PlayerSummary
from football_analytics.analytics.quality import MatchQualityReport
from football_analytics.analytics.team_summary import TeamSummary, summarize_teams
from football_analytics.events.event_summary import EventSummary, summarize_events
from football_analytics.events.schemas import MatchEvent


@dataclass(frozen=True)
class MatchSummary:
    teams: Mapping[int, TeamSummary] = field(default_factory=dict)
    players: Mapping[int, PlayerSummary] = field(default_factory=dict)
    goalkeepers: Mapping[int, GoalkeeperSummary] = field(default_factory=dict)
    officials: Mapping[int, OfficialSummary] = field(default_factory=dict)
    events: EventSummary = field(default_factory=EventSummary)
    quality: MatchQualityReport | None = None

    @property
    def score(self) -> dict[int, int]:
        """Confirmed goals per team (candidates never count)."""
        return dict(self.events.confirmed_goals_by_team)


def build_match_summary(
    *,
    player_summaries: Mapping[int, PlayerSummary],
    official_summaries: Mapping[int, OfficialSummary],
    events: Sequence[MatchEvent],
    goalkeeper_summaries: Mapping[int, GoalkeeperSummary] | None = None,
    quality: MatchQualityReport | None = None,
    team_ids: Sequence[int] = (0, 1),
) -> MatchSummary:
    overlap = set(player_summaries) & set(official_summaries)
    if overlap:
        raise ValueError(
            f"tracks {sorted(overlap)} appear as both players and officials; "
            "role voting must resolve to one role per track"
        )
    return MatchSummary(
        teams=summarize_teams(player_summaries),
        players=dict(player_summaries),
        goalkeepers=dict(goalkeeper_summaries or {}),
        officials=dict(official_summaries),
        events=summarize_events(events, team_ids=team_ids),
        quality=quality,
    )
