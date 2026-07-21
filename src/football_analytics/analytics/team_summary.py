"""Team-level aggregation of player summaries.

Inclusion rule (the single place it is enforced for physical totals): a track
contributes to a team total only when its voted role is a countable team role
(outfield player or goalkeeper) *and* it has a team id. Officials, staff,
bench-only substitutes and unknown persons are excluded by construction —
``PlayerSummary.counts_toward_team_totals`` encodes the policy.

Physical totals aggregate only over tracks with valid physical metrics, and
the count of excluded tracks is reported so a "small" total is visibly a
coverage artifact rather than a claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from football_analytics.analytics.player_summary import PlayerSummary


@dataclass(frozen=True)
class TeamSummary:
    team_id: int
    player_count: int
    players_with_valid_metrics: int
    players_without_valid_metrics: int
    total_distance_m: float | None
    max_speed_kmh: float | None
    total_sprints: int
    goalkeeper_track_ids: tuple[int, ...]


def summarize_teams(
    player_summaries: Mapping[int, PlayerSummary],
) -> dict[int, TeamSummary]:
    grouped: dict[int, list[PlayerSummary]] = {}
    for summary in player_summaries.values():
        if not summary.counts_toward_team_totals:
            continue
        assert summary.team_id is not None
        grouped.setdefault(summary.team_id, []).append(summary)

    results: dict[int, TeamSummary] = {}
    for team_id, members in sorted(grouped.items()):
        valid = [m for m in members if m.physical_metrics_valid]
        results[team_id] = TeamSummary(
            team_id=team_id,
            player_count=len(members),
            players_with_valid_metrics=len(valid),
            players_without_valid_metrics=len(members) - len(valid),
            total_distance_m=(
                sum(m.total_distance_m for m in valid if m.total_distance_m is not None)
                if valid
                else None
            ),
            max_speed_kmh=(
                max(m.max_speed_kmh for m in valid if m.max_speed_kmh is not None)
                if valid
                else None
            ),
            total_sprints=sum(m.sprint_count for m in valid),
            goalkeeper_track_ids=tuple(
                sorted(m.track_id for m in members if m.role.value == "goalkeeper")
            ),
        )
    return results
