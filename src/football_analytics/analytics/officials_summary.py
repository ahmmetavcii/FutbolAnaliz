"""Match-official summaries, strictly separated from team analytics.

Officials (referee, assistant referees, fourth official) are summarized here
and *only* here. They carry no team id, they are never included in team or
player totals, and nothing in this module feeds team aggregation. Physical
metrics still honour calibration gating (``None`` when invalid).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from football_analytics.analytics.distance import compute_distances
from football_analytics.analytics.player_metrics import PlayerMetricsConfig, PlayerSample
from football_analytics.analytics.speed import compute_speeds
from football_analytics.roles.role_classifier import PersonRole, is_official
from football_analytics.roles.role_voting import RoleVote


@dataclass(frozen=True)
class OfficialSummary:
    track_id: int
    role: PersonRole
    role_vote_share: float
    total_distance_m: float | None
    max_speed_kmh: float | None
    invalid_reason: str | None

    #: Officials never belong to a team and never count toward team totals.
    team_id: None = None

    @property
    def counts_toward_team_totals(self) -> bool:
        return False


def summarize_officials(
    role_votes: Mapping[int, RoleVote],
    samples: Iterable[PlayerSample],
    *,
    config: PlayerMetricsConfig | None = None,
    replay_flags: Sequence[bool] | None = None,
) -> dict[int, OfficialSummary]:
    """Summarize every track whose voted role is an official role."""
    materialized = list(samples)
    distances = compute_distances(materialized, config, replay_flags=replay_flags)
    speeds = compute_speeds(materialized, config, replay_flags=replay_flags)

    results: dict[int, OfficialSummary] = {}
    for track_id, vote in role_votes.items():
        if not is_official(vote.role):
            continue
        distance = distances.get(track_id)
        speed = speeds.get(track_id)
        results[track_id] = OfficialSummary(
            track_id=track_id,
            role=vote.role,
            role_vote_share=vote.vote_share,
            total_distance_m=distance.total_distance_m if distance else None,
            max_speed_kmh=speed.max_speed_kmh if speed else None,
            invalid_reason=(
                (distance.invalid_reason if distance else None)
                or (speed.invalid_reason if speed else None)
            ),
        )
    return results
