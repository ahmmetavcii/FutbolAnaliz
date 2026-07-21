"""Per-player match summaries combining role identity and physical metrics.

Rules enforced here:

- Only team-member roles (outfield player, goalkeeper) get player summaries;
  officials and staff are routed to the officials summary and never appear in
  player tables.
- Physical metrics come from calibrated samples only; when calibration was
  invalid (or too sparse) all physical fields are ``None`` — never zero.
- Summaries are pure functions of their inputs, so they can be recomputed
  after re-detection or manual role corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from football_analytics.analytics.distance import compute_distances
from football_analytics.analytics.player_metrics import PlayerMetricsConfig, PlayerSample
from football_analytics.analytics.speed import compute_speeds
from football_analytics.roles.role_classifier import PersonRole, counts_toward_team_totals
from football_analytics.roles.role_voting import RoleVote


@dataclass(frozen=True)
class PlayerSummary:
    track_id: int
    role: PersonRole
    team_id: int | None
    role_vote_share: float
    total_distance_m: float | None
    max_speed_kmh: float | None
    mean_speed_kmh: float | None
    sprint_count: int
    physical_metrics_valid: bool
    invalid_reason: str | None

    @property
    def counts_toward_team_totals(self) -> bool:
        return counts_toward_team_totals(self.role) and self.team_id is not None


def summarize_players(
    role_votes: Mapping[int, RoleVote],
    team_ids: Mapping[int, int | None],
    samples: Iterable[PlayerSample],
    *,
    config: PlayerMetricsConfig | None = None,
    replay_flags: Sequence[bool] | None = None,
) -> dict[int, PlayerSummary]:
    """Build per-track player summaries for team-member roles only."""
    materialized = list(samples)
    distances = compute_distances(materialized, config, replay_flags=replay_flags)
    speeds = compute_speeds(materialized, config, replay_flags=replay_flags)

    summaries: dict[int, PlayerSummary] = {}
    for track_id, vote in role_votes.items():
        role = vote.role
        if role not in (
            PersonRole.OUTFIELD_PLAYER,
            PersonRole.GOALKEEPER,
            PersonRole.SUBSTITUTE,
        ):
            # Officials, staff and unknown persons never get player summaries.
            continue
        distance = distances.get(track_id)
        speed = speeds.get(track_id)
        valid = bool(
            distance is not None
            and distance.total_distance_m is not None
            and speed is not None
            and speed.max_speed_kmh is not None
        )
        summaries[track_id] = PlayerSummary(
            track_id=track_id,
            role=role,
            team_id=team_ids.get(track_id),
            role_vote_share=vote.vote_share,
            total_distance_m=distance.total_distance_m if distance else None,
            max_speed_kmh=speed.max_speed_kmh if speed else None,
            mean_speed_kmh=speed.mean_speed_kmh if speed else None,
            sprint_count=speed.sprint_count if speed else 0,
            physical_metrics_valid=valid,
            invalid_reason=(
                None
                if valid
                else (
                    (distance.invalid_reason if distance else None)
                    or (speed.invalid_reason if speed else None)
                    or "no_calibrated_samples"
                )
            ),
        )
    return summaries
