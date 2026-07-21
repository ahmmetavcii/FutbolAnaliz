"""Goalkeeper-specific match summaries.

The goalkeeper is a team member: their physical metrics live in the same
player summary as everyone else and count toward team totals. This module
adds keeper-specific context (penalty-area dwell share) on top of the shared
:class:`~football_analytics.analytics.player_summary.PlayerSummary`, rather
than forking a separate metrics pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from football_analytics.analytics.heatmaps import (
    Heatmap,
    HeatmapConfig,
    HeatmapSample,
    compute_heatmap,
)
from football_analytics.analytics.player_summary import PlayerSummary
from football_analytics.roles.role_classifier import PersonRole


@dataclass(frozen=True)
class GoalkeeperSummary:
    track_id: int
    team_id: int | None
    player_summary: PlayerSummary
    penalty_area_dwell_share: float | None
    heatmap: Heatmap | None

    @property
    def counts_toward_team_totals(self) -> bool:
        # A goalkeeper is a team member; the officials exclusion never
        # applies here.
        return self.player_summary.counts_toward_team_totals


def summarize_goalkeepers(
    player_summaries: Mapping[int, PlayerSummary],
    heatmap_samples: Mapping[int, Iterable[HeatmapSample]] | None = None,
    *,
    config: HeatmapConfig | None = None,
    penalty_area_depth_m: float = 16.5,
) -> dict[int, GoalkeeperSummary]:
    """Extract goalkeeper summaries from player summaries."""
    cfg = config or HeatmapConfig()
    results: dict[int, GoalkeeperSummary] = {}
    for track_id, summary in player_summaries.items():
        if summary.role is not PersonRole.GOALKEEPER:
            continue
        heatmap: Heatmap | None = None
        dwell_share: float | None = None
        if heatmap_samples is not None and track_id in heatmap_samples:
            heatmap = compute_heatmap(heatmap_samples[track_id], cfg)
            if heatmap.total_dwell_s > 0.0:
                dwell_share = _penalty_area_share(heatmap, cfg, penalty_area_depth_m)
        results[track_id] = GoalkeeperSummary(
            track_id=track_id,
            team_id=summary.team_id,
            player_summary=summary,
            penalty_area_dwell_share=dwell_share,
            heatmap=heatmap,
        )
    return results


def _penalty_area_share(
    heatmap: Heatmap, cfg: HeatmapConfig, penalty_area_depth_m: float
) -> float:
    """Dwell share within either penalty-area depth band along the x axis.

    Without knowing attack direction we take the max of the two end bands,
    which is the honest upper bound for "time spent in own penalty area".
    """
    depth_bins = max(1, round(penalty_area_depth_m / cfg.pitch_length_m * cfg.bins_x))
    left = sum(row[ix] for row in heatmap.grid for ix in range(min(depth_bins, cfg.bins_x)))
    right = sum(
        row[ix]
        for row in heatmap.grid
        for ix in range(max(0, cfg.bins_x - depth_bins), cfg.bins_x)
    )
    return max(left, right) / heatmap.total_dwell_s
