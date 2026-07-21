"""Suppress duplicate events caused by replay / slow-motion segments."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from football_analytics.events.schemas import MatchEvent


@dataclass(frozen=True)
class DuplicateSuppressionConfig:
    match_window_ms: float = 12_000.0
    min_type_match: bool = True


def replay_frame_set(shot_segments: pd.DataFrame | None) -> set[int]:
    if shot_segments is None or shot_segments.empty:
        return set()
    if "shot_type" not in shot_segments.columns:
        return set()
    replay_like = shot_segments[
        shot_segments["shot_type"].astype(str).str.contains("replay|close|graphic", case=False, na=False)
    ]
    return set(int(x) for x in replay_like["frame_id"].tolist())


def suppress_replay_duplicates(
    events: list[MatchEvent],
    *,
    shot_segments: pd.DataFrame | None = None,
    config: DuplicateSuppressionConfig | None = None,
) -> list[MatchEvent]:
    """Keep the earliest live event; drop later replay-window duplicates."""
    cfg = config or DuplicateSuppressionConfig()
    if not events:
        return []
    replay_frames = replay_frame_set(shot_segments)
    kept: list[MatchEvent] = []
    for event in sorted(events, key=lambda item: item.timestamp_ms):
        frame_guess = int(event.timestamp_ms / 40.0)  # approx; used only as soft cue
        in_replay = frame_guess in replay_frames or bool(event.attributes.get("from_replay"))
        duplicate = False
        for prior in kept:
            if cfg.min_type_match and prior.event_type != event.event_type:
                continue
            if abs(prior.timestamp_ms - event.timestamp_ms) <= cfg.match_window_ms:
                # Later event inside match window: treat as replay duplicate.
                if event.timestamp_ms >= prior.timestamp_ms and in_replay:
                    duplicate = True
                    break
                if (
                    event.timestamp_ms >= prior.timestamp_ms
                    and prior.team_id is not None
                    and event.team_id == prior.team_id
                    and abs(prior.timestamp_ms - event.timestamp_ms) < 3_000.0
                ):
                    duplicate = True
                    break
        if not duplicate:
            kept.append(event)
    return kept
