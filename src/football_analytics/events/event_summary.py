"""Recompute-friendly event summaries.

Summaries are pure functions of (events, review log): every number can be
regenerated from scratch after re-detection or new corrections. Counting
rules:

- Only AUTO_CONFIRMED / MANUALLY_CONFIRMED events count; candidates and
  unresolved events are reported separately and never counted as confirmed.
- Goals with unresolved scorers count for the team but attribute to no one.
- Own goals count for the *opposing* team's score when both team ids are
  known, and never produce a scorer credit in the scorer table.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from football_analytics.events.event_review import ReviewLog, apply_review
from football_analytics.events.schemas import (
    EventStatus,
    EventType,
    MatchEvent,
    is_confirmed,
)


@dataclass(frozen=True)
class EventSummary:
    """Aggregate view over one match's events."""

    confirmed_goals_by_team: Mapping[int, int] = field(default_factory=dict)
    confirmed_goals_by_scorer: Mapping[int, int] = field(default_factory=dict)
    confirmed_assists_by_player: Mapping[int, int] = field(default_factory=dict)
    confirmed_shots_by_team: Mapping[int, int] = field(default_factory=dict)
    confirmed_substitutions_by_team: Mapping[int, int] = field(default_factory=dict)
    unattributed_confirmed_goals: int = 0
    status_counts: Mapping[EventStatus, int] = field(default_factory=dict)
    pending_review_event_ids: tuple[str, ...] = ()
    unresolved_event_ids: tuple[str, ...] = ()


def summarize_events(
    events: Sequence[MatchEvent],
    *,
    team_ids: Iterable[int] = (0, 1),
) -> EventSummary:
    known_teams = set(team_ids)
    goals_by_team: Counter[int] = Counter()
    goals_by_scorer: Counter[int] = Counter()
    assists_by_player: Counter[int] = Counter()
    shots_by_team: Counter[int] = Counter()
    subs_by_team: Counter[int] = Counter()
    unattributed_goals = 0
    status_counts: Counter[EventStatus] = Counter()
    pending: list[str] = []
    unresolved: list[str] = []

    for event in events:
        status_counts[event.status] += 1
        if event.status is EventStatus.CANDIDATE_REVIEW_REQUIRED:
            pending.append(event.event_id)
        elif event.status is EventStatus.UNRESOLVED:
            unresolved.append(event.event_id)
        if not is_confirmed(event.status):
            continue

        if event.event_type is EventType.GOAL:
            credited_team = _credited_team(event, known_teams)
            if credited_team is not None:
                goals_by_team[credited_team] += 1
            if event.scorer_track_id is not None and not event.own_goal:
                goals_by_scorer[event.scorer_track_id] += 1
            else:
                unattributed_goals += 1
            if event.assist_track_id is not None:
                assists_by_player[event.assist_track_id] += 1
        elif event.event_type is EventType.ASSIST:
            if event.assist_track_id is not None:
                assists_by_player[event.assist_track_id] += 1
        elif event.event_type is EventType.SHOT:
            if event.team_id is not None:
                shots_by_team[event.team_id] += 1
        elif event.event_type is EventType.SUBSTITUTION:
            if event.team_id is not None:
                subs_by_team[event.team_id] += 1

    return EventSummary(
        confirmed_goals_by_team=dict(goals_by_team),
        confirmed_goals_by_scorer=dict(goals_by_scorer),
        confirmed_assists_by_player=dict(assists_by_player),
        confirmed_shots_by_team=dict(shots_by_team),
        confirmed_substitutions_by_team=dict(subs_by_team),
        unattributed_confirmed_goals=unattributed_goals,
        status_counts=dict(status_counts),
        pending_review_event_ids=tuple(pending),
        unresolved_event_ids=tuple(unresolved),
    )


def _credited_team(event: MatchEvent, known_teams: set[int]) -> int | None:
    """Own goals credit the opposing team when it can be identified."""
    if event.team_id is None:
        return None
    if not event.own_goal:
        return event.team_id
    others = known_teams - {event.team_id}
    if len(others) == 1:
        return next(iter(others))
    return None


def recompute_summary(
    events: Sequence[MatchEvent],
    review_log: ReviewLog,
    *,
    team_ids: Iterable[int] = (0, 1),
) -> EventSummary:
    """Re-apply the manual review log to fresh detections, then summarize.

    This is the recompute path: detectors may be re-run at any time, and this
    function reproduces the human-reviewed summary deterministically.
    """
    reviewed = apply_review(events, review_log)
    return summarize_events(reviewed.events, team_ids=team_ids)
