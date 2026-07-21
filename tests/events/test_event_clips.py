from __future__ import annotations

from football_analytics.events import (
    ClipConfig,
    EventStatus,
    EventType,
    MatchEvent,
    build_clips,
    clip_for_event,
)


def goal_event(event_id: str, ts: float) -> MatchEvent:
    return MatchEvent(
        event_id=event_id,
        event_type=EventType.GOAL,
        status=EventStatus.CANDIDATE_REVIEW_REQUIRED,
        timestamp_ms=ts,
    )


def substitution_event(event_id: str, start: float, end: float) -> MatchEvent:
    return MatchEvent(
        event_id=event_id,
        event_type=EventType.SUBSTITUTION,
        status=EventStatus.CANDIDATE_REVIEW_REQUIRED,
        timestamp_ms=(start + end) / 2.0,
        attributes={"interval_start_ms": start, "interval_end_ms": end},
    )


class TestClipWindows:
    def test_event_gets_padded_window(self):
        clip = clip_for_event(goal_event("goal-1", 60_000.0), ClipConfig(pre_ms=8000, post_ms=5000))
        assert clip.start_ms == 52_000.0
        assert clip.end_ms == 65_000.0

    def test_window_never_starts_before_zero(self):
        clip = clip_for_event(goal_event("goal-1", 2_000.0), ClipConfig(pre_ms=8000, post_ms=5000))
        assert clip.start_ms == 0.0

    def test_substitution_clip_covers_full_interval(self):
        clip = clip_for_event(
            substitution_event("sub-1", 100_000.0, 130_000.0),
            ClipConfig(pre_ms=2000, post_ms=2000),
        )
        assert clip.start_ms == 98_000.0
        assert clip.end_ms == 132_000.0

    def test_replay_segments_are_excluded(self):
        clips = build_clips(
            [goal_event("goal-1", 60_000.0)],
            config=ClipConfig(pre_ms=8000, post_ms=5000),
            replay_segments=[(55_000.0, 58_000.0)],
            merge_overlapping=False,
        )
        assert [(c.start_ms, c.end_ms) for c in clips] == [
            (52_000.0, 55_000.0),
            (58_000.0, 65_000.0),
        ]

    def test_clip_fully_inside_replay_is_dropped(self):
        clips = build_clips(
            [goal_event("goal-1", 60_000.0)],
            config=ClipConfig(pre_ms=1000, post_ms=1000),
            replay_segments=[(50_000.0, 70_000.0)],
        )
        assert clips == []

    def test_overlapping_windows_of_same_type_are_merged(self):
        clips = build_clips(
            [goal_event("goal-1", 60_000.0), goal_event("goal-2", 62_000.0)],
            config=ClipConfig(pre_ms=5000, post_ms=5000),
        )
        assert len(clips) == 1
        assert clips[0].start_ms == 55_000.0
        assert clips[0].end_ms == 67_000.0
        assert "goal-1" in clips[0].event_id and "goal-2" in clips[0].event_id
