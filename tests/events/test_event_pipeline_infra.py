"""Infrastructure tests for wired event detection (honest empty on football.mp4)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from football_analytics.events.ball_trajectory import load_ball_trajectory_from_ball_state
from football_analytics.events.duplicate_suppression import suppress_replay_duplicates
from football_analytics.events.goal_detector import GoalDetector, GoalSignals
from football_analytics.events.orchestrator import match_events_frame, run_event_detection
from football_analytics.events.possession_chain import build_possession_chain, passes_from_chain
from football_analytics.events.schemas import EventStatus, EventType, MatchEvent
from football_analytics.events.scoreboard_ocr import run_scoreboard_ocr
from football_analytics.events.scorer_detector import detect_scorer
from football_analytics.events.spotting_adapter import SpottingAdapter, SpottingAdapterConfig
from football_analytics.events.touch_inference import infer_touches


def test_ball_trajectory_schema_empty():
    frame = load_ball_trajectory_from_ball_state(pd.DataFrame())
    assert list(frame.columns) == [
        "frame_id",
        "timestamp_ms",
        "ball_x_pixel",
        "ball_y_pixel",
        "pitch_x",
        "pitch_y",
        "ball_confidence",
        "visible",
        "interpolated",
        "source_camera",
    ]


def test_touch_inference_no_ball_no_touches():
    touches = infer_touches(pd.DataFrame(), pd.DataFrame())
    assert touches.empty


def test_possession_chain_and_passes():
    touches = pd.DataFrame(
        [
            {
                "touch_id": "t1",
                "frame_id": 1,
                "timestamp_ms": 1000.0,
                "global_player_id": 1,
                "track_id": 10,
                "team_id": "team_0",
                "confidence": 0.8,
                "controlled_touch": True,
                "deflection": False,
                "candidate_only": False,
                "distance_px": 10.0,
                "distance_m": 1.0,
            },
            {
                "touch_id": "t2",
                "frame_id": 20,
                "timestamp_ms": 2000.0,
                "global_player_id": 2,
                "track_id": 11,
                "team_id": "team_0",
                "confidence": 0.8,
                "controlled_touch": True,
                "deflection": False,
                "candidate_only": False,
                "distance_px": 12.0,
                "distance_m": 1.1,
            },
        ]
    )
    chain = build_possession_chain(touches)
    assert len(chain) == 1
    assert passes_from_chain(chain)[0].passer_track_id == 10


def test_scoreboard_ocr_never_invents_change(tmp_path: Path):
    # Synthetic tiny video via numpy+cv2 if available
    import cv2
    import numpy as np

    video = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 64))
    for _ in range(15):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    frame = run_scoreboard_ocr(video)
    assert "score_change" in frame.columns
    assert int(frame["score_change"].fillna(False).sum()) == 0


def test_scorer_unresolved_when_weak():
    touches = pd.DataFrame(
        [
            {
                "touch_id": "t1",
                "frame_id": 1,
                "timestamp_ms": 900.0,
                "global_player_id": 7,
                "track_id": 3,
                "team_id": "team_0",
                "confidence": 0.2,
                "controlled_touch": False,
                "deflection": False,
                "candidate_only": True,
                "distance_px": 40.0,
                "distance_m": 2.0,
            }
        ]
    )
    result = detect_scorer(goal_timestamp_ms=1000.0, touches=touches)
    assert result.scorer_global_player_id is None
    assert result.scorer_status == "unresolved"


def test_own_goal_and_penalty_no_assist_via_goal_schema():
    with pytest.raises(ValueError):
        MatchEvent(
            event_id="g1",
            event_type=EventType.GOAL,
            status=EventStatus.AUTO_CONFIRMED,
            timestamp_ms=1000.0,
            assist_track_id=2,
            attributes={"own_goal": True},
        )


def test_candidate_not_in_confirmed_totals():
    detector = GoalDetector()
    # Single weak signal -> below candidate or unresolved; two weak still capped.
    event = detector.detect(
        GoalSignals(timestamp_ms=1000.0, celebration_break_score=0.5, from_replay=False)
    )
    assert event is not None
    assert event.status is not EventStatus.AUTO_CONFIRMED


def test_replay_duplicate_suppression():
    live = MatchEvent(
        event_id="g1",
        event_type=EventType.GOAL,
        status=EventStatus.CANDIDATE_REVIEW_REQUIRED,
        timestamp_ms=1000.0,
        confidence=0.5,
        attributes={"from_replay": False},
    )
    replay = MatchEvent(
        event_id="g2",
        event_type=EventType.GOAL,
        status=EventStatus.CANDIDATE_REVIEW_REQUIRED,
        timestamp_ms=1500.0,
        confidence=0.5,
        attributes={"from_replay": True},
    )
    kept = suppress_replay_duplicates([live, replay])
    assert [e.event_id for e in kept] == ["g1"]


def test_spotting_adapter_disabled():
    adapter = SpottingAdapter(SpottingAdapterConfig(enabled=False))
    assert adapter.predict_events(Path("/tmp/nope.mp4"), Path("/tmp/out.json")) == []


def test_match_events_frame_empty():
    frame = match_events_frame([], created_at="now")
    assert "event_id" in frame.columns
    assert frame.empty


@pytest.mark.skipif(
    not Path("/home/ahmet/workspace/full_match_runs/run_football_short_model/pipeline").exists(),
    reason="football short pipeline artifacts missing",
)
def test_football_mp4_zero_event_honesty_on_existing_artifacts(tmp_path: Path):
    pipeline_root = Path("/home/ahmet/workspace/full_match_runs/run_football_short_model/pipeline")
    runs = sorted(pipeline_root.glob("run_*"))
    assert runs, "no pipeline run"
    src = runs[-1]
    # Copy minimal artifacts into temp run dir
    import shutil

    dst = tmp_path / "run"
    shutil.copytree(src, dst)
    # Disable spotting worker for unit speed; honesty still holds with geometry alone.
    manifest = run_event_detection(
        dst,
        config={"spotting": {"enabled": False}, "clips": {"export": False}},
        video_path=dst / "input" / "test_clip.mp4",
    )
    assert manifest["confirmed_event_count"] == 0
    assert manifest["candidate_event_count"] == 0
    assert manifest["events_reason"] == "no_supported_event_detected"
    events = pd.read_parquet(dst / "match_events.parquet")
    assert len(events) == 0
    assert (dst / "stage_manifests" / "event_detection.json").is_file()


def test_panel_source_has_match_events_section():
    source = Path("/home/ahmet/projects/football-analytics/apps/full_match_panel.py").read_text()
    assert "MAÇ OLAYLARI" in source
    assert "Onayla" in source
    assert "recompute_match_events.py" in source
