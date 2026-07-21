"""Unit tests for touch deduplication and football ball detector wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_analytics.opta.touch_dedup import deduplicate_touches  # noqa: E402


def test_touch_frames_41_42_43_merge():
    touches = pd.DataFrame(
        [
            {
                "touch_id": "touch-00001",
                "frame_id": 41,
                "timestamp_ms": 1640.0,
                "track_id": 10,
                "global_player_id": 10,
                "team_id": "team_0",
                "confidence": 0.61,
            },
            {
                "touch_id": "touch-00002",
                "frame_id": 42,
                "timestamp_ms": 1680.0,
                "track_id": 10,
                "global_player_id": 10,
                "team_id": "team_0",
                "confidence": 0.67,
            },
            {
                "touch_id": "touch-00003",
                "frame_id": 43,
                "timestamp_ms": 1720.0,
                "track_id": 10,
                "global_player_id": 10,
                "team_id": "team_0",
                "confidence": 0.43,
            },
        ]
    )
    merged = deduplicate_touches(touches, window_ms=300.0)
    assert len(merged) == 1
    assert int(merged.iloc[0]["start_frame"]) == 41
    assert int(merged.iloc[0]["end_frame"]) == 43
    assert int(merged.iloc[0]["peak_confidence_frame"]) == 42
    assert bool(merged.iloc[0]["deduplicated"]) is True


def test_football_ball_model_exists():
    path = Path("/home/ahmet/models/football-ball/yolo-sn-ball-opt.pt")
    assert path.is_file()
