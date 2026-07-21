from __future__ import annotations

import numpy as np
import pytest

from football_analytics.multicamera import (
    MatchedEventPair,
    offset_from_activity_signals,
    offset_from_matched_events,
)


def test_matched_events_fit_offset_and_drift():
    camera_times = [0.0, 600.0, 1200.0, 1800.0]
    pairs = [
        MatchedEventPair(reference_time_seconds=t * 1.0001 + 2.0, camera_time_seconds=t)
        for t in camera_times
    ]
    result = offset_from_matched_events("cam2", pairs)
    assert result.offset_seconds == pytest.approx(2.0, abs=1e-6)
    assert result.drift_seconds_per_second == pytest.approx(1e-4, abs=1e-8)
    assert result.confidence == pytest.approx(1.0, abs=1e-6)


def test_single_matched_event_gives_constant_offset():
    result = offset_from_matched_events(
        "cam2", [MatchedEventPair(reference_time_seconds=12.0, camera_time_seconds=10.0)]
    )
    assert result.offset_seconds == pytest.approx(2.0)
    assert result.drift_seconds_per_second == 0.0


def test_noisy_events_reduce_confidence():
    pairs = [
        MatchedEventPair(reference_time_seconds=0.0, camera_time_seconds=0.0),
        MatchedEventPair(reference_time_seconds=10.9, camera_time_seconds=10.0),
        MatchedEventPair(reference_time_seconds=19.1, camera_time_seconds=20.0),
    ]
    result = offset_from_matched_events("cam2", pairs, max_residual_seconds=0.5)
    assert result.confidence < 0.5


def test_activity_signal_alignment_recovers_offset():
    fps = 25.0
    rng = np.random.default_rng(11)
    master = rng.standard_normal(int(100 * fps))
    reference = master[int(10 * fps) : int(70 * fps)]
    camera = master[int(5 * fps) : int(75 * fps)]
    result = offset_from_activity_signals("cam2", reference, camera, fps=fps)
    assert result.offset_seconds == pytest.approx(-5.0, abs=1.0 / fps)
    assert result.confidence > 0.5


def test_matched_events_require_pairs():
    with pytest.raises(ValueError):
        offset_from_matched_events("cam2", [])
