from __future__ import annotations

import numpy as np
import pytest

from football_analytics.multicamera import (
    AudioSyncConfig,
    OffsetSource,
    cross_correlate_offset,
    estimate_audio_sync,
)

FS = 1000.0


def delayed_pair(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Reference starts at master t=10s, target at t=5s -> offset -5s."""
    rng = np.random.default_rng(seed)
    master = rng.standard_normal(int(100 * FS))
    reference = master[int(10 * FS) : int(70 * FS)]
    target = master[int(5 * FS) : int(75 * FS)]
    return reference, target


def test_cross_correlation_recovers_known_offset():
    reference, target = delayed_pair()
    config = AudioSyncConfig(sample_rate_hz=FS, max_offset_seconds=30.0)
    offset, confidence = cross_correlate_offset(reference, target, config)
    assert offset == pytest.approx(-5.0, abs=1.0 / FS)
    assert confidence > 0.5


def test_estimate_returns_offset_confidence_and_drift():
    reference, target = delayed_pair()
    config = AudioSyncConfig(sample_rate_hz=FS, max_offset_seconds=30.0)
    result = estimate_audio_sync("cam2", reference, target, config)
    assert result.estimated_offset_seconds == pytest.approx(-5.0, abs=1.0 / FS)
    assert 0.0 < result.confidence <= 1.0
    assert abs(result.drift_seconds_per_hour) < 1.0
    payload = result.to_dict()
    assert set(payload) == {
        "camera_id",
        "estimated_offset_seconds",
        "confidence",
        "drift_seconds_per_hour",
    }


def test_estimate_converts_to_offset_estimate():
    reference, target = delayed_pair()
    config = AudioSyncConfig(sample_rate_hz=FS, max_offset_seconds=30.0)
    estimate = estimate_audio_sync("cam2", reference, target, config).to_offset_estimate()
    assert estimate.camera_id == "cam2"
    assert estimate.source is OffsetSource.AUDIO
    assert estimate.offset_seconds == pytest.approx(-5.0, abs=1.0 / FS)


def test_silence_yields_zero_confidence():
    config = AudioSyncConfig(sample_rate_hz=FS, max_offset_seconds=5.0)
    offset, confidence = cross_correlate_offset(
        np.zeros(int(10 * FS)), np.zeros(int(10 * FS)), config
    )
    assert offset == 0.0
    assert confidence == 0.0


def test_offset_search_is_bounded():
    reference, target = delayed_pair()
    config = AudioSyncConfig(sample_rate_hz=FS, max_offset_seconds=2.0)
    offset, _ = cross_correlate_offset(reference, target, config)
    assert abs(offset) <= 2.0


def test_rejects_non_finite_waveform():
    config = AudioSyncConfig(sample_rate_hz=FS)
    bad = np.array([0.0, np.nan, 1.0])
    with pytest.raises(ValueError):
        cross_correlate_offset(bad, bad, config)
