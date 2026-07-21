from __future__ import annotations

import pytest

from football_analytics.multicamera import (
    CameraConfig,
    MultiCameraSetup,
    OffsetEstimate,
    OffsetSource,
    TimelineSynchronizer,
    apply_drift_model,
    fit_drift_model,
)


def measurement(t: float, offset: float, confidence: float = 0.9) -> OffsetEstimate:
    return OffsetEstimate(
        camera_id="cam2",
        offset_seconds=offset,
        confidence=confidence,
        source=OffsetSource.AUDIO,
        measured_at_seconds=t,
    )


def test_fit_recovers_linear_drift():
    measurements = [measurement(t, 1.0 + 1e-4 * t) for t in (0.0, 1800.0, 3600.0)]
    model = fit_drift_model("cam2", measurements)
    assert model.drift_rate_seconds_per_second == pytest.approx(1e-4, abs=1e-9)
    assert model.offset_at(0.0) == pytest.approx(1.0, abs=1e-6)
    assert model.correct(3600.0) == pytest.approx(3600.0 + 1.36, abs=1e-6)
    assert model.residual_seconds == pytest.approx(0.0, abs=1e-9)


def test_single_measurement_gives_zero_drift():
    model = fit_drift_model("cam2", [measurement(100.0, 2.0)])
    assert model.drift_rate_seconds_per_second == 0.0
    assert model.base_offset_seconds == pytest.approx(2.0)


def test_low_confidence_measurements_can_be_excluded():
    measurements = [
        measurement(0.0, 1.0),
        measurement(3600.0, 1.36),
        measurement(1800.0, 50.0, confidence=0.05),
    ]
    model = fit_drift_model("cam2", measurements, minimum_confidence=0.5)
    assert model.sample_count == 2
    assert model.drift_rate_seconds_per_second == pytest.approx(1e-4, abs=1e-9)


def test_other_camera_measurements_ignored():
    foreign = OffsetEstimate(
        camera_id="cam9",
        offset_seconds=1.0,
        confidence=1.0,
        source=OffsetSource.AUDIO,
        measured_at_seconds=0.0,
    )
    with pytest.raises(ValueError):
        fit_drift_model("cam2", [foreign])


def test_apply_drift_model_updates_synchronizer():
    setup = MultiCameraSetup(
        cameras=[
            CameraConfig(camera_id="cam1", video_path="a.mp4"),
            CameraConfig(camera_id="cam2", video_path="b.mp4"),
        ]
    )
    sync = TimelineSynchronizer(setup=setup)
    model = fit_drift_model(
        "cam2", [measurement(t, 1e-4 * t) for t in (0.0, 3600.0)]
    )
    apply_drift_model(sync, model)
    assert sync.to_reference_time("cam2", 3600.0) == pytest.approx(3600.36, abs=1e-6)
