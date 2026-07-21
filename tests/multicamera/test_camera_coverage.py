from __future__ import annotations

import pytest

from football_analytics.multicamera import (
    compute_camera_coverage,
    coverage_report,
    find_coverage_gaps,
)


def build_observations(make_obs):
    rows = []
    for i, t in enumerate(range(0, 60, 10)):
        rows.append(
            make_obs(
                camera_id="cam1",
                local_track_id=i % 2,
                reference_time_seconds=float(t),
                pitch_xy_m=(float(t), 0.0),
                detection_confidence=0.8,
            )
        )
    rows.append(
        make_obs(camera_id="cam2", local_track_id=1, reference_time_seconds=0.0, pitch_xy_m=(1.0, 1.0))
    )
    rows.append(
        make_obs(camera_id="cam2", local_track_id=1, reference_time_seconds=10.0, pitch_xy_m=None)
    )
    return rows


def test_per_camera_statistics(make_obs):
    coverage = compute_camera_coverage(build_observations(make_obs))
    cam1 = coverage["cam1"]
    assert cam1.observation_count == 6
    assert cam1.track_count == 2
    assert cam1.span_seconds == pytest.approx(50.0)
    assert cam1.pitch_localized_fraction == 1.0
    assert cam1.mean_detection_confidence == pytest.approx(0.8)

    cam2 = coverage["cam2"]
    assert cam2.observation_count == 2
    assert cam2.pitch_localized_fraction == pytest.approx(0.5)


def test_gap_detection(make_obs):
    gaps = find_coverage_gaps(
        build_observations(make_obs),
        expected_cameras=["cam1", "cam2"],
        duration_seconds=60.0,
        bin_seconds=10.0,
        minimum_active_cameras=2,
    )
    # cam2 stops after t=10, so bins from 20s onward lack a second camera.
    assert [gap["start_seconds"] for gap in gaps] == [20.0, 30.0, 40.0, 50.0]
    assert all(gap["missing_cameras"] == ["cam2"] for gap in gaps)


def test_report_includes_missing_cameras_and_gap_total(make_obs):
    report = coverage_report(
        build_observations(make_obs),
        expected_cameras=["cam1", "cam2", "cam3"],
        duration_seconds=60.0,
    )
    assert report["missing_cameras"] == ["cam3"]
    assert set(report["cameras"]) == {"cam1", "cam2"}
    assert report["gap_seconds"] == 0.0  # min active default is 1 and cam1 covers all bins


def test_invalid_bins_rejected(make_obs):
    with pytest.raises(ValueError):
        find_coverage_gaps([], ["cam1"], duration_seconds=10.0, bin_seconds=0.0)
