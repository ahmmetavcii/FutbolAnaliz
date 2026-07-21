from __future__ import annotations

import json

import numpy as np
import pytest

from football_analytics.multicamera import (
    CalibrationGates,
    CalibrationManager,
    calibrate_cameras,
    validate_calibration,
)

# Maps a 1920x1080 image onto a 105x68 m pitch.
SCALE_H = [
    [105.0 / 1920.0, 0.0, 0.0],
    [0.0, 68.0 / 1080.0, 0.0],
    [0.0, 0.0, 1.0],
]


def test_valid_homography_projects_to_pitch_meters():
    calibration = validate_calibration("cam1", SCALE_H)
    assert calibration.valid
    point = calibration.image_to_pitch((960.0, 540.0))
    assert point == pytest.approx((52.5, 34.0))


def test_control_points_gate_reprojection_error():
    image = [[0.0, 0.0], [1920.0, 0.0], [1920.0, 1080.0], [0.0, 1080.0]]
    pitch = [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]]
    calibration = validate_calibration("cam1", SCALE_H, image, pitch)
    assert calibration.valid
    assert calibration.reprojection_error_m == pytest.approx(0.0, abs=1e-9)


def test_excessive_reprojection_error_invalidates():
    image = [[0.0, 0.0], [1920.0, 0.0], [1920.0, 1080.0], [0.0, 1080.0]]
    shifted_pitch = [[10.0, 0.0], [95.0, 0.0], [95.0, 68.0], [10.0, 68.0]]
    calibration = validate_calibration("cam1", SCALE_H, image, shifted_pitch)
    assert not calibration.valid
    assert "reprojection error" in calibration.invalid_reason
    assert calibration.image_to_pitch((960.0, 540.0)) is None


def test_non_finite_homography_invalid():
    matrix = np.asarray(SCALE_H, dtype=float)
    matrix[0, 0] = np.nan
    calibration = validate_calibration("cam1", matrix)
    assert not calibration.valid
    assert calibration.image_to_pitch((100.0, 100.0)) is None


def test_singular_homography_invalid():
    calibration = validate_calibration("cam1", np.zeros((3, 3)))
    assert not calibration.valid
    assert calibration.image_to_pitch((100.0, 100.0)) is None


def test_wrong_shape_invalid():
    assert not validate_calibration("cam1", np.eye(4)).valid


def test_too_few_control_points_invalid():
    image = [[0.0, 0.0], [1920.0, 0.0], [1920.0, 1080.0]]
    pitch = [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0]]
    gates = CalibrationGates(minimum_control_points=4)
    assert not validate_calibration("cam1", SCALE_H, image, pitch, gates).valid


def test_manager_unregistered_camera_yields_null_coordinates():
    manager = CalibrationManager()
    assert not manager.is_valid("ghost")
    assert manager.image_to_pitch("ghost", (10.0, 10.0)) is None


def test_manager_invalid_registration_yields_null_coordinates():
    manager = CalibrationManager()
    manager.register("cam1", np.zeros((3, 3)))
    assert not manager.is_valid("cam1")
    assert manager.image_to_pitch("cam1", (10.0, 10.0)) is None


def test_calibrate_cameras_entry_point(tmp_path):
    (tmp_path / "calibrations.json").write_text(
        json.dumps(
            {
                "cam1": {"homography": SCALE_H},
                "cam2": {"homography": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]},
            }
        ),
        encoding="utf-8",
    )
    report = calibrate_cameras(prepared_dir=tmp_path, provider="manual")
    assert report["status"] == "PASS"
    assert report["cameras"]["cam1"]["valid"] is True
    assert report["cameras"]["cam2"]["valid"] is False
    assert (tmp_path / "calibration_report.json").is_file()


def test_calibrate_cameras_manual_calibration_file(tmp_path):
    manual = tmp_path / "manual.json"
    manual.write_text(
        json.dumps({"cameras": {"cam1": {"homography": SCALE_H}}}), encoding="utf-8"
    )
    report = calibrate_cameras(
        prepared_dir=tmp_path,
        provider="manual",
        manual_calibration=manual,
        output_dir=tmp_path / "out",
        force=True,
    )
    assert report["cameras"]["cam1"]["valid"] is True
    assert (tmp_path / "out" / "calibration_report.json").is_file()
