"""Per-camera calibration registry with strict validity gates.

Each camera contributes an image-to-pitch homography. A calibration is only
usable when the matrix is finite, invertible, and passes the reprojection
error gate; otherwise the camera projects every point to ``None`` (null
pitch coordinates) rather than fabricating positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

PITCH_LENGTH_METERS = 105.0
PITCH_WIDTH_METERS = 68.0

_DETERMINANT_EPSILON = 1e-12


@dataclass(frozen=True)
class CalibrationGates:
    maximum_reprojection_error_m: float = 2.0
    minimum_control_points: int = 4
    pitch_length_m: float = PITCH_LENGTH_METERS
    pitch_width_m: float = PITCH_WIDTH_METERS
    pitch_margin_m: float = 5.0


@dataclass(frozen=True)
class CameraCalibration:
    """Validated (or explicitly invalid) calibration for one camera."""

    camera_id: str
    valid: bool
    homography: NDArray[np.float64] | None
    reprojection_error_m: float | None = None
    invalid_reason: str | None = None

    def image_to_pitch(self, image_xy: Sequence[float]) -> tuple[float, float] | None:
        """Project an image point to pitch meters; None when calibration is invalid."""
        if not self.valid or self.homography is None:
            return None
        point = np.asarray(image_xy, dtype=np.float64)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            return None
        homogeneous = self.homography @ np.array([point[0], point[1], 1.0])
        if abs(homogeneous[2]) < _DETERMINANT_EPSILON:
            return None
        pitch = homogeneous[:2] / homogeneous[2]
        if not np.all(np.isfinite(pitch)):
            return None
        return float(pitch[0]), float(pitch[1])


def validate_calibration(
    camera_id: str,
    homography: Any,
    image_points: Any = None,
    pitch_points: Any = None,
    gates: CalibrationGates | None = None,
) -> CameraCalibration:
    """Run the finite / invertible / reprojection gates on a raw homography.

    When control point correspondences are supplied, the reprojection error is
    computed and gated; without them the matrix-only gates still apply and the
    reprojection error is unknown.
    """
    cfg = gates or CalibrationGates()

    def invalid(reason: str) -> CameraCalibration:
        return CameraCalibration(
            camera_id=camera_id, valid=False, homography=None, invalid_reason=reason
        )

    try:
        matrix = np.asarray(homography, dtype=np.float64)
    except (TypeError, ValueError):
        return invalid("homography is not numeric")
    if matrix.shape != (3, 3):
        return invalid(f"homography must be 3x3, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        return invalid("homography contains non-finite values")
    if abs(float(np.linalg.det(matrix))) < _DETERMINANT_EPSILON:
        return invalid("homography is not invertible")
    if abs(float(matrix[2, 2])) < _DETERMINANT_EPSILON:
        return invalid("homography is not normalizable")
    matrix = matrix / matrix[2, 2]

    reprojection_error: float | None = None
    if image_points is not None or pitch_points is not None:
        image = _points(image_points)
        pitch = _points(pitch_points)
        if image is None or pitch is None:
            return invalid("control points are malformed")
        if len(image) != len(pitch):
            return invalid("control point counts differ")
        if len(image) < cfg.minimum_control_points:
            return invalid(
                f"at least {cfg.minimum_control_points} control points are required"
            )
        margin = cfg.pitch_margin_m
        if not (
            np.all(pitch[:, 0] >= -margin)
            and np.all(pitch[:, 0] <= cfg.pitch_length_m + margin)
            and np.all(pitch[:, 1] >= -margin)
            and np.all(pitch[:, 1] <= cfg.pitch_width_m + margin)
        ):
            return invalid("pitch control points lie outside the pitch")
        homogeneous = np.hstack([image, np.ones((len(image), 1))]) @ matrix.T
        scales = homogeneous[:, 2:3]
        if np.any(np.abs(scales) < _DETERMINANT_EPSILON):
            return invalid("control points project to infinity")
        projected = homogeneous[:, :2] / scales
        reprojection_error = float(np.mean(np.linalg.norm(projected - pitch, axis=1)))
        if reprojection_error > cfg.maximum_reprojection_error_m:
            return invalid(
                f"reprojection error {reprojection_error:.3f} m exceeds gate "
                f"{cfg.maximum_reprojection_error_m:.3f} m"
            )

    return CameraCalibration(
        camera_id=camera_id,
        valid=True,
        homography=matrix,
        reprojection_error_m=reprojection_error,
    )


@dataclass
class CalibrationManager:
    """Holds one calibration per camera; missing cameras behave as invalid."""

    gates: CalibrationGates = field(default_factory=CalibrationGates)
    _calibrations: dict[str, CameraCalibration] = field(default_factory=dict, repr=False)

    def register(
        self,
        camera_id: str,
        homography: Any,
        image_points: Any = None,
        pitch_points: Any = None,
    ) -> CameraCalibration:
        calibration = validate_calibration(
            camera_id, homography, image_points, pitch_points, self.gates
        )
        self._calibrations[camera_id] = calibration
        return calibration

    def register_from_mapping(
        self, camera_id: str, payload: Mapping[str, Any]
    ) -> CameraCalibration:
        return self.register(
            camera_id,
            payload.get("homography"),
            payload.get("image_points"),
            payload.get("pitch_points"),
        )

    def calibration(self, camera_id: str) -> CameraCalibration:
        stored = self._calibrations.get(camera_id)
        if stored is not None:
            return stored
        return CameraCalibration(
            camera_id=camera_id,
            valid=False,
            homography=None,
            invalid_reason="no calibration registered",
        )

    def is_valid(self, camera_id: str) -> bool:
        return self.calibration(camera_id).valid

    def image_to_pitch(
        self, camera_id: str, image_xy: Sequence[float]
    ) -> tuple[float, float] | None:
        return self.calibration(camera_id).image_to_pitch(image_xy)


def _points(raw: Any) -> NDArray[np.float64] | None:
    if raw is None:
        return None
    try:
        points = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        return None
    return points


def calibrate_cameras(
    *,
    prepared_dir=None,
    run_dir=None,
    provider: str = "auto",
    camera_ids: Sequence[str] | None = None,
    manual_calibration=None,
    calibrations: Mapping[str, Any] | None = None,
    output_dir=None,
    force: bool = False,
    **_,
) -> dict[str, Any]:
    """Entry point used by ``scripts/calibrate_cameras.py``.

    Calibration payloads (per-camera homographies with optional control
    points) come from, in order of precedence: the ``calibrations`` argument,
    a ``manual_calibration`` JSON file (``provider="manual"``), or
    ``<prepared_dir>/calibrations.json``. Every payload runs through the
    finite / invertible / reprojection gates; cameras that fail are reported
    invalid and will yield null pitch coordinates downstream.
    """
    import json
    from pathlib import Path

    from football_analytics.utils.io import write_json

    destination_dir = Path(output_dir or run_dir or prepared_dir or ".")
    destination_dir.mkdir(parents=True, exist_ok=True)
    report_path = destination_dir / "calibration_report.json"
    if report_path.is_file() and not force and output_dir is not None:
        return json.loads(report_path.read_text(encoding="utf-8"))

    payload = dict(calibrations or {})
    if not payload and manual_calibration is not None:
        payload = json.loads(Path(manual_calibration).read_text(encoding="utf-8"))
        if isinstance(payload.get("cameras"), Mapping):
            payload = dict(payload["cameras"])
    if not payload and prepared_dir:
        candidate = Path(prepared_dir) / "calibrations.json"
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))

    if camera_ids:
        wanted = {str(camera_id) for camera_id in camera_ids}
        payload = {
            camera_id: item for camera_id, item in payload.items() if str(camera_id) in wanted
        }

    manager = CalibrationManager()
    cameras: dict[str, Any] = {}
    for camera_id, item in payload.items():
        if isinstance(item, Mapping):
            result = manager.register_from_mapping(str(camera_id), item)
        else:
            result = manager.register(str(camera_id), item)
        cameras[str(camera_id)] = {
            "valid": result.valid,
            "invalid_reason": result.invalid_reason,
            "reprojection_error_m": result.reprojection_error_m,
        }
    report = {
        "status": "PASS",
        "provider": provider,
        "cameras": cameras,
        "note": (
            "Invalid calibrations yield null pitch coordinates; values are never invented."
        ),
    }
    write_json(report_path, report)
    return report
