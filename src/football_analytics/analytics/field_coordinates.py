"""Guarded image-to-pitch coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from football_analytics.analytics.calibration import CalibrationResult, PitchOrientation


@dataclass(frozen=True)
class FieldCoordinate:
    x: float
    y: float


class FieldCoordinateTransformer:
    """Apply a homography only when calibration passed every validity gate."""

    def __init__(
        self,
        calibration: CalibrationResult,
        *,
        normalize_orientation: bool = False,
        allow_outside_pitch: bool = False,
    ) -> None:
        self.calibration = calibration
        self.normalize_orientation = normalize_orientation
        self.allow_outside_pitch = allow_outside_pitch

    @property
    def valid(self) -> bool:
        return bool(self.calibration.valid and self.calibration.homography is not None)

    def transform_point(self, point: Sequence[float]) -> FieldCoordinate | None:
        if not self.valid:
            return None
        if len(point) != 2:
            raise ValueError("point must contain x and y")
        transformed = self.transform_points([point])
        return transformed[0] if transformed else None

    def transform_points(
        self, points: Sequence[Sequence[float]]
    ) -> list[FieldCoordinate] | None:
        if not self.valid:
            return None
        try:
            source = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("points must be numeric x/y pairs") from exc
        if source.size == 0:
            return []
        if source.ndim != 2 or source.shape[1] != 2 or not np.all(np.isfinite(source)):
            raise ValueError("points must be finite x/y pairs")

        assert self.calibration.homography is not None
        target: NDArray[np.float64] = cv2.perspectiveTransform(
            source.reshape(1, -1, 2), self.calibration.homography
        )[0]
        output: list[FieldCoordinate] = []
        for x_raw, y_raw in target:
            x, y = float(x_raw), float(y_raw)
            if not np.isfinite(x) or not np.isfinite(y):
                return None
            if (
                self.normalize_orientation
                and self.calibration.orientation is PitchOrientation.RIGHT_TO_LEFT
            ):
                x = self.calibration.pitch_length_m - x
                y = self.calibration.pitch_width_m - y
            if not self.allow_outside_pitch and not (
                0.0 <= x <= self.calibration.pitch_length_m
                and 0.0 <= y <= self.calibration.pitch_width_m
            ):
                return None
            output.append(FieldCoordinate(x=x, y=y))
        return output

    transform = transform_point


def image_to_field(
    point: Sequence[float],
    calibration: CalibrationResult,
    *,
    normalize_orientation: bool = False,
) -> FieldCoordinate | None:
    return FieldCoordinateTransformer(
        calibration, normalize_orientation=normalize_orientation
    ).transform_point(point)
