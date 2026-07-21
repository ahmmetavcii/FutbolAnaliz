"""Conservative pitch calibration from explicit metadata or manual JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

PITCH_LENGTH_METERS = 105.0
PITCH_WIDTH_METERS = 68.0


class PitchOrientation(str, Enum):
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"


@dataclass(frozen=True)
class CalibrationConfig:
    pitch_length_m: float = PITCH_LENGTH_METERS
    pitch_width_m: float = PITCH_WIDTH_METERS
    maximum_reprojection_error: float = 8.0
    minimum_visible_pitch_coverage: float = 0.15
    minimum_confidence: float = 0.50
    orientation: PitchOrientation | None = None


@dataclass(frozen=True)
class CalibrationResult:
    valid: bool
    homography: NDArray[np.float64] | None
    provider: str | None
    orientation: PitchOrientation | None
    pitch_length_m: float = PITCH_LENGTH_METERS
    pitch_width_m: float = PITCH_WIDTH_METERS
    reprojection_error: float | None = None
    visible_pitch_coverage: float | None = None
    confidence: float = 0.0
    invalid_reason: str | None = None

    @classmethod
    def invalid(
        cls,
        reason: str,
        *,
        provider: str | None = None,
        config: CalibrationConfig | None = None,
    ) -> "CalibrationResult":
        cfg = config or CalibrationConfig()
        return cls(
            valid=False,
            homography=None,
            provider=provider,
            orientation=cfg.orientation,
            pitch_length_m=cfg.pitch_length_m,
            pitch_width_m=cfg.pitch_width_m,
            invalid_reason=reason,
        )


class CalibrationProvider(Protocol):
    name: str

    def load(self, metadata: Mapping[str, Any] | None = None) -> Mapping[str, Any] | None:
        """Return explicit calibration data, or None when unavailable."""


@dataclass(frozen=True)
class MetadataCalibrationProvider:
    name: str = "metadata"
    key: str = "calibration"

    def load(self, metadata: Mapping[str, Any] | None = None) -> Mapping[str, Any] | None:
        if metadata is None:
            return None
        nested = metadata.get(self.key)
        if isinstance(nested, Mapping):
            return nested
        if "homography" in metadata or (
            "image_points" in metadata and "pitch_points" in metadata
        ):
            return metadata
        return None


@dataclass(frozen=True)
class ManualJsonCalibrationProvider:
    path: Path | str
    name: str = "manual_json"

    def load(self, metadata: Mapping[str, Any] | None = None) -> Mapping[str, Any] | None:
        del metadata
        path = Path(self.path)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, Mapping):
            return None
        calibration = payload.get("calibration", payload)
        return calibration if isinstance(calibration, Mapping) else None


class CalibrationProviderChain:
    """Try only explicit providers; never invent image-to-pitch correspondences."""

    def __init__(
        self,
        providers: Sequence[CalibrationProvider] | None = None,
        config: CalibrationConfig | None = None,
    ) -> None:
        self.config = config or CalibrationConfig()
        self.providers = tuple(providers or (MetadataCalibrationProvider(),))
        unsupported = [
            provider.name
            for provider in self.providers
            if not isinstance(provider, (MetadataCalibrationProvider, ManualJsonCalibrationProvider))
        ]
        if unsupported:
            raise ValueError(
                "only metadata and manual_json calibration providers are supported: "
                + ", ".join(unsupported)
            )

    def calibrate(self, metadata: Mapping[str, Any] | None = None) -> CalibrationResult:
        failures: list[str] = []
        for provider in self.providers:
            payload = provider.load(metadata)
            if payload is None:
                failures.append(f"{provider.name}: unavailable")
                continue
            result = calibration_from_mapping(payload, provider.name, self.config)
            if result.valid:
                return result
            failures.append(f"{provider.name}: {result.invalid_reason}")
        return CalibrationResult.invalid(
            "; ".join(failures) if failures else "no calibration provider configured",
            config=self.config,
        )

    resolve = calibrate


def calibration_from_mapping(
    payload: Mapping[str, Any],
    provider: str = "metadata",
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    cfg = config or CalibrationConfig()
    orientation = _orientation(payload.get("orientation"), cfg.orientation)
    if orientation is None:
        return CalibrationResult.invalid(
            "orientation is missing or invalid", provider=provider, config=cfg
        )

    image_points = _points(payload.get("image_points"))
    pitch_points = _points(payload.get("pitch_points"))
    homography = _homography(payload.get("homography"))
    reprojection_error = _optional_float(payload.get("reprojection_error"))
    coverage = _optional_float(
        payload.get("visible_pitch_coverage", payload.get("coverage"))
    )

    if homography is None:
        if image_points is None or pitch_points is None:
            return _invalid("homography or corresponding points are required", provider, cfg)
        if len(image_points) != len(pitch_points) or len(image_points) < 4:
            return _invalid("at least four matched point pairs are required", provider, cfg)
        if not _pitch_points_valid(pitch_points, cfg):
            return _invalid("pitch points lie outside the canonical pitch", provider, cfg)
        homography_raw, _ = cv2.findHomography(image_points, pitch_points, method=0)
        homography = _homography(homography_raw)
        if homography is None:
            return _invalid("point correspondences are degenerate", provider, cfg)
        projected = cv2.perspectiveTransform(image_points.reshape(1, -1, 2), homography)[0]
        reprojection_error = float(np.mean(np.linalg.norm(projected - pitch_points, axis=1)))
        coverage = _pitch_coverage(pitch_points, cfg)
    else:
        # A matrix without quality metadata cannot be safely accepted.
        if reprojection_error is None or coverage is None:
            return _invalid(
                "homography requires reprojection_error and visible_pitch_coverage",
                provider,
                cfg,
            )
        if image_points is not None and pitch_points is not None:
            if len(image_points) != len(pitch_points) or len(image_points) < 4:
                return _invalid("invalid point pairs", provider, cfg)
            projected = cv2.perspectiveTransform(image_points.reshape(1, -1, 2), homography)[0]
            reprojection_error = float(np.mean(np.linalg.norm(projected - pitch_points, axis=1)))

    confidence = _optional_float(payload.get("confidence"))
    if confidence is None:
        error_quality = max(0.0, 1.0 - reprojection_error / cfg.maximum_reprojection_error)
        coverage_quality = min(1.0, coverage / max(cfg.minimum_visible_pitch_coverage, 1e-9))
        confidence = error_quality * coverage_quality
    if reprojection_error > cfg.maximum_reprojection_error:
        return _invalid("reprojection error exceeds configured maximum", provider, cfg)
    if not 0.0 <= coverage <= 1.0 or coverage < cfg.minimum_visible_pitch_coverage:
        return _invalid("visible pitch coverage is insufficient", provider, cfg)
    if not 0.0 <= confidence <= 1.0 or confidence < cfg.minimum_confidence:
        return _invalid("calibration confidence is insufficient", provider, cfg)

    return CalibrationResult(
        valid=True,
        homography=homography,
        provider=provider,
        orientation=orientation,
        pitch_length_m=cfg.pitch_length_m,
        pitch_width_m=cfg.pitch_width_m,
        reprojection_error=reprojection_error,
        visible_pitch_coverage=coverage,
        confidence=confidence,
    )


def _invalid(reason: str, provider: str, config: CalibrationConfig) -> CalibrationResult:
    return CalibrationResult.invalid(reason, provider=provider, config=config)


def _orientation(
    raw: Any, fallback: PitchOrientation | None
) -> PitchOrientation | None:
    if raw is None:
        return fallback
    try:
        return PitchOrientation(str(raw))
    except ValueError:
        return None


def _points(raw: Any) -> NDArray[np.float32] | None:
    if raw is None:
        return None
    try:
        points = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        return None
    return points


def _homography(raw: Any) -> NDArray[np.float64] | None:
    if raw is None:
        return None
    try:
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None
    if abs(float(np.linalg.det(matrix))) < 1e-12 or abs(float(matrix[2, 2])) < 1e-12:
        return None
    matrix = matrix / matrix[2, 2]
    return matrix


def _optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _pitch_points_valid(points: NDArray[np.float32], config: CalibrationConfig) -> bool:
    return bool(
        np.all(points[:, 0] >= 0.0)
        and np.all(points[:, 0] <= config.pitch_length_m)
        and np.all(points[:, 1] >= 0.0)
        and np.all(points[:, 1] <= config.pitch_width_m)
    )


def _pitch_coverage(points: NDArray[np.float32], config: CalibrationConfig) -> float:
    hull = cv2.convexHull(points)
    area = float(cv2.contourArea(hull))
    return float(np.clip(area / (config.pitch_length_m * config.pitch_width_m), 0.0, 1.0))
