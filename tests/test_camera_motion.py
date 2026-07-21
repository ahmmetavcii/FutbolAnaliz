"""Unit tests for streaming camera-motion estimation."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_analytics.analytics.calibration import (  # noqa: E402
    CalibrationConfig,
    CalibrationProviderChain,
    CalibrationResult,
    ManualJsonCalibrationProvider,
    MetadataCalibrationProvider,
    PitchOrientation,
    calibration_from_mapping,
)
from football_analytics.analytics.camera_motion import (  # noqa: E402
    CameraMotionConfig,
    CameraMotionEstimator,
)
from football_analytics.analytics.field_coordinates import (  # noqa: E402
    FieldCoordinateTransformer,
    image_to_field,
)
from football_analytics.analytics.shot_classifier import (  # noqa: E402
    PlayerStats,
    ShotClassifier,
    ShotLabel,
)


def _textured_frame(width: int, height: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, size=(height, width), dtype=np.uint8)
    base = cv2.GaussianBlur(noise, (0, 0), 1.4)
    frame = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    for index in range(18):
        x = int((index * 37 + seed * 11) % max(width - 20, 1))
        y = int((index * 53 + seed * 7) % max(height - 20, 1))
        cv2.circle(frame, (x, y), 4, (20 + index * 10, 180, 40), -1)
        cv2.rectangle(
            frame,
            (x + 8, y + 4),
            (min(width - 1, x + 22), min(height - 1, y + 16)),
            (30, 60, 220),
            1,
        )
    return frame


def _translate(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    height, width = frame.shape[:2]
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def test_feature_mask_uses_normalized_border_regions_not_fixed_pixels() -> None:
    estimator = CameraMotionEstimator(CameraMotionConfig(border_region_ratio=0.1))
    small = estimator.feature_mask((100, 200))
    large = estimator.feature_mask((400, 800))

    assert small.shape == (100, 200)
    assert large.shape == (400, 800)
    assert np.count_nonzero(small[:10, :]) == 200 * 10
    assert np.count_nonzero(large[:40, :]) == 800 * 40
    assert np.count_nonzero(small[:, :20]) == 100 * 20
    assert np.count_nonzero(large[:, :80]) == 400 * 80


def test_feature_mask_excludes_bboxes_in_pixel_and_normalized_coords() -> None:
    estimator = CameraMotionEstimator(
        CameraMotionConfig(border_region_ratio=0.2, bbox_padding_ratio=0.0)
    )
    pixel_mask = estimator.feature_mask((100, 100), exclude_bboxes=[(10, 10, 40, 40)])
    assert pixel_mask[25, 25] == 0
    assert pixel_mask[5, 50] == 255

    normalized_mask = estimator.feature_mask(
        (200, 400),
        exclude_bboxes=[(0.25, 0.25, 0.5, 0.5)],
        bboxes_normalized=True,
    )
    assert normalized_mask[75, 150] == 0
    assert normalized_mask[20, 20] == 255


def test_streaming_estimator_reports_translation_and_confidence() -> None:
    estimator = CameraMotionEstimator(
        CameraMotionConfig(
            border_region_ratio=0.25,
            max_corners=400,
            minimum_feature_distance=4,
            minimum_inliers=6,
            minimum_inlier_ratio=0.2,
            scene_cut_reset=False,
        )
    )
    first = _textured_frame(320, 180, seed=3)
    second = _translate(first, dx=6.0, dy=-4.0)

    bootstrap = estimator.update(first)
    assert bootstrap.reset_reason == "first_frame"
    motion = estimator.update(second)

    assert motion.reset_reason is None
    assert motion.inlier_count >= 6
    assert motion.confidence > 0.0
    assert motion.dx == pytest.approx(6.0, abs=1.5)
    assert motion.dy == pytest.approx(-4.0, abs=1.5)
    assert motion.scale == pytest.approx(1.0, abs=0.05)
    assert abs(motion.rotation) < 3.0


def test_variable_resolution_resets_state_without_full_frame_buffer() -> None:
    estimator = CameraMotionEstimator()
    assert estimator._previous_gray is None

    first = estimator.update(_textured_frame(160, 120, seed=1))
    assert first.reset_reason == "first_frame"
    assert estimator._previous_gray is not None
    assert estimator._previous_gray.shape == (120, 160)

    changed = estimator.update(_textured_frame(240, 180, seed=2))
    assert changed.reset_reason == "resolution_change"
    assert changed.dx == 0.0
    assert changed.dy == 0.0
    assert estimator._previous_gray.shape == (180, 240)

    resumed = estimator.update(_translate(_textured_frame(240, 180, seed=2), 3.0, 1.0))
    assert resumed.reset_reason in {
        None,
        "low_inlier_support",
        "insufficient_tracks",
        "insufficient_features",
    }


def test_invalid_calibration_is_null_and_blocks_field_transform(tmp_path: Path) -> None:
    chain = CalibrationProviderChain(
        providers=[
            MetadataCalibrationProvider(),
            ManualJsonCalibrationProvider(tmp_path / "missing.json"),
        ]
    )
    unavailable = chain.calibrate(metadata={})
    assert unavailable.valid is False
    assert unavailable.homography is None
    assert unavailable.invalid_reason is not None

    bad_points = calibration_from_mapping(
        {
            "orientation": PitchOrientation.LEFT_TO_RIGHT.value,
            "image_points": [[0, 0], [1, 0], [1, 1]],
            "pitch_points": [[0, 0], [1, 0], [1, 1]],
        }
    )
    assert bad_points.valid is False

    raw_homography = calibration_from_mapping(
        {
            "orientation": "left_to_right",
            "homography": np.eye(3).tolist(),
        }
    )
    assert raw_homography.valid is False
    assert "reprojection_error" in (raw_homography.invalid_reason or "")

    valid = calibration_from_mapping(
        {
            "orientation": "left_to_right",
            "image_points": [[0, 0], [100, 0], [100, 50], [0, 50]],
            "pitch_points": [[0, 0], [105, 0], [105, 68], [0, 68]],
            "confidence": 0.9,
        },
        config=CalibrationConfig(minimum_visible_pitch_coverage=0.5, minimum_confidence=0.5),
    )
    assert valid.valid is True
    assert valid.homography is not None
    assert valid.pitch_length_m == 105.0
    assert valid.pitch_width_m == 68.0

    transformer = FieldCoordinateTransformer(valid)
    point = transformer.transform_point((50.0, 25.0))
    assert point is not None
    assert point.x == pytest.approx(52.5, abs=1e-3)
    assert point.y == pytest.approx(34.0, abs=1e-3)

    invalid = CalibrationResult.invalid("unavailable")
    assert FieldCoordinateTransformer(invalid).transform_point((10.0, 10.0)) is None
    assert image_to_field((10.0, 10.0), invalid) is None


def test_shot_classifier_streaming_labels_and_optional_player_stats() -> None:
    classifier = ShotClassifier()
    green = np.zeros((180, 320, 3), dtype=np.uint8)
    green[:, :] = (40, 180, 40)
    wide = classifier.update(green)
    assert wide.label is ShotLabel.MAIN_WIDE
    assert wide.green_ratio > 0.3

    close = classifier.update(
        np.full((180, 320, 3), 90, dtype=np.uint8),
        player_stats=PlayerStats(count=1, median_height_ratio=0.5, largest_height_ratio=0.55),
    )
    assert close.label is ShotLabel.CLOSE_UP
    assert close.player_stats is not None
