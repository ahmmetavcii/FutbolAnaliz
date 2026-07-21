"""Scene-cut reset behavior for camera motion and shot classification."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_analytics.analytics.camera_motion import (  # noqa: E402
    CameraMotionConfig,
    CameraMotionEstimator,
)
from football_analytics.analytics.shot_classifier import (  # noqa: E402
    ShotClassifier,
    ShotClassifierConfig,
)


def _textured_frame(width: int, height: int, seed: int, tint: tuple[int, int, int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, size=(height, width), dtype=np.uint8)
    base = cv2.GaussianBlur(noise, (0, 0), 1.2)
    frame = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    overlay = np.full_like(frame, tint)
    blended = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0.0)
    for index in range(24):
        x = (index * 41 + seed * 13) % max(width - 12, 1)
        y = (index * 29 + seed * 17) % max(height - 12, 1)
        cv2.circle(blended, (x, y), 5, (255 - tint[0], tint[1], 255 - tint[2]), -1)
    return blended


def test_camera_motion_resets_on_scene_cut_and_does_not_carry_prior_motion() -> None:
    estimator = CameraMotionEstimator(
        CameraMotionConfig(
            border_region_ratio=0.2,
            scene_cut_reset=True,
            scene_cut_histogram_threshold=0.35,
            minimum_inliers=6,
            minimum_inlier_ratio=0.2,
        )
    )
    scene_a = _textured_frame(320, 180, seed=11, tint=(20, 160, 40))
    scene_a_shift = cv2.warpAffine(
        scene_a,
        np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], dtype=np.float32),
        (320, 180),
        borderMode=cv2.BORDER_REFLECT101,
    )
    scene_b = _textured_frame(320, 180, seed=99, tint=(180, 40, 40))

    estimator.update(scene_a)
    before_cut = estimator.update(scene_a_shift)
    assert before_cut.reset_reason != "scene_cut"
    assert before_cut.histogram_diff < 0.35

    cut = estimator.update(scene_b)
    assert cut.scene_cut is True
    assert cut.reset_reason == "scene_cut"
    assert cut.dx == 0.0
    assert cut.dy == 0.0
    assert cut.rotation == 0.0
    assert cut.scale == 1.0
    assert cut.inlier_count == 0
    assert cut.confidence == 0.0
    assert cut.histogram_diff >= 0.35


def test_camera_motion_scene_cut_reset_can_be_disabled() -> None:
    estimator = CameraMotionEstimator(
        CameraMotionConfig(
            border_region_ratio=0.2,
            scene_cut_reset=False,
            scene_cut_histogram_threshold=0.35,
            minimum_inliers=4,
            minimum_inlier_ratio=0.1,
        )
    )
    scene_a = _textured_frame(240, 160, seed=3, tint=(30, 150, 40))
    scene_b = _textured_frame(240, 160, seed=77, tint=(200, 30, 30))
    estimator.update(scene_a)
    result = estimator.update(scene_b)
    assert result.reset_reason != "scene_cut"
    assert result.scene_cut is False


def test_shot_classifier_marks_scene_cut_from_histogram_diff() -> None:
    classifier = ShotClassifier(
        ShotClassifierConfig(scene_cut_histogram_threshold=0.35, minimum_confidence=0.0)
    )
    first = _textured_frame(240, 160, seed=1, tint=(20, 170, 40))
    second = _textured_frame(240, 160, seed=2, tint=(20, 170, 45))
    cut_frame = _textured_frame(240, 160, seed=50, tint=(40, 40, 210))

    first_result = classifier.update(first)
    assert first_result.scene_cut is False
    assert first_result.histogram_diff == 0.0

    continuous = classifier.update(second)
    assert continuous.scene_cut is False

    cut = classifier.update(cut_frame)
    assert cut.scene_cut is True
    assert cut.histogram_diff >= 0.35
    assert cut.flow == 0.0
