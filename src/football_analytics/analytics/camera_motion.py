"""Resolution-independent streaming camera-motion estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraMotionConfig:
    border_region_ratio: float = 0.12
    max_corners: int = 300
    quality_level: float = 0.01
    minimum_feature_distance: float = 7.0
    forward_backward_error: float = 1.5
    ransac_reprojection_threshold: float = 2.5
    minimum_inlier_ratio: float = 0.35
    minimum_inliers: int = 8
    scene_cut_reset: bool = True
    scene_cut_histogram_threshold: float = 0.48
    bbox_padding_ratio: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.border_region_ratio <= 0.5:
            raise ValueError("border_region_ratio must be in (0, 0.5]")
        if self.max_corners < 3 or self.minimum_inliers < 3:
            raise ValueError("at least three corners/inliers are required")


@dataclass(frozen=True)
class CameraMotionEstimate:
    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0
    scale: float = 1.0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    tracked_count: int = 0
    confidence: float = 0.0
    histogram_diff: float = 0.0
    scene_cut: bool = False
    reset_reason: str | None = None

    @property
    def dx_pixel(self) -> float:
        return self.dx

    @property
    def dy_pixel(self) -> float:
        return self.dy

    @property
    def rotation_deg(self) -> float:
        return self.rotation

    @property
    def valid(self) -> bool:
        return self.reset_reason is None and self.inlier_count > 0


class CameraMotionEstimator:
    """Estimate global affine motion without buffering a frame sequence."""

    def __init__(self, config: CameraMotionConfig | None = None) -> None:
        self.config = config or CameraMotionConfig()
        self._previous_gray: NDArray[np.uint8] | None = None
        self._previous_histogram: tuple[NDArray[np.float32], ...] | None = None
        self._previous_mask: NDArray[np.uint8] | None = None

    def reset(self) -> None:
        self._previous_gray = None
        self._previous_histogram = None
        self._previous_mask = None

    def update(
        self,
        frame: NDArray[np.uint8],
        *,
        exclude_bboxes: Sequence[Sequence[float]] | None = None,
        bboxes_normalized: bool = False,
    ) -> CameraMotionEstimate:
        _validate_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        histogram = _color_histogram(frame)
        mask = self.feature_mask(
            frame.shape[:2],
            exclude_bboxes=exclude_bboxes,
            bboxes_normalized=bboxes_normalized,
        )

        if self._previous_gray is None:
            self._set_previous(gray, histogram, mask)
            return CameraMotionEstimate(reset_reason="first_frame")
        if self._previous_gray.shape != gray.shape:
            self._set_previous(gray, histogram, mask)
            return CameraMotionEstimate(reset_reason="resolution_change")

        histogram_diff = _histogram_difference(self._previous_histogram, histogram)
        if (
            self.config.scene_cut_reset
            and histogram_diff >= self.config.scene_cut_histogram_threshold
        ):
            self._set_previous(gray, histogram, mask)
            return CameraMotionEstimate(
                histogram_diff=histogram_diff,
                scene_cut=True,
                reset_reason="scene_cut",
            )

        estimate = self._estimate(gray, mask, histogram_diff)
        self._set_previous(gray, histogram, mask)
        return estimate

    process_frame = update
    estimate = update

    def feature_mask(
        self,
        frame_shape: tuple[int, int],
        *,
        exclude_bboxes: Sequence[Sequence[float]] | None = None,
        bboxes_normalized: bool = False,
    ) -> NDArray[np.uint8]:
        """Build normalized border bands and remove dynamic-object boxes."""
        height, width = frame_shape
        if height <= 0 or width <= 0:
            raise ValueError("frame dimensions must be positive")
        y_border = max(1, round(height * self.config.border_region_ratio))
        x_border = max(1, round(width * self.config.border_region_ratio))
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[:y_border, :] = 255
        mask[height - y_border :, :] = 255
        mask[:, :x_border] = 255
        mask[:, width - x_border :] = 255

        padding = round(min(height, width) * self.config.bbox_padding_ratio)
        for raw_box in exclude_bboxes or ():
            if len(raw_box) != 4:
                raise ValueError("each bbox must contain x1, y1, x2, y2")
            x1, y1, x2, y2 = (float(value) for value in raw_box)
            if bboxes_normalized:
                x1, x2 = x1 * width, x2 * width
                y1, y2 = y1 * height, y2 * height
            left = int(np.clip(np.floor(min(x1, x2)) - padding, 0, width))
            right = int(np.clip(np.ceil(max(x1, x2)) + padding, 0, width))
            top = int(np.clip(np.floor(min(y1, y2)) - padding, 0, height))
            bottom = int(np.clip(np.ceil(max(y1, y2)) + padding, 0, height))
            mask[top:bottom, left:right] = 0
        return mask

    def _estimate(
        self,
        current_gray: NDArray[np.uint8],
        current_mask: NDArray[np.uint8],
        histogram_diff: float,
    ) -> CameraMotionEstimate:
        assert self._previous_gray is not None
        previous_points = cv2.goodFeaturesToTrack(
            self._previous_gray,
            mask=self._previous_mask,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.minimum_feature_distance,
        )
        if previous_points is None or len(previous_points) < 3:
            return CameraMotionEstimate(
                histogram_diff=histogram_diff, reset_reason="insufficient_features"
            )

        current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            self._previous_gray, current_gray, previous_points, None
        )
        if current_points is None or forward_status is None:
            return CameraMotionEstimate(
                histogram_diff=histogram_diff, reset_reason="tracking_failed"
            )
        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, self._previous_gray, current_points, None
        )
        if backward_points is None or backward_status is None:
            return CameraMotionEstimate(
                histogram_diff=histogram_diff, reset_reason="tracking_failed"
            )

        previous_xy = previous_points.reshape(-1, 2)
        current_xy = current_points.reshape(-1, 2)
        backward_xy = backward_points.reshape(-1, 2)
        status = forward_status.reshape(-1).astype(bool) & backward_status.reshape(-1).astype(bool)
        fb_error = np.linalg.norm(previous_xy - backward_xy, axis=1)
        status &= np.isfinite(fb_error) & (fb_error <= self.config.forward_backward_error)
        rounded = np.rint(current_xy).astype(np.int32)
        inside = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < current_gray.shape[1])
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < current_gray.shape[0])
        )
        allowed = np.zeros(len(rounded), dtype=bool)
        allowed[inside] = current_mask[rounded[inside, 1], rounded[inside, 0]] > 0
        status &= allowed
        source = previous_xy[status]
        target = current_xy[status]
        tracked_count = len(source)
        if tracked_count < 3:
            return CameraMotionEstimate(
                tracked_count=tracked_count,
                histogram_diff=histogram_diff,
                reset_reason="insufficient_tracks",
            )

        affine, inlier_mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_reprojection_threshold,
        )
        if affine is None or inlier_mask is None:
            return CameraMotionEstimate(
                tracked_count=tracked_count,
                histogram_diff=histogram_diff,
                reset_reason="ransac_failed",
            )
        inlier_count = int(np.count_nonzero(inlier_mask))
        inlier_ratio = inlier_count / tracked_count
        a, b, dx = (float(value) for value in affine[0])
        c, d, dy = (float(value) for value in affine[1])
        scale = float(np.sqrt(max(0.0, a * a + c * c)))
        rotation = float(np.degrees(np.arctan2(c, a)))
        enough = (
            inlier_count >= self.config.minimum_inliers
            and inlier_ratio >= self.config.minimum_inlier_ratio
        )
        feature_support = min(1.0, inlier_count / max(self.config.minimum_inliers * 2, 1))
        confidence = inlier_ratio * feature_support if enough else 0.0
        return CameraMotionEstimate(
            dx=dx,
            dy=dy,
            rotation=rotation,
            scale=scale if np.isfinite(scale) and scale > 0 else 1.0,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            tracked_count=tracked_count,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            histogram_diff=histogram_diff,
            reset_reason=None if enough else "low_inlier_support",
        )

    def _set_previous(
        self,
        gray: NDArray[np.uint8],
        histogram: tuple[NDArray[np.float32], ...],
        mask: NDArray[np.uint8],
    ) -> None:
        self._previous_gray = gray
        self._previous_histogram = histogram
        self._previous_mask = mask


def _color_histogram(frame: NDArray[np.uint8]) -> tuple[NDArray[np.float32], ...]:
    channels: list[NDArray[np.float32]] = []
    for channel in range(3):
        histogram = cv2.calcHist([frame], [channel], None, [32], [0, 256])
        channels.append(
            cv2.normalize(histogram, histogram, norm_type=cv2.NORM_L1).astype(np.float32)
        )
    return tuple(channels)


def _histogram_difference(
    previous: tuple[NDArray[np.float32], ...] | None,
    current: tuple[NDArray[np.float32], ...],
) -> float:
    if previous is None:
        return 0.0
    return float(
        np.mean(
            [
                cv2.compareHist(old, new, cv2.HISTCMP_BHATTACHARYYA)
                for old, new in zip(previous, current)
            ]
        )
    )


def _validate_frame(frame: NDArray[np.uint8]) -> None:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("frame must be a non-empty HxWx3 BGR image")
