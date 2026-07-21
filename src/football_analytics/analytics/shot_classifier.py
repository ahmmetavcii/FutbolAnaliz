"""Streaming, explainable football broadcast shot classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


class ShotLabel(str, Enum):
    MAIN_WIDE = "main_wide"
    MAIN_MEDIUM = "main_medium"
    CLOSE_UP = "close_up"
    REPLAY = "replay"
    CROWD = "crowd"
    BENCH = "bench"
    GRAPHIC = "graphic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PlayerStats:
    """Optional detector summary for one frame.

    ``median_height_ratio`` and ``largest_height_ratio`` are relative to frame
    height, so the classifier behaves consistently at different resolutions.
    """

    count: int = 0
    median_height_ratio: float = 0.0
    largest_height_ratio: float = 0.0


@dataclass(frozen=True)
class ShotClassifierConfig:
    green_hue_min: int = 32
    green_hue_max: int = 95
    green_saturation_min: int = 45
    green_value_min: int = 35
    wide_green_ratio_min: float = 0.32
    medium_green_ratio_min: float = 0.18
    close_up_player_height_ratio: float = 0.35
    graphic_edge_density_min: float = 0.22
    scene_cut_histogram_threshold: float = 0.48
    high_flow_threshold: float = 0.025
    crowd_texture_min: float = 0.12
    bench_player_count_min: int = 2
    minimum_confidence: float = 0.45
    histogram_bins: int = 32
    flow_sample_size: tuple[int, int] = (160, 90)


@dataclass(frozen=True)
class ShotClassification:
    label: ShotLabel
    confidence: float
    green_ratio: float
    histogram_diff: float
    flow: float
    graphic_density: float
    scene_cut: bool
    player_stats: PlayerStats | None = None

    @property
    def shot_type(self) -> str:
        return self.label.value


class ShotClassifier:
    """Classify frames one at a time while retaining only the prior frame."""

    def __init__(self, config: ShotClassifierConfig | None = None) -> None:
        self.config = config or ShotClassifierConfig()
        self._previous_histogram: NDArray[np.float32] | None = None
        self._previous_flow_gray: NDArray[np.uint8] | None = None

    def reset(self) -> None:
        self._previous_histogram = None
        self._previous_flow_gray = None

    def update(
        self,
        frame: NDArray[np.uint8],
        *,
        player_stats: PlayerStats | Mapping[str, float | int] | None = None,
        replay: bool = False,
    ) -> ShotClassification:
        _validate_frame(frame)
        stats = _coerce_player_stats(player_stats)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        histogram = _hsv_histogram(hsv, self.config.histogram_bins)
        histogram_diff = _histogram_difference(self._previous_histogram, histogram)
        scene_cut = (
            self._previous_histogram is not None
            and histogram_diff >= self.config.scene_cut_histogram_threshold
        )

        flow_gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            self.config.flow_sample_size,
            interpolation=cv2.INTER_AREA,
        )
        flow = 0.0 if scene_cut else self._flow_score(flow_gray)
        green_ratio = _green_ratio(hsv, self.config)
        graphic_density = float(np.count_nonzero(cv2.Canny(flow_gray, 80, 180))) / flow_gray.size
        texture = float(cv2.Laplacian(flow_gray, cv2.CV_32F).var()) / (255.0**2)

        label, confidence = self._choose_label(
            green_ratio=green_ratio,
            flow=flow,
            graphic_density=graphic_density,
            texture=texture,
            stats=stats,
            replay=replay,
        )
        if confidence < self.config.minimum_confidence:
            label = ShotLabel.UNKNOWN

        self._previous_histogram = histogram
        self._previous_flow_gray = flow_gray
        return ShotClassification(
            label=label,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            green_ratio=green_ratio,
            histogram_diff=histogram_diff,
            flow=flow,
            graphic_density=graphic_density,
            scene_cut=scene_cut,
            player_stats=stats,
        )

    process_frame = update
    classify = update

    def _flow_score(self, gray: NDArray[np.uint8]) -> float:
        if self._previous_flow_gray is None:
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self._previous_flow_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.linalg.norm(flow, axis=2)
        diagonal = float(np.hypot(*gray.shape))
        return float(np.median(magnitude) / max(diagonal, 1.0))

    def _choose_label(
        self,
        *,
        green_ratio: float,
        flow: float,
        graphic_density: float,
        texture: float,
        stats: PlayerStats | None,
        replay: bool,
    ) -> tuple[ShotLabel, float]:
        cfg = self.config
        if replay:
            return ShotLabel.REPLAY, 0.95
        if green_ratio < cfg.medium_green_ratio_min * 0.5 and (
            graphic_density >= cfg.graphic_edge_density_min
        ):
            return ShotLabel.GRAPHIC, 0.55 + min(graphic_density, 0.4)
        if stats and stats.largest_height_ratio >= cfg.close_up_player_height_ratio:
            return ShotLabel.CLOSE_UP, 0.65 + 0.3 * stats.largest_height_ratio
        if green_ratio >= cfg.wide_green_ratio_min:
            size_evidence = 1.0
            if stats and stats.count:
                size_evidence = 1.0 - min(stats.median_height_ratio / 0.35, 1.0)
            score = 0.55 + 0.3 * min(green_ratio / max(cfg.wide_green_ratio_min, 1e-6), 1.0)
            return ShotLabel.MAIN_WIDE, score * (0.75 + 0.25 * size_evidence)
        if green_ratio >= cfg.medium_green_ratio_min:
            return ShotLabel.MAIN_MEDIUM, 0.55 + 0.35 * green_ratio
        if stats and stats.count >= cfg.bench_player_count_min:
            return ShotLabel.BENCH, 0.55 + min(stats.count, 10) * 0.025
        if texture >= cfg.crowd_texture_min and (stats is None or stats.count == 0):
            return ShotLabel.CROWD, 0.55 + min(texture, 0.35)
        if flow >= cfg.high_flow_threshold and green_ratio < cfg.medium_green_ratio_min:
            return ShotLabel.CLOSE_UP, 0.5 + min(flow * 2.0, 0.3)
        return ShotLabel.UNKNOWN, 0.0


def _coerce_player_stats(
    value: PlayerStats | Mapping[str, float | int] | None,
) -> PlayerStats | None:
    if value is None or isinstance(value, PlayerStats):
        return value
    return PlayerStats(
        count=int(value.get("count", 0)),
        median_height_ratio=float(value.get("median_height_ratio", 0.0)),
        largest_height_ratio=float(value.get("largest_height_ratio", 0.0)),
    )


def player_stats_from_bboxes(
    bboxes: Sequence[Sequence[float]], frame_height: int
) -> PlayerStats:
    if frame_height <= 0:
        raise ValueError("frame_height must be positive")
    heights = [max(0.0, float(box[3]) - float(box[1])) / frame_height for box in bboxes]
    if not heights:
        return PlayerStats()
    return PlayerStats(len(heights), float(np.median(heights)), max(heights))


def _green_ratio(hsv: NDArray[np.uint8], config: ShotClassifierConfig) -> float:
    lower = np.array(
        [config.green_hue_min, config.green_saturation_min, config.green_value_min],
        dtype=np.uint8,
    )
    upper = np.array([config.green_hue_max, 255, 255], dtype=np.uint8)
    return float(np.count_nonzero(cv2.inRange(hsv, lower, upper))) / float(hsv.shape[0] * hsv.shape[1])


def _hsv_histogram(hsv: NDArray[np.uint8], bins: int) -> NDArray[np.float32]:
    histogram = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram, norm_type=cv2.NORM_L1).astype(np.float32)


def _histogram_difference(
    previous: NDArray[np.float32] | None, current: NDArray[np.float32]
) -> float:
    if previous is None:
        return 0.0
    return float(cv2.compareHist(previous, current, cv2.HISTCMP_BHATTACHARYYA))


def _validate_frame(frame: NDArray[np.uint8]) -> None:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("frame must be a non-empty HxWx3 BGR image")
