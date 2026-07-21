"""Bounding-box geometry with explicit image-boundary quality measures."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


@dataclass(frozen=True, slots=True)
class CropQuality:
    """Quality signals for a clipped player crop."""

    valid: bool
    visible_fraction: float
    foot_point_confidence: float
    out_of_frame: bool
    area: float

    @property
    def score(self) -> float:
        if not self.valid:
            return 0.0
        return min(self.visible_fraction, self.foot_point_confidence)


@dataclass(frozen=True, slots=True)
class BBox:
    """A continuous ``(x1, y1, x2, y2)`` box using half-open bounds."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "BBox":
        if len(values) != 4:
            raise ValueError("bbox must contain exactly four coordinates")
        return cls(*(float(value) for value in values))

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def finite(self) -> bool:
        return all(isfinite(value) for value in self.as_tuple())

    def is_valid(self, min_area: float = 1.0) -> bool:
        if min_area < 0:
            raise ValueError("min_area must be non-negative")
        return self.finite and self.x2 > self.x1 and self.y2 > self.y1 and self.area >= min_area

    def clip(self, frame_width: int | float, frame_height: int | float) -> "BBox":
        width, height = _frame_size(frame_width, frame_height)
        if not self.finite:
            return BBox(0.0, 0.0, 0.0, 0.0)
        return BBox(
            min(max(self.x1, 0.0), width),
            min(max(self.y1, 0.0), height),
            min(max(self.x2, 0.0), width),
            min(max(self.y2, 0.0), height),
        )

    def visible_fraction(self, frame_width: int | float, frame_height: int | float) -> float:
        if not self.is_valid(min_area=0.0) or self.area == 0.0:
            return 0.0
        return min(1.0, self.clip(frame_width, frame_height).area / self.area)

    def is_out_of_frame(self, frame_width: int | float, frame_height: int | float) -> bool:
        return self.clip(frame_width, frame_height).area <= 0.0

    def canonical_foot_point(
        self, frame_width: int | float | None = None, frame_height: int | float | None = None
    ) -> tuple[float, float]:
        """Return bottom-centre; optionally constrain it to valid pixel coordinates."""
        x, y = (self.x1 + self.x2) / 2.0, self.y2
        if frame_width is None and frame_height is None:
            return x, y
        if frame_width is None or frame_height is None:
            raise ValueError("frame_width and frame_height must be supplied together")
        width, height = _frame_size(frame_width, frame_height)
        return min(max(x, 0.0), width), min(max(y, 0.0), height)

    @property
    def foot_point(self) -> tuple[float, float]:
        return self.canonical_foot_point()

    def foot_point_confidence(self, frame_width: int | float, frame_height: int | float) -> float:
        """Estimate reliability from visibility of the box's lower edge.

        A crop whose bottom lies outside the image cannot localise the feet.
        Horizontal truncation reduces confidence in proportion to visible width.
        """
        width, height = _frame_size(frame_width, frame_height)
        if not self.is_valid(min_area=0.0) or self.y2 <= 0.0 or self.y2 > height:
            return 0.0
        visible_width = max(0.0, min(self.x2, width) - max(self.x1, 0.0))
        return min(1.0, visible_width / self.width) if self.width > 0.0 else 0.0

    def player_crop_quality(
        self,
        frame_width: int | float,
        frame_height: int | float,
        *,
        min_area: float = 64.0,
        min_visible_fraction: float = 0.5,
    ) -> CropQuality:
        clipped = self.clip(frame_width, frame_height)
        visible = self.visible_fraction(frame_width, frame_height)
        valid = clipped.is_valid(min_area) and visible >= min_visible_fraction
        return CropQuality(
            valid=valid,
            visible_fraction=visible,
            foot_point_confidence=self.foot_point_confidence(frame_width, frame_height),
            out_of_frame=clipped.area <= 0.0,
            area=clipped.area,
        )


def _frame_size(width: int | float, height: int | float) -> tuple[float, float]:
    width, height = float(width), float(height)
    if not isfinite(width) or not isfinite(height) or width <= 0.0 or height <= 0.0:
        raise ValueError("frame dimensions must be finite and positive")
    return width, height


def _bbox(value: BBox | Sequence[float]) -> BBox:
    return value if isinstance(value, BBox) else BBox.from_sequence(value)


def clip_bbox(
    bbox: BBox | Sequence[float], frame_width: int | float, frame_height: int | float
) -> BBox:
    return _bbox(bbox).clip(frame_width, frame_height)


def canonical_foot_point(bbox: BBox | Sequence[float]) -> tuple[float, float]:
    return _bbox(bbox).canonical_foot_point()


def foot_point_confidence(
    bbox: BBox | Sequence[float], frame_width: int | float, frame_height: int | float
) -> float:
    return _bbox(bbox).foot_point_confidence(frame_width, frame_height)


def player_crop_quality(
    bbox: BBox | Sequence[float],
    frame_width: int | float,
    frame_height: int | float,
    *,
    min_area: float = 64.0,
    min_visible_fraction: float = 0.5,
) -> CropQuality:
    return _bbox(bbox).player_crop_quality(
        frame_width,
        frame_height,
        min_area=min_area,
        min_visible_fraction=min_visible_fraction,
    )
