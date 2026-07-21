"""Shared types and label conventions for jersey recognition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

UNKNOWN_LABEL = -1
UNKNOWN_CLASS_INDEX = 100
NUM_CLASSES = 101
KNOWN_LABELS = tuple(range(100))


def label_to_class(label: int) -> int:
    if label == UNKNOWN_LABEL:
        return UNKNOWN_CLASS_INDEX
    if label not in KNOWN_LABELS:
        raise ValueError(f"jersey label must be -1 or 0..99, got {label}")
    return label


def class_to_label(class_index: int) -> int:
    if class_index == UNKNOWN_CLASS_INDEX:
        return UNKNOWN_LABEL
    if class_index not in KNOWN_LABELS:
        raise ValueError(f"class index must be 0..100, got {class_index}")
    return class_index


@dataclass(frozen=True, slots=True)
class TrackletRecord:
    tracklet_id: str
    label: int
    frame_paths: tuple[Path, ...]
    source_split: str


@dataclass(frozen=True, slots=True)
class JerseyPrediction:
    tracklet_id: str
    jersey_number: int
    confidence: float
    predicted_class: int
    raw_confidence: float
    num_frames: int
    frame_weights: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
