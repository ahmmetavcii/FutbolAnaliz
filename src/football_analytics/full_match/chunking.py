"""Chunk planning and bounded-memory chunk iteration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .schemas import (
    DEFAULT_CHUNK_SECONDS,
    MAX_CHUNK_SECONDS,
    MIN_CHUNK_SECONDS,
    ChunkRecord,
    VideoProbe,
)


class ChunkingError(ValueError):
    """Invalid chunking parameters."""


def validate_chunk_seconds(chunk_seconds: float | None) -> float:
    if chunk_seconds is None:
        return DEFAULT_CHUNK_SECONDS
    value = float(chunk_seconds)
    if not MIN_CHUNK_SECONDS <= value <= MAX_CHUNK_SECONDS:
        raise ChunkingError(
            f"chunk_seconds must be within [{MIN_CHUNK_SECONDS:g}, "
            f"{MAX_CHUNK_SECONDS:g}] seconds, got {value:g}"
        )
    return value


def plan_chunks(
    camera_id: str, probe: VideoProbe, chunk_seconds: float | None = None
) -> list[ChunkRecord]:
    """Split one camera timeline into contiguous chunk records."""
    seconds = validate_chunk_seconds(chunk_seconds)
    fps = probe.avg_frame_rate
    total_frames = probe.nb_frames or int(round(probe.duration_seconds * fps))
    if total_frames <= 0:
        raise ChunkingError(f"camera {camera_id} reports no frames")
    frames_per_chunk = max(1, int(round(seconds * fps)))
    chunk_count = math.ceil(total_frames / frames_per_chunk)

    records: list[ChunkRecord] = []
    for index in range(chunk_count):
        frame_start = index * frames_per_chunk
        frame_end = min(total_frames, frame_start + frames_per_chunk)
        records.append(
            ChunkRecord(
                camera_id=camera_id,
                chunk_index=index,
                start_seconds=frame_start / fps,
                end_seconds=frame_end / fps,
                frame_start=frame_start,
                frame_end=frame_end,
            )
        )
    return records


def iter_chunk_frames(
    path: Path, record: ChunkRecord
) -> Iterator[tuple[int, np.ndarray]]:
    """Stream (frame_index, image) pairs for one chunk; one frame in memory at a time."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        if record.frame_start > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(record.frame_start))
        frame_index = record.frame_start
        while frame_index < record.frame_end:
            ok, image = capture.read()
            if not ok or image is None:
                break
            yield frame_index, image
            frame_index += 1
    finally:
        capture.release()
