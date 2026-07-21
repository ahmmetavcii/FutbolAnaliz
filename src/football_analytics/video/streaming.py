"""Bounded-memory video iteration with explicit chunk boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VideoFrame:
    frame_id: int
    timestamp_ms: float
    chunk_id: int
    chunk_start: bool
    image: np.ndarray


class StreamingVideoReader:
    """Yield one frame at a time; ownership of the frame ends at next iteration."""

    def __init__(self, path: Path, chunk_seconds: float = 300.0) -> None:
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")
        self.path = Path(path)
        self.chunk_seconds = float(chunk_seconds)

    def __iter__(self) -> Iterator[VideoFrame]:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            capture.release()
            raise RuntimeError(f"Video reports invalid FPS: {self.path}")
        chunk_frames = max(1, int(round(self.chunk_seconds * fps)))
        frame_id = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                yield VideoFrame(
                    frame_id=frame_id,
                    timestamp_ms=frame_id * 1000.0 / fps,
                    chunk_id=frame_id // chunk_frames,
                    chunk_start=(frame_id % chunk_frames == 0),
                    image=image,
                )
                frame_id += 1
        finally:
            capture.release()
