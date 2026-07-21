"""Streaming video exporters: annotated main video and multi-camera review grid.

All exporters process one frame per iteration; full-video frame buffers are
never held in memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from football_analytics.video.ffprobe import probe_video, summarize_probe
from football_analytics.video.opencv_io import verify_opencv_readable
from football_analytics.video.streaming import StreamingVideoReader

FrameAnnotator = Callable[[np.ndarray, int, float], np.ndarray | None]


def _format_timestamp(timestamp_ms: float) -> str:
    total_seconds = max(0.0, timestamp_ms) / 1000.0
    minutes, seconds = divmod(int(total_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    millis = int(round((total_seconds - int(total_seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _draw_label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    scale = max(0.5, min(image.shape[1], image.shape[0]) / 1200.0)
    thickness = max(1, int(round(scale * 2)))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x, y = origin
    cv2.rectangle(
        image,
        (x - 4, y - text_height - baseline - 4),
        (x + text_width + 4, y + baseline),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {path}")
    return writer


def export_annotated_video(
    input_video: Path,
    output_video: Path,
    annotate: FrameAnnotator | None = None,
    *,
    overlay_timestamp: bool = True,
) -> dict[str, Any]:
    """Stream input frames, apply the annotation callback, write the main video.

    ``annotate`` receives (image, frame_id, timestamp_ms) and may draw in place
    or return a replacement image of identical size.
    """
    input_video = Path(input_video)
    output_video = Path(output_video)
    reader = StreamingVideoReader(input_video)
    writer: cv2.VideoWriter | None = None
    written = 0
    fps = 0.0
    size = (0, 0)
    try:
        for frame in reader:
            image = frame.image
            if annotate is not None:
                replacement = annotate(image, frame.frame_id, frame.timestamp_ms)
                if replacement is not None:
                    image = replacement
            if overlay_timestamp:
                _draw_label(image, _format_timestamp(frame.timestamp_ms), (12, 28))
            if writer is None:
                size = (image.shape[1], image.shape[0])
                fps = summarize_probe(probe_video(input_video))["avg_frame_rate"] or 25.0
                writer = _open_writer(output_video, fps, size)
            writer.write(image)
            written += 1
    finally:
        if writer is not None:
            writer.release()
    if written == 0:
        raise RuntimeError(f"Annotated export wrote zero frames from {input_video}")
    validation = validate_video_export(output_video, expected_frames=written)
    return {
        "path": str(output_video),
        "frames": written,
        "fps": fps,
        "width": size[0],
        "height": size[1],
        "validation": validation,
    }


class _SyncedCamera:
    """Advance one camera stream to a target timestamp, holding a single frame."""

    def __init__(self, path: Path, label: str) -> None:
        self.label = label
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera video: {path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.fps <= 0:
            self.capture.release()
            raise RuntimeError(f"Camera video reports invalid FPS: {path}")
        self.next_frame_id = 0
        self.current: np.ndarray | None = None
        self.finished = False

    def frame_at(self, timestamp_ms: float) -> np.ndarray | None:
        """Return the latest frame whose native timestamp <= timestamp_ms."""
        while (
            not self.finished
            and (self.next_frame_id * 1000.0 / self.fps) <= timestamp_ms + 1e-6
        ):
            ok, image = self.capture.read()
            if not ok:
                self.finished = True
                self.current = None
                break
            self.current = image
            self.next_frame_id += 1
        return self.current

    def release(self) -> None:
        self.capture.release()


def export_review_grid_video(
    camera_videos: Sequence[Path],
    output_video: Path,
    *,
    camera_labels: Sequence[str] | None = None,
    tile_size: tuple[int, int] = (640, 360),
    fps: float | None = None,
) -> dict[str, Any]:
    """Compose a synchronized 2- or 4-camera review grid.

    Cameras are aligned by native timestamp against a shared output clock, so
    streams with differing frame rates stay in sync; every tile carries the
    shared timestamp and its camera label. A camera that ends early shows a
    'NO SIGNAL' tile until all cameras are exhausted.
    """
    camera_videos = [Path(video) for video in camera_videos]
    output_video = Path(output_video)
    if len(camera_videos) not in (2, 4):
        raise ValueError(f"Review grid supports 2 or 4 cameras, got {len(camera_videos)}")
    labels = list(camera_labels or [f"CAM {index + 1}" for index in range(len(camera_videos))])
    if len(labels) != len(camera_videos):
        raise ValueError("camera_labels length must match camera_videos")

    cameras = [_SyncedCamera(path, label) for path, label in zip(camera_videos, labels)]
    grid_columns = 2
    grid_rows = 1 if len(cameras) == 2 else 2
    tile_width, tile_height = tile_size
    canvas_size = (grid_columns * tile_width, grid_rows * tile_height)

    output_fps = float(fps or max(camera.fps for camera in cameras))
    writer = _open_writer(output_video, output_fps, canvas_size)
    written = 0
    try:
        output_frame_id = 0
        while True:
            timestamp_ms = output_frame_id * 1000.0 / output_fps
            tiles: list[np.ndarray] = []
            live = 0
            for camera in cameras:
                image = camera.frame_at(timestamp_ms)
                if image is None:
                    tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
                    _draw_label(tile, f"{camera.label} - NO SIGNAL", (12, tile_height // 2))
                else:
                    live += 1
                    tile = cv2.resize(image, (tile_width, tile_height))
                    _draw_label(tile, camera.label, (12, 28))
                _draw_label(tile, _format_timestamp(timestamp_ms), (12, tile_height - 14))
                tiles.append(tile)
            if live == 0:
                break
            rows = [
                np.hstack(tiles[row * grid_columns : (row + 1) * grid_columns])
                for row in range(grid_rows)
            ]
            writer.write(np.vstack(rows))
            written += 1
            output_frame_id += 1
    finally:
        writer.release()
        for camera in cameras:
            camera.release()
    if written == 0:
        raise RuntimeError("Review grid export wrote zero frames")
    validation = validate_video_export(output_video, expected_frames=written)
    return {
        "path": str(output_video),
        "frames": written,
        "fps": output_fps,
        "width": canvas_size[0],
        "height": canvas_size[1],
        "cameras": labels,
        "validation": validation,
    }


def validate_video_export(
    path: Path,
    *,
    expected_frames: int | None = None,
    expected_size: tuple[int, int] | None = None,
    frame_tolerance: int = 2,
) -> dict[str, Any]:
    """Validate an exported video with ffprobe metadata plus OpenCV decoding."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Video export missing or empty: {path}")
    probe_summary = summarize_probe(probe_video(path))
    opencv_summary = verify_opencv_readable(path)
    width = int(probe_summary["width"])
    height = int(probe_summary["height"])
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Video export has invalid dimensions: {path}")
    if expected_size is not None and (width, height) != tuple(expected_size):
        raise RuntimeError(
            f"Video export size mismatch for {path}: "
            f"expected {expected_size}, found {(width, height)}"
        )
    reported = probe_summary["nb_frames"] or opencv_summary["reported_frame_count"]
    if expected_frames is not None and reported:
        if abs(int(reported) - int(expected_frames)) > frame_tolerance:
            raise RuntimeError(
                f"Video export frame-count mismatch for {path}: "
                f"expected ~{expected_frames}, found {reported}"
            )
    return {
        "path": str(path),
        "ffprobe": probe_summary,
        "opencv": opencv_summary,
        "validated": True,
    }
