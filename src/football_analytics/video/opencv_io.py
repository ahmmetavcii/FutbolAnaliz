"""OpenCV video readability checks and clip preparation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2


def verify_opencv_readable(path: Path, max_frames: int = 30) -> dict[str, float | int | bool]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    read_ok = 0
    for _ in range(max_frames):
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        read_ok += 1
    capture.release()
    if read_ok < 1:
        raise RuntimeError(f"OpenCV read zero frames from {path}")
    return {
        "opened": True,
        "width": width,
        "height": height,
        "fps": fps,
        "reported_frame_count": frame_count,
        "sampled_frames_ok": read_ok,
    }


def prepare_test_clip(
    source: Path,
    target: Path,
    max_seconds: float = 20.0,
    duration_seconds: float | None = None,
) -> dict[str, float | str | bool]:
    """Copy or trim source into target. Trim only when longer than max_seconds."""
    target.parent.mkdir(parents=True, exist_ok=True)
    duration = duration_seconds
    if duration is None:
        from football_analytics.video.ffprobe import probe_video, summarize_probe

        duration = float(summarize_probe(probe_video(source))["duration_seconds"])
    if duration <= max_seconds:
        if source.resolve() != target.resolve():
            target.write_bytes(source.read_bytes())
        return {
            "trimmed": False,
            "source": str(source),
            "target": str(target),
            "duration_seconds": duration,
        }
    temporary = target.with_suffix(target.suffix + ".part.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-t",
            str(max_seconds),
            "-c",
            "copy",
            str(temporary),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    temporary.replace(target)
    return {
        "trimmed": True,
        "source": str(source),
        "target": str(target),
        "duration_seconds": max_seconds,
    }
