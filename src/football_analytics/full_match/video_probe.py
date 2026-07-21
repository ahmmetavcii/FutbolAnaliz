"""Camera input probing: ffprobe metadata plus OpenCV decode spot checks."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2

from football_analytics.video.ffprobe import probe_video, summarize_probe

from .schemas import FrameCheck, VideoProbe


class ProbeError(RuntimeError):
    """The input cannot be probed or is not a decodable video."""


def _check_frame(capture: cv2.VideoCapture, frame_index: int, position: str) -> FrameCheck:
    capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, frame = capture.read()
    mean_intensity = float(frame.mean()) if ok and frame is not None else None
    return FrameCheck(
        position=position,
        frame_index=frame_index,
        ok=bool(ok and frame is not None),
        mean_intensity=mean_intensity,
    )


def opencv_frame_checks(path: Path, total_frames: int) -> list[FrameCheck]:
    """Decode first, middle, and last frames without loading the full video."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ProbeError(f"OpenCV could not open {path}")
    try:
        if total_frames <= 0:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            raise ProbeError(f"video reports no frames: {path}")
        indices = {
            "first": 0,
            "middle": max(0, total_frames // 2),
            # Some containers overreport the frame count; step back slightly.
            "last": max(0, total_frames - 2),
        }
        return [
            _check_frame(capture, frame_index, position)
            for position, frame_index in indices.items()
        ]
    finally:
        capture.release()


def probe_camera(path: Path, frame_checks: bool = True) -> VideoProbe:
    """Probe a camera video and verify it is decodable end to end."""
    path = Path(path)
    if not path.is_file():
        raise ProbeError(f"camera input does not exist: {path}")
    try:
        summary = summarize_probe(probe_video(path))
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        raise ProbeError(f"ffprobe failed for {path}: {exc}") from exc

    duration = float(summary.get("duration_seconds") or 0.0)
    fps = float(summary.get("avg_frame_rate") or 0.0)
    if duration <= 0 or fps <= 0:
        raise ProbeError(
            f"invalid metadata for {path}: duration={duration}, fps={fps}"
        )

    nb_frames = summary.get("nb_frames")
    total_frames = int(nb_frames) if nb_frames else int(round(duration * fps))
    checks = opencv_frame_checks(path, total_frames) if frame_checks else []

    return VideoProbe(
        path=str(path),
        duration_seconds=duration,
        avg_frame_rate=fps,
        width=int(summary["width"]),
        height=int(summary["height"]),
        codec_name=summary.get("codec_name"),
        nb_frames=int(nb_frames) if nb_frames else None,
        size_bytes=int(summary.get("size_bytes") or 0),
        bit_rate=int(summary.get("bit_rate") or 0),
        frame_checks=checks,
    )
