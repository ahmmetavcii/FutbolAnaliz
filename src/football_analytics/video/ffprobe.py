"""FFprobe metadata helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def probe_video(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return json.loads(completed.stdout)


def summarize_probe(probe: dict[str, Any]) -> dict[str, Any]:
    video_stream = next(
        (stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        video_stream = next(
            (
                stream
                for stream in probe.get("streams", [])
                if stream.get("codec_name") in {"h264", "hevc", "mpeg4", "vp9", "av1"}
            ),
            {},
        )
    format_info = probe.get("format", {})
    fps_raw = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    fps = 0.0
    if isinstance(fps_raw, str) and "/" in fps_raw:
        numerator, denominator = fps_raw.split("/", 1)
        denom = float(denominator)
        fps = float(numerator) / denom if denom else 0.0
    return {
        "duration_seconds": float(format_info.get("duration") or 0.0),
        "size_bytes": int(format_info.get("size") or 0),
        "bit_rate": int(format_info.get("bit_rate") or 0),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "codec_name": video_stream.get("codec_name"),
        "avg_frame_rate": fps,
        "nb_frames": int(video_stream.get("nb_frames") or 0)
        if str(video_stream.get("nb_frames", "")).isdigit()
        else None,
    }
