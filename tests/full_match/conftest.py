from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def make_test_video(
    path: Path,
    seconds: float = 6.0,
    fps: float = 10.0,
    size: tuple[int, int] = (64, 48),
) -> Path:
    """Write a tiny synthetic mp4 whose frames encode their index as intensity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    assert writer.isOpened(), f"OpenCV could not create {path}"
    total = int(round(seconds * fps))
    for index in range(total):
        frame = np.full((size[1], size[0], 3), (index * 7) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert path.stat().st_size > 0
    return path


@pytest.fixture()
def test_video(tmp_path: Path) -> Path:
    return make_test_video(tmp_path / "camera.mp4")


@pytest.fixture()
def make_video():
    return make_test_video


@pytest.fixture()
def base_config() -> dict:
    return {
        "full_match": {
            "schema_version": "1.0.0",
            "profile": "single_camera",
            "chunk_seconds": 30,
            "resume": True,
            "fail_fast": True,
        },
        "cameras": {"expected_count": 1, "processing": "sequential"},
    }
