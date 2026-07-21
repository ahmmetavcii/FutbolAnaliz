from __future__ import annotations

from pathlib import Path

import pytest

from football_analytics.full_match.health import (
    MIN_ESTIMATE_BYTES,
    DiskGuardError,
    check_disk_guard,
    estimate_run_bytes,
    free_disk_bytes,
)
from football_analytics.full_match.schemas import (
    CameraRole,
    CameraSpec,
    MatchManifest,
    VideoProbe,
)


def manifest_with_input_bytes(size_bytes: int) -> MatchManifest:
    probe = VideoProbe(
        path="/videos/cam.mp4",
        duration_seconds=5400.0,
        avg_frame_rate=25.0,
        width=1920,
        height=1080,
        size_bytes=size_bytes,
    )
    return MatchManifest(
        match_id="m1",
        cameras=[
            CameraSpec(
                camera_id="cam",
                role=CameraRole.TACTICAL_FULL,
                path="/videos/cam.mp4",
                sha256="a" * 64,
                probe=probe,
            )
        ],
    )


def test_free_disk_bytes_walks_up_to_existing_parent(tmp_path: Path) -> None:
    missing = tmp_path / "does" / "not" / "exist"
    assert free_disk_bytes(missing) == free_disk_bytes(tmp_path)
    assert free_disk_bytes(tmp_path) > 0


def test_estimate_scales_with_input_size() -> None:
    small = estimate_run_bytes(manifest_with_input_bytes(1024))
    large = estimate_run_bytes(manifest_with_input_bytes(100 * 1024**3))
    assert small == MIN_ESTIMATE_BYTES
    assert large == int(100 * 1024**3 * 0.10)
    assert large > small


def test_disk_guard_passes_for_small_requirement(tmp_path: Path) -> None:
    report = check_disk_guard(tmp_path, required_bytes=1024, min_free_bytes=0)
    assert report["ok"] is True
    assert report["free_bytes"] > 0


def test_disk_guard_raises_when_space_is_insufficient(tmp_path: Path) -> None:
    absurd = free_disk_bytes(tmp_path) * 10
    with pytest.raises(DiskGuardError, match="insufficient disk space"):
        check_disk_guard(tmp_path, required_bytes=absurd)
