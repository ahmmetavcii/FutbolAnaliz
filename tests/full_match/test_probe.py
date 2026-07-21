from __future__ import annotations

from pathlib import Path

import pytest

from football_analytics.full_match.video_probe import ProbeError, probe_camera


def test_probe_camera_returns_metadata_and_frame_checks(test_video: Path) -> None:
    probe = probe_camera(test_video)
    assert probe.duration_seconds == pytest.approx(6.0, abs=0.5)
    assert probe.avg_frame_rate == pytest.approx(10.0, abs=0.1)
    assert probe.width == 64
    assert probe.height == 48
    assert probe.size_bytes > 0

    positions = {check.position for check in probe.frame_checks}
    assert positions == {"first", "middle", "last"}
    assert all(check.ok for check in probe.frame_checks)
    assert probe.decodable


def test_probe_camera_can_skip_frame_checks(test_video: Path) -> None:
    probe = probe_camera(test_video, frame_checks=False)
    assert probe.frame_checks == []
    assert not probe.decodable


def test_probe_camera_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ProbeError, match="does not exist"):
        probe_camera(tmp_path / "nope.mp4")


def test_probe_camera_rejects_non_video(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"this is not a video at all")
    with pytest.raises(ProbeError):
        probe_camera(bogus)
