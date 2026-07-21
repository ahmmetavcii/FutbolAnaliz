from __future__ import annotations

from pathlib import Path

import pytest

from football_analytics.full_match.chunking import (
    ChunkingError,
    iter_chunk_frames,
    plan_chunks,
    validate_chunk_seconds,
)
from football_analytics.full_match.schemas import VideoProbe


def make_probe(duration: float, fps: float = 25.0) -> VideoProbe:
    return VideoProbe(
        path="/videos/cam.mp4",
        duration_seconds=duration,
        avg_frame_rate=fps,
        width=1920,
        height=1080,
        size_bytes=1000,
        nb_frames=int(round(duration * fps)),
    )


def test_default_chunk_seconds_is_120() -> None:
    assert validate_chunk_seconds(None) == 120.0


@pytest.mark.parametrize("value", [30, 120, 300])
def test_chunk_seconds_bounds_accepted(value: float) -> None:
    assert validate_chunk_seconds(value) == float(value)


@pytest.mark.parametrize("value", [29.99, 300.01, 0, -1])
def test_chunk_seconds_out_of_bounds_rejected(value: float) -> None:
    with pytest.raises(ChunkingError, match="chunk_seconds"):
        validate_chunk_seconds(value)


def test_plan_chunks_covers_full_timeline_contiguously() -> None:
    probe = make_probe(duration=605.0, fps=25.0)  # 15125 frames
    records = plan_chunks("cam", probe, 120.0)
    assert len(records) == 6  # 5 full chunks of 3000 frames + 125-frame tail

    assert records[0].frame_start == 0
    assert records[-1].frame_end == 15125
    for previous, current in zip(records, records[1:]):
        assert current.frame_start == previous.frame_end
        assert current.chunk_index == previous.chunk_index + 1
    for record in records:
        assert record.status.value == "PENDING"
        assert record.end_seconds > record.start_seconds


def test_plan_chunks_short_video_single_chunk() -> None:
    records = plan_chunks("cam", make_probe(duration=45.0), 120.0)
    assert len(records) == 1
    assert records[0].frame_end == 45 * 25


def test_iter_chunk_frames_streams_only_requested_range(tmp_path: Path, make_video) -> None:
    video = make_video(tmp_path / "v.mp4", seconds=6.0, fps=10.0)
    probe = make_probe(duration=6.0, fps=10.0)
    records = plan_chunks("cam", probe, 30.0)
    assert len(records) == 1

    # Constrain the record to a sub-range and verify only that range streams.
    record = records[0].model_copy(update={"frame_start": 20, "frame_end": 40})
    frames = list(iter_chunk_frames(video, record))
    assert [index for index, _ in frames] == list(range(20, 40))
    for _, image in frames:
        assert image.shape == (48, 64, 3)
