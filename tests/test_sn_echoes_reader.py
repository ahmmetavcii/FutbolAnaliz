from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_analytics.integrations.sn_echoes_reader import (
    DEFAULT_DATASET_ROOT,
    CommentarySegment,
    SnEchoesFormatError,
    SnEchoesReader,
    format_srt_timestamp,
    iter_raw_segments,
    segments_to_events,
    segments_to_srt,
    write_event_timeline,
    write_srt,
)


def _write_asr(path: Path, segments: dict[str, list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"segments": segments}), encoding="utf-8")


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "Dataset"
    game_a = root / "whisper_v1" / "england_epl" / "2014-2015" / "2015-05-17 - United 1 - 1 Arsenal"
    _write_asr(
        game_a / "1_asr.json",
        {
            "0": [0.0, 3.0, "Kickoff in the first half."],
            "1": [3.0, 5.5, "A long ball forward."],
            "2": [100.0, 104.0, "Great save by the keeper!"],
        },
    )
    _write_asr(
        game_a / "2_asr.json",
        {
            "0": [1.0, 4.0, "Second half underway."],
            "1": [50.0, 53.0, "Goal! What a strike."],
        },
    )
    game_b = root / "whisper_v1" / "spain_laliga" / "2015-2016" / "2016-01-01 - Betis 0 - 2 Sevilla"
    _write_asr(game_b / "1_asr.json", {"0": [0.0, 2.0, "Derby day in Seville."]})
    game_c = root / "whisper_v2_en" / "england_epl" / "2014-2015" / "2015-05-17 - United 1 - 1 Arsenal"
    _write_asr(game_c / "1_asr.json", {"0": [0.0, 3.0, "Kickoff, translated."]})
    return root


# ---------------------------------------------------------------------------
# Streaming parser
# ---------------------------------------------------------------------------


def test_iter_raw_segments_parses_triples(dataset_root: Path) -> None:
    path = (
        dataset_root
        / "whisper_v1"
        / "england_epl"
        / "2014-2015"
        / "2015-05-17 - United 1 - 1 Arsenal"
        / "1_asr.json"
    )
    rows = list(iter_raw_segments(path))
    assert rows == [
        (0, 0.0, 3.0, "Kickoff in the first half."),
        (1, 3.0, 5.5, "A long ball forward."),
        (2, 100.0, 104.0, "Great save by the keeper!"),
    ]


def test_iter_raw_segments_streams_across_chunk_boundaries(tmp_path: Path) -> None:
    # Text far larger than the internal read chunk forces buffer refills.
    long_text = "commentary " * 20_000
    path = tmp_path / "1_asr.json"
    _write_asr(path, {"0": [0.0, 1.0, long_text], "1": [1.0, 2.0, "short"]})
    rows = list(iter_raw_segments(path))
    assert len(rows) == 2
    assert rows[0][3] == long_text
    assert rows[1] == (1, 1.0, 2.0, "short")


def test_iter_raw_segments_empty_mapping(tmp_path: Path) -> None:
    path = tmp_path / "1_asr.json"
    _write_asr(path, {})
    assert list(iter_raw_segments(path)) == []


@pytest.mark.parametrize(
    "payload",
    [
        {"not_segments": {}},
        {"segments": {"0": [0.0, 1.0]}},
        {"segments": {"0": [0.0, "x", "text"]}},
        {"segments": {"": [0.0, 1.0, "text"]}},
    ],
)
def test_iter_raw_segments_rejects_malformed(tmp_path: Path, payload: dict) -> None:
    path = tmp_path / "1_asr.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SnEchoesFormatError):
        list(iter_raw_segments(path))


def test_iter_raw_segments_rejects_truncated_json(tmp_path: Path) -> None:
    path = tmp_path / "1_asr.json"
    path.write_text('{"segments": {"0": [0.0, 1.0, "text"', encoding="utf-8")
    with pytest.raises(SnEchoesFormatError):
        list(iter_raw_segments(path))


# ---------------------------------------------------------------------------
# Reader traversal and filtering
# ---------------------------------------------------------------------------


def test_versions(dataset_root: Path) -> None:
    reader = SnEchoesReader(dataset_root)
    assert reader.versions() == ["whisper_v1", "whisper_v2_en"]


def test_missing_root_raises() -> None:
    with pytest.raises(FileNotFoundError):
        SnEchoesReader("/nonexistent/sn-echoes-root")


def test_iter_matches_filters(dataset_root: Path) -> None:
    reader = SnEchoesReader(dataset_root)
    all_matches = list(reader.iter_matches())
    assert len(all_matches) == 3

    v1 = list(reader.iter_matches(version="whisper_v1"))
    assert {m.league for m in v1} == {"england_epl", "spain_laliga"}

    arsenal = list(reader.iter_matches(game="arsenal"))
    assert len(arsenal) == 2
    assert {m.version for m in arsenal} == {"whisper_v1", "whisper_v2_en"}

    none = list(reader.iter_matches(version="whisper_v3"))
    assert none == []


def test_match_id(dataset_root: Path) -> None:
    reader = SnEchoesReader(dataset_root)
    match = next(reader.iter_matches(version="whisper_v1", league="spain_laliga"))
    assert match.match_id == "spain_laliga/2015-2016/2016-01-01 - Betis 0 - 2 Sevilla"


def test_iter_segments_version_and_half_filter(dataset_root: Path) -> None:
    reader = SnEchoesReader(dataset_root)
    segments = list(reader.iter_segments(version="whisper_v1", league="england_epl", half=2))
    assert [s.text for s in segments] == ["Second half underway.", "Goal! What a strike."]
    assert all(s.half == 2 for s in segments)
    assert all(s.version == "whisper_v1" for s in segments)


def test_iter_segments_timestamp_range_overlap(dataset_root: Path) -> None:
    reader = SnEchoesReader(dataset_root)
    # Range [2, 4) overlaps segments 0 and 1 of half 1, but not segment 2.
    segments = list(
        reader.iter_segments(
            version="whisper_v1", league="england_epl", half=1, start_s=2.0, end_s=4.0
        )
    )
    assert [s.segment_id for s in segments] == [0, 1]

    # Boundary: a segment ending exactly at start_s is excluded.
    segments = list(
        reader.iter_segments(version="whisper_v1", league="england_epl", half=1, start_s=3.0)
    )
    assert [s.segment_id for s in segments] == [1, 2]

    # Boundary: a segment starting exactly at end_s is excluded.
    segments = list(
        reader.iter_segments(version="whisper_v1", league="england_epl", half=1, end_s=100.0)
    )
    assert [s.segment_id for s in segments] == [0, 1]


def test_segment_ms_properties(dataset_root: Path) -> None:
    reader = SnEchoesReader(dataset_root)
    segment = next(reader.iter_segments(version="whisper_v1", league="england_epl", half=1))
    assert segment.start_ms == 0.0
    assert segment.end_ms == 3000.0


# ---------------------------------------------------------------------------
# SRT export
# ---------------------------------------------------------------------------


def test_format_srt_timestamp() -> None:
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(3.5) == "00:00:03,500"
    assert format_srt_timestamp(3661.042) == "01:01:01,042"
    assert format_srt_timestamp(-1.0) == "00:00:00,000"


def test_segments_to_srt(dataset_root: Path) -> None:
    reader = SnEchoesReader(dataset_root)
    segments = reader.iter_segments(version="whisper_v1", league="england_epl", half=1)
    srt = segments_to_srt(segments)
    blocks = srt.strip().split("\n\n")
    assert len(blocks) == 3
    assert blocks[0] == "1\n00:00:00,000 --> 00:00:03,000\nKickoff in the first half."
    assert blocks[2].startswith("3\n00:01:40,000 --> 00:01:44,000")


def test_write_srt(dataset_root: Path, tmp_path: Path) -> None:
    reader = SnEchoesReader(dataset_root)
    out = write_srt(
        reader.iter_segments(version="whisper_v1", league="spain_laliga"),
        tmp_path / "out.srt",
    )
    content = out.read_text(encoding="utf-8")
    assert "Derby day in Seville." in content
    assert content.startswith("1\n00:00:00,000 --> 00:00:02,000")


# ---------------------------------------------------------------------------
# Canonical event timeline export
# ---------------------------------------------------------------------------


def test_segments_to_events_canonical_fields_and_order(dataset_root: Path) -> None:
    reader = SnEchoesReader(dataset_root)
    segments = list(reader.iter_segments(version="whisper_v1", league="england_epl"))
    # Shuffle across halves to confirm the export sorts.
    events = segments_to_events(reversed(segments))
    assert [(e["half"], e["timestamp_ms"]) for e in events] == [
        (1, 0.0),
        (1, 3000.0),
        (1, 100000.0),
        (2, 1000.0),
        (2, 50000.0),
    ]
    first = events[0]
    assert first["event_type"] == "commentary"
    assert first["match_id"] == "england_epl/2014-2015/2015-05-17 - United 1 - 1 Arsenal"
    assert first["source_method"] == "sn_echoes/whisper_v1"
    assert first["duration_ms"] == 3000.0
    assert first["valid"] is True
    assert first["schema_version"] == "1.0.0"


def test_segments_to_events_flags_invalid_span() -> None:
    segment = CommentarySegment(
        version="whisper_v1",
        league="l",
        season="s",
        game="g",
        half=1,
        segment_id=0,
        start_s=5.0,
        end_s=4.0,
        text="broken",
    )
    events = segments_to_events([segment])
    assert events[0]["valid"] is False


def test_write_event_timeline_jsonl(dataset_root: Path, tmp_path: Path) -> None:
    reader = SnEchoesReader(dataset_root)
    out = write_event_timeline(
        reader.iter_segments(version="whisper_v1", league="england_epl", half=2),
        tmp_path / "events.jsonl",
    )
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[1]["text"] == "Goal! What a strike."
    assert records[1]["timestamp_ms"] == 50000.0


# ---------------------------------------------------------------------------
# Optional smoke test against the real local dataset
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DEFAULT_DATASET_ROOT.is_dir(), reason="local sn-echoes dataset not available"
)
def test_real_dataset_smoke() -> None:
    reader = SnEchoesReader(DEFAULT_DATASET_ROOT)
    versions = reader.versions()
    assert versions, "expected at least one whisper version"
    match = next(reader.iter_matches(version=versions[0]))
    segments = []
    for segment in reader.iter_match_segments(match):
        segments.append(segment)
        if len(segments) >= 5:
            break
    assert segments, f"no segments found for {match.match_id}"
    for segment in segments:
        assert segment.end_s >= segment.start_s >= 0.0
        assert isinstance(segment.text, str)
