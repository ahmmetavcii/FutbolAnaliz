"""Clean-room reader for the local SoccerNet-Echoes commentary dataset.

The dataset layout on disk is::

    <root>/<version>/<league>/<season>/<game>/<half>_asr.json

where each JSON file holds a single top-level ``segments`` mapping::

    {"segments": {"0": [start_s, end_s, "text"], "1": [...], ...}}

This module only reads local files; it does not vendor or copy any upstream
source code. Parsing streams segments incrementally so arbitrarily large
files never need to be fully materialised in memory.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

SCHEMA_VERSION = "1.0.0"

DEFAULT_DATASET_ROOT = Path("/home/ahmet/projects/soccernet/sn-echoes/Dataset")

_ASR_FILE_RE = re.compile(r"^(?P<half>\d+)_asr\.json$")
_WHITESPACE_RE = re.compile(r"[ \t\n\r]*")
_READ_CHUNK_SIZE = 64 * 1024


class SnEchoesFormatError(ValueError):
    """Raised when a JSON file does not match the expected sn-echoes shape."""


@dataclass(frozen=True)
class MatchRef:
    """A single game directory within one transcription version."""

    version: str
    league: str
    season: str
    game: str

    @property
    def match_id(self) -> str:
        return f"{self.league}/{self.season}/{self.game}"


@dataclass(frozen=True)
class CommentarySegment:
    """One timed commentary line from an ``<half>_asr.json`` file."""

    version: str
    league: str
    season: str
    game: str
    half: int
    segment_id: int | str
    start_s: float
    end_s: float
    text: str

    @property
    def match_id(self) -> str:
        return f"{self.league}/{self.season}/{self.game}"

    @property
    def start_ms(self) -> float:
        return self.start_s * 1000.0

    @property
    def end_ms(self) -> float:
        return self.end_s * 1000.0


# ---------------------------------------------------------------------------
# Streaming JSON parsing
# ---------------------------------------------------------------------------


class _ChunkBuffer:
    """Incrementally buffered view over a text stream for raw_decode parsing."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self.data = ""
        self.pos = 0
        self.eof = False

    def _fill(self) -> bool:
        chunk = self._stream.read(_READ_CHUNK_SIZE)
        if not chunk:
            self.eof = True
            return False
        # Drop consumed prefix to keep the buffer bounded.
        if self.pos > 0:
            self.data = self.data[self.pos :]
            self.pos = 0
        self.data += chunk
        return True

    def skip_whitespace(self) -> None:
        while True:
            match = _WHITESPACE_RE.match(self.data, self.pos)
            self.pos = match.end()
            if self.pos < len(self.data) or self.eof:
                return
            if not self._fill():
                return

    def peek(self) -> str:
        """Return the next non-consumed character, or '' at EOF."""
        self.skip_whitespace()
        if self.pos < len(self.data):
            return self.data[self.pos]
        return ""

    def expect(self, char: str, context: str) -> None:
        got = self.peek()
        if got != char:
            raise SnEchoesFormatError(f"expected {char!r} {context}, found {got!r}")
        self.pos += 1

    def decode_value(self, decoder: json.JSONDecoder, context: str) -> Any:
        """Decode one JSON value at the current position, reading more as needed."""
        self.skip_whitespace()
        while True:
            try:
                value, end = decoder.raw_decode(self.data, self.pos)
            except ValueError:
                # Possibly an incomplete value split across chunks.
                if self._fill():
                    continue
                snippet = self.data[self.pos : self.pos + 40]
                raise SnEchoesFormatError(f"invalid JSON {context}: {snippet!r}") from None
            # A value ending exactly at the buffer edge may be a truncated
            # number/literal; read ahead once to be sure it is complete.
            if end == len(self.data) and not self.eof and self._fill():
                continue
            self.pos = end
            return value


def iter_raw_segments(path: Path | str) -> Iterator[tuple[int | str, float, float, str]]:
    """Stream ``(segment_id, start_s, end_s, text)`` tuples from one ASR file.

    Parses the top-level ``{"segments": {...}}`` mapping incrementally so the
    whole document is never held in memory at once.
    """
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as stream:
        buf = _ChunkBuffer(stream)
        buf.expect("{", "at start of document")
        key = buf.decode_value(decoder, "reading top-level key")
        if key != "segments":
            raise SnEchoesFormatError(f"expected top-level 'segments' key, found {key!r}")
        buf.expect(":", "after 'segments' key")
        buf.expect("{", "at start of segments mapping")
        first = True
        while True:
            char = buf.peek()
            if char == "}":
                buf.pos += 1
                break
            if not first:
                buf.expect(",", "between segment entries")
            first = False
            seg_key = buf.decode_value(decoder, "reading segment key")
            buf.expect(":", "after segment key")
            value = buf.decode_value(decoder, "reading segment value")
            yield _normalise_segment(seg_key, value)


def _normalise_segment(key: Any, value: Any) -> tuple[int | str, float, float, str]:
    try:
        segment_id = int(key)
    except (TypeError, ValueError):
        # English variants contain a small number of translated identifier
        # keys (for example "eleven" and "8th"). Preserve these real upstream
        # IDs instead of rejecting otherwise valid commentary files.
        if not isinstance(key, str) or not key:
            raise SnEchoesFormatError(f"invalid segment key: {key!r}") from None
        segment_id = key
    if not isinstance(value, list) or len(value) != 3:
        raise SnEchoesFormatError(f"segment {key!r} is not a [start, end, text] triple")
    start, end, text = value
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        raise SnEchoesFormatError(f"segment {key!r} has non-numeric timestamps")
    if not isinstance(text, str):
        raise SnEchoesFormatError(f"segment {key!r} has non-string text")
    return segment_id, float(start), float(end), text


# ---------------------------------------------------------------------------
# Dataset traversal
# ---------------------------------------------------------------------------


class SnEchoesReader:
    """Read-only access to a local SoccerNet-Echoes ``Dataset`` directory."""

    def __init__(self, dataset_root: Path | str = DEFAULT_DATASET_ROOT) -> None:
        self.root = Path(dataset_root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"sn-echoes dataset root not found: {self.root}")

    def versions(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def iter_matches(
        self,
        version: str | None = None,
        league: str | None = None,
        season: str | None = None,
        game: str | None = None,
    ) -> Iterator[MatchRef]:
        """Yield matches, optionally filtered.

        ``version``, ``league`` and ``season`` are exact directory names;
        ``game`` is a case-insensitive substring of the game directory name.
        """
        for version_dir in self._select_dirs(self.root, version):
            for league_dir in self._select_dirs(version_dir, league):
                for season_dir in self._select_dirs(league_dir, season):
                    for game_dir in sorted(p for p in season_dir.iterdir() if p.is_dir()):
                        if game is not None and game.lower() not in game_dir.name.lower():
                            continue
                        yield MatchRef(
                            version=version_dir.name,
                            league=league_dir.name,
                            season=season_dir.name,
                            game=game_dir.name,
                        )

    def iter_segments(
        self,
        version: str | None = None,
        league: str | None = None,
        season: str | None = None,
        game: str | None = None,
        half: int | None = None,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> Iterator[CommentarySegment]:
        """Stream segments matching the filters.

        ``start_s``/``end_s`` select segments whose time span overlaps the
        half-open range ``[start_s, end_s)``.
        """
        for match in self.iter_matches(version=version, league=league, season=season, game=game):
            yield from self.iter_match_segments(match, half=half, start_s=start_s, end_s=end_s)

    def iter_match_segments(
        self,
        match: MatchRef,
        half: int | None = None,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> Iterator[CommentarySegment]:
        game_dir = self.root / match.version / match.league / match.season / match.game
        for file_half, asr_path in self._asr_files(game_dir):
            if half is not None and file_half != half:
                continue
            for segment_id, seg_start, seg_end, text in iter_raw_segments(asr_path):
                if start_s is not None and seg_end <= start_s:
                    continue
                if end_s is not None and seg_start >= end_s:
                    continue
                yield CommentarySegment(
                    version=match.version,
                    league=match.league,
                    season=match.season,
                    game=match.game,
                    half=file_half,
                    segment_id=segment_id,
                    start_s=seg_start,
                    end_s=seg_end,
                    text=text,
                )

    @staticmethod
    def _select_dirs(parent: Path, name: str | None) -> list[Path]:
        if name is not None:
            candidate = parent / name
            return [candidate] if candidate.is_dir() else []
        return sorted(p for p in parent.iterdir() if p.is_dir())

    @staticmethod
    def _asr_files(game_dir: Path) -> list[tuple[int, Path]]:
        files = []
        if not game_dir.is_dir():
            return files
        for path in game_dir.iterdir():
            match = _ASR_FILE_RE.match(path.name)
            if match:
                files.append((int(match.group("half")), path))
        return sorted(files)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT ``HH:MM:SS,mmm`` timestamp."""
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    hours, rem = divmod(total_s, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def segments_to_srt(segments: Iterable[CommentarySegment]) -> str:
    """Render segments as an SRT subtitle document (one cue per segment)."""
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = format_srt_timestamp(segment.start_s)
        end = format_srt_timestamp(segment.end_s)
        blocks.append(f"{index}\n{start} --> {end}\n{segment.text.strip()}\n")
    return "\n".join(blocks)


def write_srt(segments: Iterable[CommentarySegment], path: Path | str) -> Path:
    path = Path(path)
    path.write_text(segments_to_srt(segments), encoding="utf-8")
    return path


def segments_to_events(segments: Iterable[CommentarySegment]) -> list[dict[str, Any]]:
    """Convert segments to canonical timeline event records.

    Field names follow the project's canonical column conventions
    (``schema_version``, ``match_id``, ``timestamp_ms``, ``source_method``).
    Events are sorted by half then start time.
    """
    events = [
        {
            "schema_version": SCHEMA_VERSION,
            "match_id": segment.match_id,
            "half": segment.half,
            "event_type": "commentary",
            "timestamp_ms": segment.start_ms,
            "end_timestamp_ms": segment.end_ms,
            "duration_ms": segment.end_ms - segment.start_ms,
            "text": segment.text.strip(),
            "source_method": f"sn_echoes/{segment.version}",
            "valid": segment.end_s >= segment.start_s,
        }
        for segment in segments
    ]
    events.sort(key=lambda event: (event["half"], event["timestamp_ms"]))
    return events


def write_event_timeline(segments: Iterable[CommentarySegment], path: Path | str) -> Path:
    """Write canonical events as JSON Lines."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as stream:
        for event in segments_to_events(segments):
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path
