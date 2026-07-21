"""Review-clip windows for detected events.

Each event gets a padded time window so a reviewer sees context before and
after the moment. Substitution events use their full interval. Overlapping
windows of the same event type can be merged to avoid duplicate review work.
Replay segments are subtracted because reviewing a replay of a moment instead
of the live footage leads to double counting.

:func:`export_clip_mp4` can optionally materialize a window as a short MP4 by
streaming frames through OpenCV; ``cv2`` is an optional dependency and the
rest of this module works without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from football_analytics.events.schemas import EventType, MatchEvent
from football_analytics.events.substitution_detector import interval_of

try:
    import cv2
except ImportError:  # pragma: no cover - environment-dependent
    cv2 = None


def opencv_available() -> bool:
    return cv2 is not None


@dataclass(frozen=True)
class ClipWindow:
    event_id: str
    event_type: EventType
    start_ms: float
    end_ms: float

    def __post_init__(self) -> None:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def overlaps(self, other: "ClipWindow") -> bool:
        return self.start_ms <= other.end_ms and other.start_ms <= self.end_ms


@dataclass(frozen=True)
class ClipConfig:
    pre_ms: float = 8_000.0
    post_ms: float = 5_000.0
    #: Windows of the same event type closer than this are merged.
    merge_gap_ms: float = 1_000.0

    def __post_init__(self) -> None:
        if self.pre_ms < 0.0 or self.post_ms < 0.0 or self.merge_gap_ms < 0.0:
            raise ValueError("clip paddings must be non-negative")


def clip_for_event(event: MatchEvent, config: ClipConfig | None = None) -> ClipWindow:
    cfg = config or ClipConfig()
    interval = interval_of(event) if event.event_type is EventType.SUBSTITUTION else None
    if interval is not None:
        start = interval.start_ms - cfg.pre_ms
        end = interval.end_ms + cfg.post_ms
    else:
        start = event.timestamp_ms - cfg.pre_ms
        end = event.timestamp_ms + cfg.post_ms
    return ClipWindow(event.event_id, event.event_type, max(0.0, start), end)


def build_clips(
    events: Sequence[MatchEvent],
    *,
    config: ClipConfig | None = None,
    replay_segments: Sequence[tuple[float, float]] = (),
    merge_overlapping: bool = True,
) -> list[ClipWindow]:
    """Build review clips for events, excluding replay footage.

    ``replay_segments`` are (start_ms, end_ms) spans of broadcast replay;
    clip windows are trimmed so they never cover replay footage. A window
    fully swallowed by a replay is dropped (there is nothing live to review).
    """
    cfg = config or ClipConfig()
    windows = [clip_for_event(event, cfg) for event in events]
    trimmed: list[ClipWindow] = []
    for window in windows:
        for piece in _subtract_segments(window, replay_segments):
            trimmed.append(piece)
    trimmed.sort(key=lambda w: (w.event_type.value, w.start_ms))
    if not merge_overlapping:
        return trimmed
    return _merge(trimmed, cfg.merge_gap_ms)


def _subtract_segments(
    window: ClipWindow, segments: Sequence[tuple[float, float]]
) -> list[ClipWindow]:
    pieces: list[tuple[float, float]] = [(window.start_ms, window.end_ms)]
    for seg_start, seg_end in segments:
        if seg_end < seg_start:
            raise ValueError("replay segment end must be >= start")
        next_pieces: list[tuple[float, float]] = []
        for start, end in pieces:
            if seg_end <= start or seg_start >= end:
                next_pieces.append((start, end))
                continue
            if start < seg_start:
                next_pieces.append((start, seg_start))
            if seg_end < end:
                next_pieces.append((seg_end, end))
        pieces = next_pieces
    return [
        ClipWindow(window.event_id, window.event_type, start, end)
        for start, end in pieces
        if end > start
    ]


def _merge(windows: list[ClipWindow], gap_ms: float) -> list[ClipWindow]:
    merged: list[ClipWindow] = []
    for window in windows:
        if (
            merged
            and merged[-1].event_type is window.event_type
            and window.start_ms - merged[-1].end_ms <= gap_ms
        ):
            last = merged[-1]
            merged[-1] = ClipWindow(
                event_id=f"{last.event_id}+{window.event_id}"
                if window.event_id not in last.event_id
                else last.event_id,
                event_type=last.event_type,
                start_ms=last.start_ms,
                end_ms=max(last.end_ms, window.end_ms),
            )
        else:
            merged.append(window)
    return merged


def export_clip_mp4(
    window: ClipWindow,
    source_video: Path | str,
    output_path: Path | str,
    *,
    max_frames: int = 2_000,
) -> dict[str, Any]:
    """Write one clip window to a short MP4 by streaming frames with OpenCV.

    Frames are read and written one at a time (never buffered in memory).
    Raises ``RuntimeError`` when OpenCV is not installed and
    ``FileNotFoundError`` when the source cannot be opened. ``max_frames``
    caps the output so a misconfigured window cannot produce a huge file.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is not installed; MP4 export is unavailable")
    source_video = Path(source_video)
    output_path = Path(output_path)
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"cannot open video: {source_video}")
    writer = None
    written = 0
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        start_frame = max(0, int(window.start_ms / 1000.0 * fps))
        end_frame = int(window.end_ms / 1000.0 * fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(min(end_frame - start_frame + 1, max_frames)):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"cannot open MP4 writer for {output_path}")
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if written == 0 and output_path.exists():
        output_path.unlink()
    return {
        "event_id": window.event_id,
        "path": str(output_path),
        "frames_written": written,
        "wrote_file": written > 0,
    }
