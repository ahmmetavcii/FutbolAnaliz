"""Audio-based synchronization via normalized cross-correlation of waveforms.

The public API operates directly on mono waveforms (1-D float arrays) and never
touches audio files or codecs. It produces
:class:`~football_analytics.multicamera.synchronization.OffsetEstimate` values
with the convention ``reference_time = local_time + offset_seconds``: a
positive offset means the camera's clock lags the reference recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from .synchronization import OffsetEstimate, OffsetSource

_EPSILON = 1e-12


@dataclass(frozen=True)
class AudioSyncConfig:
    sample_rate_hz: float = 16_000.0
    max_offset_seconds: float = 30.0
    window_seconds: float = 20.0
    hop_seconds: float = 60.0
    minimum_window_confidence: float = 0.3

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.max_offset_seconds <= 0:
            raise ValueError("max_offset_seconds must be positive")
        if self.window_seconds <= 0 or self.hop_seconds <= 0:
            raise ValueError("window_seconds and hop_seconds must be positive")


@dataclass(frozen=True)
class WindowedOffset:
    """Offset measured in one correlation window."""

    window_start_seconds: float
    offset_seconds: float
    confidence: float


@dataclass(frozen=True)
class AudioSyncResult:
    camera_id: str
    offset_seconds: float
    confidence: float
    drift_seconds_per_second: float
    windows: tuple[WindowedOffset, ...]

    @property
    def estimated_offset_seconds(self) -> float:
        return self.offset_seconds

    @property
    def drift_seconds_per_hour(self) -> float:
        return self.drift_seconds_per_second * 3600.0

    def to_dict(self) -> dict[str, float | str]:
        return {
            "camera_id": self.camera_id,
            "estimated_offset_seconds": self.estimated_offset_seconds,
            "confidence": self.confidence,
            "drift_seconds_per_hour": self.drift_seconds_per_hour,
        }

    def to_offset_estimate(self) -> OffsetEstimate:
        anchor = self.windows[0].window_start_seconds if self.windows else 0.0
        return OffsetEstimate(
            camera_id=self.camera_id,
            offset_seconds=self.offset_seconds,
            confidence=self.confidence,
            source=OffsetSource.AUDIO,
            measured_at_seconds=anchor,
        )


def _as_waveform(raw: Sequence[float] | NDArray[np.floating]) -> NDArray[np.float64]:
    waveform = np.asarray(raw, dtype=np.float64)
    if waveform.ndim != 1:
        raise ValueError(f"waveform must be 1-D, got shape {waveform.shape}")
    if waveform.size == 0:
        raise ValueError("waveform is empty")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("waveform contains non-finite samples")
    return waveform - float(np.mean(waveform))


def cross_correlate_offset(
    reference: Sequence[float] | NDArray[np.floating],
    target: Sequence[float] | NDArray[np.floating],
    config: AudioSyncConfig | None = None,
) -> tuple[float, float]:
    """Estimate the offset that aligns ``target`` onto ``reference``.

    Returns ``(offset_seconds, confidence)`` where confidence is the peak
    normalized correlation in ``[0, 1]``. The search range is limited to
    ``config.max_offset_seconds`` in either direction.
    """
    cfg = config or AudioSyncConfig()
    ref = _as_waveform(reference)
    tgt = _as_waveform(target)

    correlation = signal.correlate(ref, tgt, mode="full", method="auto")
    # Lag k means: target sample i aligns with reference sample i + k.
    lags = signal.correlation_lags(ref.size, tgt.size, mode="full")
    norm = float(np.linalg.norm(ref) * np.linalg.norm(tgt))
    if norm < _EPSILON:
        return 0.0, 0.0
    normalized = correlation / norm

    max_lag_samples = int(round(cfg.max_offset_seconds * cfg.sample_rate_hz))
    in_range = np.abs(lags) <= max_lag_samples
    if not np.any(in_range):
        return 0.0, 0.0
    candidate_scores = np.where(in_range, normalized, -np.inf)
    best = int(np.argmax(candidate_scores))
    offset_seconds = float(lags[best]) / cfg.sample_rate_hz
    confidence = float(np.clip(normalized[best], 0.0, 1.0))
    return offset_seconds, confidence


def windowed_offsets(
    reference: Sequence[float] | NDArray[np.floating],
    target: Sequence[float] | NDArray[np.floating],
    config: AudioSyncConfig | None = None,
) -> tuple[WindowedOffset, ...]:
    """Measure the alignment offset in successive windows of the recordings.

    Windows are cut on the reference timeline; the target segment is expanded
    by the maximum offset on each side so the true peak stays in range.
    """
    cfg = config or AudioSyncConfig()
    ref = _as_waveform(reference)
    tgt = _as_waveform(target)

    window_samples = int(round(cfg.window_seconds * cfg.sample_rate_hz))
    hop_samples = int(round(cfg.hop_seconds * cfg.sample_rate_hz))
    margin_samples = int(round(cfg.max_offset_seconds * cfg.sample_rate_hz))
    results: list[WindowedOffset] = []
    for start in range(0, max(ref.size - window_samples + 1, 1), hop_samples):
        ref_window = ref[start : start + window_samples]
        if ref_window.size < window_samples // 2:
            break
        tgt_start = max(start - margin_samples, 0)
        tgt_window = tgt[tgt_start : start + window_samples + margin_samples]
        if tgt_window.size < window_samples // 2:
            break
        offset, confidence = cross_correlate_offset(ref_window, tgt_window, cfg)
        # Compensate for the target window being cut earlier than the reference
        # window: cutting at tgt_start reduces the apparent lag by start - tgt_start.
        offset += float(start - tgt_start) / cfg.sample_rate_hz
        results.append(
            WindowedOffset(
                window_start_seconds=start / cfg.sample_rate_hz,
                offset_seconds=offset,
                confidence=confidence,
            )
        )
    return tuple(results)


def estimate_audio_sync(
    camera_id: str,
    reference: Sequence[float] | NDArray[np.floating],
    target: Sequence[float] | NDArray[np.floating],
    config: AudioSyncConfig | None = None,
) -> AudioSyncResult:
    """Full audio sync estimate: global offset, confidence, and linear drift.

    Drift is a least-squares slope over per-window offsets; it is only
    reported when at least two confident windows exist.
    """
    cfg = config or AudioSyncConfig()
    windows = windowed_offsets(reference, target, cfg)
    confident = [w for w in windows if w.confidence >= cfg.minimum_window_confidence]
    if not confident:
        offset, confidence = cross_correlate_offset(reference, target, cfg)
        return AudioSyncResult(
            camera_id=camera_id,
            offset_seconds=offset,
            confidence=confidence,
            drift_seconds_per_second=0.0,
            windows=windows,
        )

    weights = np.asarray([w.confidence for w in confident], dtype=np.float64)
    offsets = np.asarray([w.offset_seconds for w in confident], dtype=np.float64)
    times = np.asarray([w.window_start_seconds for w in confident], dtype=np.float64)
    mean_offset = float(np.average(offsets, weights=weights))
    mean_confidence = float(np.average(weights))

    drift = 0.0
    if len(confident) >= 2 and float(np.ptp(times)) > _EPSILON:
        centered_t = times - np.average(times, weights=weights)
        centered_o = offsets - mean_offset
        denominator = float(np.sum(weights * centered_t * centered_t))
        if denominator > _EPSILON:
            drift = float(np.sum(weights * centered_t * centered_o) / denominator)

    return AudioSyncResult(
        camera_id=camera_id,
        offset_seconds=mean_offset,
        confidence=mean_confidence,
        drift_seconds_per_second=drift,
        windows=windows,
    )
