"""Timeline synchronization between cameras using explicit offset estimates.

Offsets always map a camera's local clock onto the shared reference timeline:
``reference_time = local_time + offset_seconds``. Manual offsets from
:class:`~football_analytics.multicamera.camera_config.CameraConfig` are the
baseline; measured estimates (audio or visual) can refine them when their
confidence clears a configurable threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .camera_config import MultiCameraSetup


class OffsetSource(str, Enum):
    MANUAL = "manual"
    AUDIO = "audio"
    VISUAL = "visual"


@dataclass(frozen=True)
class OffsetEstimate:
    """A single measurement of a camera's offset against the reference timeline."""

    camera_id: str
    offset_seconds: float
    confidence: float
    source: OffsetSource
    # Reference-timeline instant at which the estimate was taken; used to
    # anchor drift models. ``None`` means "valid for the whole recording".
    measured_at_seconds: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class TimelineSynchronizer:
    """Maps per-camera local timestamps onto a shared reference timeline."""

    setup: MultiCameraSetup
    minimum_estimate_confidence: float = 0.6
    _estimates: dict[str, OffsetEstimate] = field(default_factory=dict, repr=False)
    _drift_rates: dict[str, float] = field(default_factory=dict, repr=False)

    def apply_estimate(self, estimate: OffsetEstimate) -> bool:
        """Adopt a measured offset estimate; returns True when accepted.

        Estimates are rejected when the camera is unknown, when confidence is
        below the threshold, or when a stored estimate has higher confidence.
        """
        if estimate.camera_id not in self.setup:
            raise KeyError(f"unknown camera_id: {estimate.camera_id}")
        if estimate.camera_id == self.setup.reference_camera_id:
            return False
        if estimate.confidence < self.minimum_estimate_confidence:
            return False
        current = self._estimates.get(estimate.camera_id)
        if current is not None and current.confidence > estimate.confidence:
            return False
        self._estimates[estimate.camera_id] = estimate
        return True

    def set_drift_rate(self, camera_id: str, seconds_per_second: float) -> None:
        """Set a linear clock drift rate for ``camera_id`` (see drift_correction)."""
        if camera_id not in self.setup:
            raise KeyError(f"unknown camera_id: {camera_id}")
        self._drift_rates[camera_id] = float(seconds_per_second)

    def offset_seconds(self, camera_id: str, local_time_seconds: float = 0.0) -> float:
        """Effective offset for a camera at a given local time (drift-aware)."""
        camera = self.setup.camera(camera_id)
        estimate = self._estimates.get(camera_id)
        base = estimate.offset_seconds if estimate is not None else camera.manual_offset_seconds
        drift_rate = self._drift_rates.get(camera_id, 0.0)
        anchor = 0.0
        if estimate is not None and estimate.measured_at_seconds is not None:
            anchor = estimate.measured_at_seconds
        return base + drift_rate * (local_time_seconds - anchor)

    def offset_source(self, camera_id: str) -> OffsetSource:
        estimate = self._estimates.get(camera_id)
        return estimate.source if estimate is not None else OffsetSource.MANUAL

    def to_reference_time(self, camera_id: str, local_time_seconds: float) -> float:
        return local_time_seconds + self.offset_seconds(camera_id, local_time_seconds)

    def to_local_time(self, camera_id: str, reference_time_seconds: float) -> float:
        """Invert :meth:`to_reference_time`, accounting for linear drift."""
        camera = self.setup.camera(camera_id)
        estimate = self._estimates.get(camera_id)
        base = estimate.offset_seconds if estimate is not None else camera.manual_offset_seconds
        drift_rate = self._drift_rates.get(camera_id, 0.0)
        anchor = 0.0
        if estimate is not None and estimate.measured_at_seconds is not None:
            anchor = estimate.measured_at_seconds
        # reference = local + base + drift_rate * (local - anchor)
        denominator = 1.0 + drift_rate
        if abs(denominator) < 1e-12:
            raise ValueError(f"drift rate for {camera_id} makes the timeline non-invertible")
        return (reference_time_seconds - base + drift_rate * anchor) / denominator

    def to_reference_frame(self, camera_id: str, frame_index: int) -> float:
        """Reference-timeline timestamp (seconds) of a camera's frame index."""
        camera = self.setup.camera(camera_id)
        local_time = frame_index / camera.fps
        return self.to_reference_time(camera_id, local_time)


def load_offsets_file(path) -> dict[str, float]:
    """Read a manual offsets mapping from JSON or YAML.

    Accepted layouts: ``{"cam1": 0.0, ...}`` or ``{"offsets": {"cam1": 0.0}}``.
    """
    import json
    from pathlib import Path

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if isinstance(raw, dict) and isinstance(raw.get("offsets"), dict):
        raw = raw["offsets"]
    if not isinstance(raw, dict):
        raise ValueError(f"offsets file must contain a mapping: {path}")
    return {str(camera_id): float(value) for camera_id, value in raw.items()}


def _manifest_camera_ids(prepared_dir) -> list[str]:
    import json

    manifest_path = prepared_dir / "match_manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [camera["camera_id"] for camera in manifest.get("cameras", [])]


def synchronize_cameras(
    *,
    prepared_dir=None,
    run_dir=None,
    method: str = "manual",
    reference_camera: str | None = None,
    offsets_path=None,
    manual_offsets=None,
    output_path=None,
    max_offset_seconds: float = 30.0,
    **_,
):
    """CLI/orchestration entry point used by ``scripts/sync_cameras.py``.

    - ``manual``: offsets come from ``offsets_path`` / ``manual_offsets`` and
      are re-expressed relative to ``reference_camera``.
    - ``audio``: per-camera mono waveforms are loaded from
      ``<prepared_dir>/audio/<camera_id>.npy`` (plus ``sample_rate.json`` with
      ``{"sample_rate_hz": ...}``) and aligned by cross-correlation.
    - ``timecode``: not implemented; reported as such (no offsets invented).
    """
    import json
    from pathlib import Path

    from football_analytics.utils.io import write_json

    prepared_dir = Path(prepared_dir) if prepared_dir else None
    run_dir = Path(run_dir) if run_dir else prepared_dir
    if run_dir is None and output_path is None:
        raise ValueError("run_dir, prepared_dir, or output_path is required")
    destination = Path(output_path) if output_path else run_dir / "sync_report.json"

    camera_ids = _manifest_camera_ids(prepared_dir) if prepared_dir else []
    offsets: dict[str, float] = {}
    per_camera: dict[str, dict] = {}
    status = "PASS"
    note = ""

    if method == "manual":
        offsets = dict(manual_offsets or {})
        if offsets_path is not None:
            offsets.update(load_offsets_file(offsets_path))
        if not offsets:
            offsets = {camera_id: 0.0 for camera_id in camera_ids}
            note = "no offsets provided; defaulted every manifest camera to 0.0"
        if reference_camera is None and offsets:
            reference_camera = next(iter(offsets))
        if reference_camera is not None and reference_camera in offsets:
            reference_value = offsets[reference_camera]
            offsets = {cid: value - reference_value for cid, value in offsets.items()}
        per_camera = {
            camera_id: {"offset_seconds": value, "confidence": 1.0, "source": "manual"}
            for camera_id, value in offsets.items()
        }
    elif method == "audio":
        from .audio_sync import AudioSyncConfig, estimate_audio_sync

        if prepared_dir is None:
            raise ValueError("prepared_dir is required for audio synchronization")
        audio_dir = prepared_dir / "audio"
        waveform_paths = sorted(audio_dir.glob("*.npy")) if audio_dir.is_dir() else []
        if not waveform_paths:
            status = "FAIL"
            note = f"no audio waveforms found under {audio_dir}"
        else:
            import numpy as np

            sample_rate = 16_000.0
            meta_path = audio_dir / "sample_rate.json"
            if meta_path.is_file():
                sample_rate = float(
                    json.loads(meta_path.read_text(encoding="utf-8"))["sample_rate_hz"]
                )
            waveforms = {path.stem: np.load(path) for path in waveform_paths}
            if reference_camera is None:
                reference_camera = next(iter(waveforms))
            if reference_camera not in waveforms:
                raise KeyError(f"reference camera {reference_camera!r} has no waveform")
            config = AudioSyncConfig(
                sample_rate_hz=sample_rate, max_offset_seconds=max_offset_seconds
            )
            for camera_id, waveform in waveforms.items():
                if camera_id == reference_camera:
                    offsets[camera_id] = 0.0
                    per_camera[camera_id] = {
                        "offset_seconds": 0.0,
                        "confidence": 1.0,
                        "drift_seconds_per_second": 0.0,
                        "source": "reference",
                    }
                    continue
                result = estimate_audio_sync(
                    camera_id, waveforms[reference_camera], waveform, config
                )
                offsets[camera_id] = result.offset_seconds
                per_camera[camera_id] = {
                    "offset_seconds": result.offset_seconds,
                    "confidence": result.confidence,
                    "drift_seconds_per_second": result.drift_seconds_per_second,
                    "source": "audio",
                }
            note = f"audio cross-correlation against reference {reference_camera!r}"
    elif method == "timecode":
        status = "FAIL"
        note = "timecode synchronization is not implemented; use audio or manual"
    else:
        raise ValueError(f"unknown synchronization method: {method!r}")

    report = {
        "status": status,
        "method": method,
        "reference_camera": reference_camera,
        "offsets_seconds": offsets,
        "cameras": per_camera,
        "note": note,
    }
    write_json(destination, report)
    return report
