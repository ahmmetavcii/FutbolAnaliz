from __future__ import annotations

import json

import pytest

from football_analytics.multicamera import (
    CameraConfig,
    MultiCameraSetup,
    OffsetEstimate,
    OffsetSource,
    TimelineSynchronizer,
    load_offsets_file,
    synchronize_cameras,
)


def build_setup() -> MultiCameraSetup:
    return MultiCameraSetup(
        cameras=[
            CameraConfig(camera_id="cam1", video_path="a.mp4"),
            CameraConfig(camera_id="cam2", video_path="b.mp4", offset_seconds=2.0),
        ]
    )


def test_manual_offsets_map_to_reference_timeline():
    sync = TimelineSynchronizer(setup=build_setup())
    assert sync.to_reference_time("cam1", 10.0) == 10.0
    assert sync.to_reference_time("cam2", 10.0) == 12.0
    assert sync.offset_source("cam2") is OffsetSource.MANUAL


def test_confident_estimate_overrides_manual_offset():
    sync = TimelineSynchronizer(setup=build_setup())
    accepted = sync.apply_estimate(
        OffsetEstimate("cam2", offset_seconds=2.5, confidence=0.9, source=OffsetSource.AUDIO)
    )
    assert accepted
    assert sync.to_reference_time("cam2", 10.0) == pytest.approx(12.5)
    assert sync.offset_source("cam2") is OffsetSource.AUDIO


def test_low_confidence_estimate_is_rejected():
    sync = TimelineSynchronizer(setup=build_setup())
    accepted = sync.apply_estimate(
        OffsetEstimate("cam2", offset_seconds=9.0, confidence=0.2, source=OffsetSource.AUDIO)
    )
    assert not accepted
    assert sync.to_reference_time("cam2", 10.0) == 12.0


def test_reference_camera_never_takes_estimates():
    sync = TimelineSynchronizer(setup=build_setup())
    assert not sync.apply_estimate(
        OffsetEstimate("cam1", offset_seconds=5.0, confidence=1.0, source=OffsetSource.AUDIO)
    )


def test_roundtrip_with_drift():
    sync = TimelineSynchronizer(setup=build_setup())
    sync.set_drift_rate("cam2", 1e-4)
    local = 1800.0
    reference = sync.to_reference_time("cam2", local)
    assert sync.to_local_time("cam2", reference) == pytest.approx(local, abs=1e-9)


def test_frame_index_mapping():
    sync = TimelineSynchronizer(setup=build_setup())
    assert sync.to_reference_frame("cam2", 50) == pytest.approx(50 / 25.0 + 2.0)


def test_load_offsets_file_json(tmp_path):
    path = tmp_path / "offsets.json"
    path.write_text(json.dumps({"offsets": {"cam1": 1.0, "cam2": 3.5}}), encoding="utf-8")
    assert load_offsets_file(path) == {"cam1": 1.0, "cam2": 3.5}


def test_synchronize_cameras_manual(tmp_path):
    offsets_path = tmp_path / "offsets.json"
    offsets_path.write_text(json.dumps({"cam1": 1.0, "cam2": 3.5}), encoding="utf-8")
    report = synchronize_cameras(
        prepared_dir=tmp_path,
        method="manual",
        reference_camera="cam1",
        offsets_path=offsets_path,
    )
    assert report["status"] == "PASS"
    assert report["offsets_seconds"] == {"cam1": 0.0, "cam2": 2.5}
    saved = json.loads((tmp_path / "sync_report.json").read_text(encoding="utf-8"))
    assert saved["offsets_seconds"]["cam2"] == 2.5


def test_synchronize_cameras_manual_requires_target(tmp_path):
    with pytest.raises(ValueError):
        synchronize_cameras(method="manual")


def test_synchronize_cameras_timecode_not_implemented(tmp_path):
    report = synchronize_cameras(prepared_dir=tmp_path, method="timecode")
    assert report["status"] == "FAIL"


def test_synchronize_cameras_audio_from_prepared_dir(tmp_path):
    import numpy as np

    rng = np.random.default_rng(7)
    fs = 1000
    master = rng.standard_normal(100 * fs)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    np.save(audio_dir / "cam1.npy", master[10 * fs : 70 * fs])
    np.save(audio_dir / "cam2.npy", master[5 * fs : 75 * fs])
    (audio_dir / "sample_rate.json").write_text(
        json.dumps({"sample_rate_hz": fs}), encoding="utf-8"
    )
    report = synchronize_cameras(
        prepared_dir=tmp_path, method="audio", reference_camera="cam1"
    )
    assert report["status"] == "PASS"
    assert report["offsets_seconds"]["cam1"] == 0.0
    assert report["offsets_seconds"]["cam2"] == pytest.approx(-5.0, abs=0.01)
    assert report["cameras"]["cam2"]["confidence"] > 0.3
