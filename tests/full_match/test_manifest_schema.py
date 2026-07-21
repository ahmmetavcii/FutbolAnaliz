from __future__ import annotations

import pytest
from pydantic import ValidationError

from football_analytics.full_match.schemas import (
    CameraRole,
    CameraSpec,
    ChunkManifest,
    ChunkRecord,
    ChunkStatus,
    Fingerprints,
    MatchManifest,
)

SHA = "a" * 64


def make_camera(camera_id: str, role: CameraRole = CameraRole.TACTICAL_FULL) -> CameraSpec:
    return CameraSpec(camera_id=camera_id, role=role, path=f"/videos/{camera_id}.mp4", sha256=SHA)


@pytest.mark.parametrize("count", [1, 2, 4])
def test_manifest_accepts_supported_camera_counts(count: int) -> None:
    manifest = MatchManifest(
        match_id="m1",
        cameras=[make_camera(f"cam_{i}") for i in range(count)],
    )
    assert len(manifest.cameras) == count
    assert manifest.chunk_seconds == 120.0


@pytest.mark.parametrize("count", [0, 3, 5, 6])
def test_manifest_rejects_unsupported_camera_counts(count: int) -> None:
    with pytest.raises(ValidationError, match="cameras"):
        MatchManifest(
            match_id="m1",
            cameras=[make_camera(f"cam_{i}") for i in range(count)],
        )


def test_manifest_rejects_duplicate_camera_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        MatchManifest(match_id="m1", cameras=[make_camera("cam"), make_camera("cam")])


def test_all_required_camera_roles_exist() -> None:
    expected = {
        "tactical_full",
        "tactical_left",
        "tactical_right",
        "broadcast",
        "goal_left",
        "goal_right",
        "reverse_angle",
        "custom",
    }
    assert {role.value for role in CameraRole} == expected


@pytest.mark.parametrize("value", [29.9, 300.1, 0, -5])
def test_manifest_rejects_out_of_range_chunk_seconds(value: float) -> None:
    with pytest.raises(ValidationError):
        MatchManifest(match_id="m1", chunk_seconds=value, cameras=[make_camera("cam")])


def test_chunk_statuses_are_complete() -> None:
    expected = {
        "PENDING",
        "RUNNING",
        "PASS",
        "FAILED",
        "RETRY",
        "SKIPPED",
        "INVALID_INPUT",
        "INVALIDATED",
    }
    assert {status.value for status in ChunkStatus} == expected


def test_chunk_manifest_counts_by_status() -> None:
    manifest = ChunkManifest(
        match_id="m1",
        chunk_seconds=120,
        records=[
            ChunkRecord(
                camera_id="cam",
                chunk_index=i,
                start_seconds=i * 120.0,
                end_seconds=(i + 1) * 120.0,
                frame_start=i * 3000,
                frame_end=(i + 1) * 3000,
                status=status,
            )
            for i, status in enumerate([ChunkStatus.PASS, ChunkStatus.PASS, ChunkStatus.FAILED])
        ],
    )
    counts = manifest.counts_by_status()
    assert counts["PASS"] == 2
    assert counts["FAILED"] == 1
    assert counts["PENDING"] == 0


def test_fingerprints_combined_changes_with_each_component() -> None:
    base = Fingerprints(config="c1", model="m1", inputs={"cam": "i1"})
    assert base.combined("cam") == base.combined("cam")
    assert base.combined("cam") != Fingerprints(
        config="c2", model="m1", inputs={"cam": "i1"}
    ).combined("cam")
    assert base.combined("cam") != Fingerprints(
        config="c1", model="m2", inputs={"cam": "i1"}
    ).combined("cam")
    assert base.combined("cam") != Fingerprints(
        config="c1", model="m1", inputs={"cam": "i2"}
    ).combined("cam")
