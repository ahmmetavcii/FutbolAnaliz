from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_analytics.full_match import (
    ChunkManifest,
    ChunkStatus,
    MatchManifest,
    RunState,
    prepare_full_match,
    resume_full_match,
    run_full_match,
)
from football_analytics.full_match.manifest import (
    CHUNK_MANIFEST_NAME,
    RUN_STATE_NAME,
    load_model,
)
from football_analytics.full_match.resume import (
    build_fingerprints,
    invalidate_stale_chunks,
    mark_chunk_result,
    should_run_chunk,
)
from football_analytics.full_match.scheduler import ChunkScheduler


@pytest.fixture()
def prepared(tmp_path: Path, make_video, base_config) -> dict:
    video = make_video(tmp_path / "cam.mp4", seconds=65.0, fps=10.0)
    prepared_dir = tmp_path / "prepared"
    report = prepare_full_match(
        inputs=[video],
        camera_ids=["cam_1"],
        config=base_config,
        output_dir=prepared_dir,
        match_id="match-001",
    )
    return {
        "video": video,
        "prepared_dir": prepared_dir,
        "run_dir": tmp_path / "run",
        "config": base_config,
        "report": report,
    }


def load_chunks(run_dir: Path) -> ChunkManifest:
    return load_model(run_dir / CHUNK_MANIFEST_NAME, ChunkManifest)


def test_prepare_writes_manifests_and_plans_chunks(prepared) -> None:
    report = prepared["report"]
    assert report["status"] == "PASS"
    assert report["total_chunks"] == 3  # 65s at 30s chunks
    manifest = load_model(
        prepared["prepared_dir"] / "match_manifest.json", MatchManifest
    )
    assert manifest.match_id == "match-001"
    assert manifest.cameras[0].probe is not None
    assert manifest.cameras[0].probe.decodable


def test_prepare_refuses_overwrite_without_force(prepared, base_config) -> None:
    with pytest.raises(ValueError, match="force"):
        prepare_full_match(
            inputs=[prepared["video"]],
            camera_ids=["cam_1"],
            config=base_config,
            output_dir=prepared["prepared_dir"],
            match_id="match-001",
        )


def test_prepare_rejects_three_cameras(tmp_path: Path, make_video, base_config) -> None:
    videos = [make_video(tmp_path / f"c{i}.mp4", seconds=4.0) for i in range(3)]
    with pytest.raises(ValueError, match="cameras"):
        prepare_full_match(
            inputs=videos,
            config=base_config,
            output_dir=tmp_path / "prep3",
            match_id="m3",
        )


def test_run_full_match_processes_all_chunks_honestly(prepared) -> None:
    result = run_full_match(
        prepared_dir=prepared["prepared_dir"],
        run_dir=prepared["run_dir"],
        config=prepared["config"],
    )
    assert result["status"] == "PASS"
    assert result["chunks"]["counts"]["PASS"] == 3

    state = load_model(prepared["run_dir"] / RUN_STATE_NAME, RunState)
    stages = {record.name: record.status.value for record in state.stages}
    assert stages["chunks"] == "PASS"
    assert stages["consolidation"] == "PASS"
    # Model stages are not fabricated in infrastructure-only runs.
    assert stages["detection"] == "SKIPPED"
    assert stages["tracking"] == "SKIPPED"
    assert stages["events"] == "SKIPPED"

    for record in load_chunks(prepared["run_dir"]).records:
        payload = json.loads(
            (prepared["run_dir"] / record.result_path).read_text(encoding="utf-8")
        )
        assert payload["frames_decoded"] > 0
        assert payload["model_outputs"] is None
        assert payload["model_stage_status"] == "NOT_AVAILABLE"


def test_resume_skips_pass_chunks(prepared) -> None:
    run_full_match(
        prepared_dir=prepared["prepared_dir"],
        run_dir=prepared["run_dir"],
        config=prepared["config"],
    )
    plan = resume_full_match(run_dir=prepared["run_dir"], dry_run=True)
    assert plan["chunks_to_run"] == 0

    result = resume_full_match(run_dir=prepared["run_dir"])
    assert result["status"] == "PASS"
    assert result["chunks"]["executed_chunks"] == 0
    assert result["chunks"]["skipped_chunks"] == 3


def test_resume_repairs_interrupted_running_chunks(prepared) -> None:
    run_full_match(
        prepared_dir=prepared["prepared_dir"],
        run_dir=prepared["run_dir"],
        config=prepared["config"],
    )
    manifest = load_chunks(prepared["run_dir"])
    manifest.records[1].status = ChunkStatus.RUNNING
    from football_analytics.full_match.manifest import save_model

    save_model(prepared["run_dir"] / CHUNK_MANIFEST_NAME, manifest)

    result = resume_full_match(run_dir=prepared["run_dir"], repair_manifests=True)
    assert result["repaired_chunks"] == 1
    assert result["status"] == "PASS"
    assert result["chunks"]["executed_chunks"] == 1


def test_failed_chunks_retry_up_to_limit(prepared) -> None:
    match_manifest = load_model(
        prepared["prepared_dir"] / "match_manifest.json", MatchManifest
    )
    chunk_manifest = load_model(
        prepared["prepared_dir"] / CHUNK_MANIFEST_NAME, ChunkManifest
    )
    fingerprints = build_fingerprints(prepared["config"], match_manifest)

    def always_fails(video_path, record):
        raise RuntimeError("synthetic stage crash")

    scheduler = ChunkScheduler(
        run_dir=prepared["run_dir"],
        match_manifest=match_manifest,
        chunk_manifest=chunk_manifest,
        fingerprints=fingerprints,
        retry_limit=3,
        fail_fast=False,
        processor=always_fails,
    )
    for expected_status in (ChunkStatus.RETRY, ChunkStatus.RETRY, ChunkStatus.FAILED):
        scheduler.run()
        assert chunk_manifest.records[0].status == expected_status

    record = chunk_manifest.records[0]
    assert record.attempts == 3
    assert "synthetic stage crash" in record.error
    # Retry budget exhausted: the chunk must not be scheduled again.
    assert not should_run_chunk(record, fingerprints, retry_limit=3)
    result = scheduler.run()
    assert result["executed_chunks"] == 0


def test_flaky_chunk_recovers_on_retry(prepared) -> None:
    from football_analytics.full_match.scheduler import default_chunk_processor

    match_manifest = load_model(
        prepared["prepared_dir"] / "match_manifest.json", MatchManifest
    )
    chunk_manifest = load_model(
        prepared["prepared_dir"] / CHUNK_MANIFEST_NAME, ChunkManifest
    )
    fingerprints = build_fingerprints(prepared["config"], match_manifest)
    attempts: dict[int, int] = {}

    def flaky(video_path, record):
        attempts[record.chunk_index] = attempts.get(record.chunk_index, 0) + 1
        if attempts[record.chunk_index] == 1:
            raise RuntimeError("transient failure")
        return default_chunk_processor(video_path, record)

    scheduler = ChunkScheduler(
        run_dir=prepared["run_dir"],
        match_manifest=match_manifest,
        chunk_manifest=chunk_manifest,
        fingerprints=fingerprints,
        retry_limit=3,
        fail_fast=False,
        processor=flaky,
    )
    scheduler.run()  # every chunk fails once -> RETRY
    assert {r.status for r in chunk_manifest.records} == {ChunkStatus.RETRY}
    result = scheduler.run()  # second attempt succeeds
    assert result["ok"] is True
    assert {r.status for r in chunk_manifest.records} == {ChunkStatus.PASS}
    assert all(r.attempts == 2 for r in chunk_manifest.records)


def test_config_change_invalidates_completed_chunks(prepared) -> None:
    first = run_full_match(
        prepared_dir=prepared["prepared_dir"],
        run_dir=prepared["run_dir"],
        config=prepared["config"],
    )
    assert first["invalidated_chunks"] == 0

    changed_config = dict(prepared["config"])
    changed_config["inference"] = {"batch_size": 1, "new_setting": True}
    second = run_full_match(
        prepared_dir=prepared["prepared_dir"],
        run_dir=prepared["run_dir"],
        config=changed_config,
    )
    assert second["invalidated_chunks"] == 3
    assert second["chunks"]["executed_chunks"] == 3
    assert second["status"] == "PASS"

    # Same config again: nothing is invalidated or re-executed.
    third = run_full_match(
        prepared_dir=prepared["prepared_dir"],
        run_dir=prepared["run_dir"],
        config=changed_config,
    )
    assert third["invalidated_chunks"] == 0
    assert third["chunks"]["executed_chunks"] == 0


def test_model_change_invalidates_via_fingerprints(prepared) -> None:
    match_manifest = load_model(
        prepared["prepared_dir"] / "match_manifest.json", MatchManifest
    )
    chunk_manifest = load_model(
        prepared["prepared_dir"] / CHUNK_MANIFEST_NAME, ChunkManifest
    )
    fingerprints = build_fingerprints(prepared["config"], match_manifest)
    record = chunk_manifest.records[0]
    mark_chunk_result(record, ok=True, fingerprints=fingerprints)
    assert record.status == ChunkStatus.PASS

    weights = prepared["prepared_dir"] / "weights.bin"
    weights.write_bytes(b"new model weights")
    new_fingerprints = build_fingerprints(
        prepared["config"], match_manifest, model_files={"detector": weights}
    )
    invalidated = invalidate_stale_chunks(chunk_manifest, new_fingerprints)
    assert [r.chunk_index for r in invalidated] == [0]
    assert record.status == ChunkStatus.INVALIDATED
    assert should_run_chunk(record, new_fingerprints)


def test_rerun_from_stage_invalidates_chunks_and_downstream(prepared) -> None:
    run_full_match(
        prepared_dir=prepared["prepared_dir"],
        run_dir=prepared["run_dir"],
        config=prepared["config"],
    )
    result = resume_full_match(run_dir=prepared["run_dir"], rerun_from_stage="chunks")
    assert "chunks" in result["invalidated_stages"]
    assert "consolidation" in result["invalidated_stages"]
    assert "prepare" not in result["invalidated_stages"]
    assert result["chunks"]["executed_chunks"] == 3
    assert result["status"] == "PASS"
