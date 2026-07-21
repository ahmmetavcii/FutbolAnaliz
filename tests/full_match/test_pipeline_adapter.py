"""Adapter contract tests: interface shape, context, and input validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from football_analytics.full_match.pipeline_adapter import (
    ChunkContext,
    ExistingPipelineAdapter,
    PipelineAdapterError,
    run_chunk_via_existing_pipeline,
)
from football_analytics.full_match.schemas import ChunkRecord


def _record() -> ChunkRecord:
    return ChunkRecord(
        camera_id="camera_1",
        chunk_index=0,
        start_seconds=0.0,
        end_seconds=30.0,
        frame_start=0,
        frame_end=750,
    )


def test_adapter_module_importable_and_callable() -> None:
    assert callable(run_chunk_via_existing_pipeline)
    adapter = ExistingPipelineAdapter()
    for name in (
        "validate_inputs",
        "prepare",
        "run_chunk",
        "validate_outputs",
        "get_artifact_manifest",
        "cleanup",
    ):
        assert callable(getattr(adapter, name))


def test_chunk_context_from_record_carries_required_fields(tmp_path: Path) -> None:
    context = ChunkContext.from_record(
        tmp_path / "video.mp4",
        _record(),
        run_id="run-x",
        output_dir=tmp_path / "out",
        config={"model": {"batch": 1}},
        period=2,
    )
    assert context.run_id == "run-x"
    assert context.camera_id == "camera_1"
    assert context.period == 2
    assert context.chunk_index == 0
    assert context.source_path == tmp_path / "video.mp4"
    assert (context.frame_start, context.frame_end) == (0, 750)
    assert (context.start_seconds, context.end_seconds) == (0.0, 30.0)
    assert context.config == {"model": {"batch": 1}}
    assert context.output_dir == tmp_path / "out"


def test_validate_inputs_rejects_missing_video(tmp_path: Path) -> None:
    adapter = ExistingPipelineAdapter()
    context = ChunkContext.from_record(
        tmp_path / "missing.mp4",
        _record(),
        run_id="run-x",
        output_dir=tmp_path,
    )
    with pytest.raises(PipelineAdapterError):
        adapter.validate_inputs(context)


def test_validate_outputs_rejects_empty_run_dir(tmp_path: Path) -> None:
    adapter = ExistingPipelineAdapter()
    with pytest.raises(PipelineAdapterError):
        adapter.validate_outputs(tmp_path)
