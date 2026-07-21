from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

import pyarrow.parquet as pq

from football_analytics.full_match.manifest import (
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_table,
    save_chunk_status_parquet,
)
from football_analytics.full_match.schemas import (
    CHUNK_STATUS_SCHEMA,
    ChunkRecord,
    ChunkStatus,
    chunk_records_to_table,
)


def no_leftover_tmp_files(directory: Path) -> bool:
    return not [item for item in directory.iterdir() if item.suffix == ".tmp"]


def test_atomic_json_roundtrip_without_tmp_leftovers(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "payload.json"
    atomic_write_json(target, {"a": 1, "b": [1, 2, 3]})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": [1, 2, 3]}
    assert no_leftover_tmp_files(target.parent)


def test_atomic_json_replaces_existing_content(tmp_path: Path) -> None:
    target = tmp_path / "payload.json"
    atomic_write_json(target, {"version": 1})
    atomic_write_json(target, {"version": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 2}
    assert no_leftover_tmp_files(tmp_path)


def test_atomic_json_failed_rename_keeps_old_file_and_cleans_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "payload.json"
    atomic_write_json(target, {"version": 1})

    def broken_replace(src, dst):
        raise OSError("simulated crash during rename")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="simulated crash"):
        atomic_write_json(target, {"version": 2})
    monkeypatch.undo()

    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}
    assert no_leftover_tmp_files(tmp_path)


def test_atomic_parquet_roundtrip_without_tmp_leftovers(tmp_path: Path) -> None:
    target = tmp_path / "table.parquet"
    frame = pd.DataFrame({"chunk": [0, 1, 2], "status": ["PASS", "PASS", "FAILED"]})
    atomic_write_parquet(target, frame)
    loaded = pd.read_parquet(target)
    pd.testing.assert_frame_equal(loaded, frame)
    assert no_leftover_tmp_files(tmp_path)


def test_atomic_parquet_failure_cleans_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "table.parquet"

    def broken_replace(src, dst):
        raise OSError("simulated crash during rename")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="simulated crash"):
        atomic_write_parquet(target, pd.DataFrame({"a": [1]}))
    monkeypatch.undo()

    assert not target.exists()
    assert no_leftover_tmp_files(tmp_path)


def test_typed_chunk_status_schema_roundtrip(tmp_path: Path) -> None:
    records = [
        ChunkRecord(
            camera_id="cam",
            chunk_index=0,
            start_seconds=0.0,
            end_seconds=120.0,
            frame_start=0,
            frame_end=3000,
            status=ChunkStatus.PASS,
            attempts=1,
            wall_seconds=1.25,
            result_path="chunks/cam/chunk_00000.json",
        )
    ]
    table = chunk_records_to_table(records)
    assert table.schema.equals(CHUNK_STATUS_SCHEMA)

    target = tmp_path / "chunk_status.parquet"
    save_chunk_status_parquet(target, records)
    loaded = pq.read_table(target)
    assert loaded.schema.equals(CHUNK_STATUS_SCHEMA)
    assert loaded.column("status").to_pylist() == ["PASS"]
    assert no_leftover_tmp_files(tmp_path)

    atomic_write_table(tmp_path / "direct.parquet", table)
    assert pq.read_table(tmp_path / "direct.parquet").num_rows == 1
