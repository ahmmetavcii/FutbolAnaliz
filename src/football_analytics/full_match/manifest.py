"""Atomic persistence for full-match manifests and run state."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

MATCH_MANIFEST_NAME = "match_manifest.json"
CHUNK_MANIFEST_NAME = "chunk_manifest.json"
CHUNK_STATUS_PARQUET_NAME = "chunk_status.parquet"
RUN_STATE_NAME = "run_state.json"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a temporary file and atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Write a Parquet artifact via a temporary file and atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        frame.to_parquet(tmp_name, index=False)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def atomic_write_table(path: Path, table: pa.Table) -> None:
    """Write a typed PyArrow Table via a temporary file and atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        pq.write_table(table, tmp_name)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def save_chunk_status_parquet(path: Path, records: list[Any]) -> None:
    """Persist chunk records using the typed CHUNK_STATUS_SCHEMA."""
    from .schemas import chunk_records_to_table

    atomic_write_table(path, chunk_records_to_table(records))


def save_model(path: Path, model: BaseModel) -> None:
    atomic_write_json(path, model.model_dump(mode="json"))


def load_model(path: Path, model_type: type[BaseModel]) -> Any:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return model_type.model_validate(payload)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload
