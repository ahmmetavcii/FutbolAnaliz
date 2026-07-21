"""Atomic Parquet export with schema-aware read-back validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def export_parquet(
    path: Path,
    frame: pd.DataFrame | None = None,
    *,
    rows: list[dict[str, Any]] | None = None,
    schema: pa.Schema | None = None,
) -> dict[str, Any]:
    """Atomically write a Parquet artifact and validate by reading it back.

    Accepts either a DataFrame or typed rows plus an explicit pyarrow schema
    (the latter produces correctly typed empty artifacts, matching the
    conventions in football_analytics.utils.io).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if (frame is None) == (rows is None):
        raise ValueError("Provide exactly one of frame= or rows=")
    if rows is not None:
        if schema is None:
            raise ValueError("rows= requires an explicit schema=")
        table = pa.Table.from_pylist(rows, schema=schema)
    else:
        table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".part")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        pq.write_table(table, tmp_path)
        read_back = pq.read_table(tmp_path)
        if read_back.num_rows != table.num_rows:
            raise RuntimeError(f"Parquet read-back row-count mismatch for {path}")
        if read_back.schema.names != table.schema.names:
            raise RuntimeError(f"Parquet read-back column mismatch for {path}")
        if schema is not None and not read_back.schema.equals(schema, check_metadata=False):
            raise RuntimeError(f"Parquet read-back schema mismatch for {path}")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {
        "path": str(path),
        "rows": int(table.num_rows),
        "columns": list(table.schema.names),
        "size_bytes": path.stat().st_size,
        "validated": True,
    }
