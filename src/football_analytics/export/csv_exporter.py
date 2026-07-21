"""Atomic CSV export with read-back validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def export_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    """Atomically write frame as CSV and validate shape/columns on read-back."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".part")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        frame.to_csv(tmp_path, index=False, encoding="utf-8", lineterminator="\n")
        if len(frame.columns) == 0:
            # A zero-column frame yields a header-less file pandas cannot parse.
            read_back = pd.DataFrame()
        else:
            read_back = pd.read_csv(tmp_path, encoding="utf-8")
        if list(read_back.columns) != [str(column) for column in frame.columns]:
            raise RuntimeError(f"CSV read-back column mismatch for {path}")
        if len(read_back) != len(frame):
            raise RuntimeError(
                f"CSV read-back row-count mismatch for {path}: "
                f"wrote {len(frame)}, read {len(read_back)}"
            )
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "size_bytes": path.stat().st_size,
        "validated": True,
    }
