"""Atomic JSON export with read-back validation."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _to_jsonable(value: Any) -> Any:
    """Coerce numpy/path scalars so payloads built from pipeline outputs serialize."""
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def export_json(path: Path, payload: dict[str, Any], *, indent: int = 2) -> dict[str, Any]:
    """Atomically write payload as JSON and validate by reading it back.

    The temporary file lives in the destination directory so os.replace is
    atomic on POSIX filesystems; a crash mid-write never leaves a truncated
    artifact at the final path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _to_jsonable(payload)
    text = json.dumps(normalized, indent=indent, ensure_ascii=False, sort_keys=False) + "\n"

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        read_back = json.loads(tmp_path.read_text(encoding="utf-8"))
        if read_back != normalized:
            raise RuntimeError(f"JSON read-back mismatch for {path}")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "validated": True}
