"""Shared helpers for canonical MVP-2 stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_video_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "video_manifest.json").read_text(encoding="utf-8"))


def video_fps(run_dir: Path) -> float:
    manifest = load_video_manifest(run_dir)
    value = float(manifest["working_summary"].get("avg_frame_rate") or 0.0)
    return value if value > 0 else float(manifest["opencv"].get("fps") or 25.0)


def video_frame_count(run_dir: Path) -> int:
    manifest = load_video_manifest(run_dir)
    value = manifest["working_summary"].get("nb_frames")
    if value:
        return int(value)
    return int(manifest["opencv"].get("reported_frame_count") or 0)


def match_id(run_dir: Path) -> str:
    manifest = load_video_manifest(run_dir)
    return Path(manifest["source_path"]).stem


def canonical_common(
    run_dir: Path,
    frame_id: int,
    timestamp_ms: float,
    source_method: str,
    confidence: float,
    valid: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "run_id": run_dir.name,
        "match_id": match_id(run_dir),
        "frame_id": int(frame_id),
        "timestamp_ms": float(timestamp_ms),
        "source_method": source_method,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "valid": bool(valid),
    }


def read_required_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)
