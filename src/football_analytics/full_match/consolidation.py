"""Consolidate per-chunk artifacts into match-level summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .manifest import atomic_write_json, atomic_write_parquet
from .schemas import ChunkManifest, ChunkStatus, utc_now_iso

CONSOLIDATED_DIR = "consolidated"
CHUNK_SUMMARY_PARQUET = "chunk_summary.parquet"
RUN_SUMMARY_JSON = "summary.json"


def _load_chunk_payload(run_dir: Path, result_path: str | None) -> dict[str, Any]:
    if not result_path:
        return {}
    path = run_dir / result_path
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def consolidate_run(run_dir: Path, chunk_manifest: ChunkManifest) -> dict[str, Any]:
    """Build the chunk summary table and match-level summary atomically."""
    run_dir = Path(run_dir)
    rows: list[dict[str, Any]] = []
    for record in chunk_manifest.records:
        payload = _load_chunk_payload(run_dir, record.result_path)
        rows.append(
            {
                "camera_id": record.camera_id,
                "chunk_index": record.chunk_index,
                "status": record.status.value,
                "start_seconds": record.start_seconds,
                "end_seconds": record.end_seconds,
                "frame_start": record.frame_start,
                "frame_end": record.frame_end,
                "attempts": record.attempts,
                "wall_seconds": record.wall_seconds,
                "frames_decoded": payload.get("frames_decoded"),
                "mean_intensity": payload.get("mean_intensity"),
                "model_stage_status": payload.get("model_stage_status"),
            }
        )
    frame = pd.DataFrame(rows)

    out_dir = run_dir / CONSOLIDATED_DIR
    parquet_path = out_dir / CHUNK_SUMMARY_PARQUET
    atomic_write_parquet(parquet_path, frame)

    corrections_dir = run_dir / "corrections"
    corrections = (
        sorted(item.name for item in corrections_dir.glob("*.json"))
        if corrections_dir.is_dir()
        else []
    )

    per_camera: dict[str, dict[str, Any]] = {}
    for record in chunk_manifest.records:
        camera = per_camera.setdefault(
            record.camera_id,
            {"chunks": 0, "passed": 0, "duration_seconds": 0.0},
        )
        camera["chunks"] += 1
        if record.status == ChunkStatus.PASS:
            camera["passed"] += 1
            camera["duration_seconds"] += record.end_seconds - record.start_seconds

    summary = {
        "generated_at": utc_now_iso(),
        "match_id": chunk_manifest.match_id,
        "chunk_seconds": chunk_manifest.chunk_seconds,
        "counts": chunk_manifest.counts_by_status(),
        "per_camera": per_camera,
        "corrections_applied": corrections,
        "chunk_summary_parquet": str(parquet_path.relative_to(run_dir)),
    }
    atomic_write_json(out_dir / RUN_SUMMARY_JSON, summary)
    return summary
