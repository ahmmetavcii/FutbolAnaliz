"""Run progress reporting derived from chunk and stage manifests."""

from __future__ import annotations

from .schemas import ChunkManifest, ChunkStatus, RunState


def chunk_progress(manifest: ChunkManifest) -> dict[str, object]:
    counts = manifest.counts_by_status()
    total = len(manifest.records)
    done = counts[ChunkStatus.PASS.value] + counts[ChunkStatus.SKIPPED.value]
    completed_wall = [
        record.wall_seconds
        for record in manifest.records
        if record.status == ChunkStatus.PASS and record.wall_seconds
    ]
    remaining = (
        counts[ChunkStatus.PENDING.value]
        + counts[ChunkStatus.RUNNING.value]
        + counts[ChunkStatus.RETRY.value]
        + counts[ChunkStatus.INVALIDATED.value]
    )
    eta_seconds = None
    if completed_wall:
        eta_seconds = (sum(completed_wall) / len(completed_wall)) * remaining
    return {
        "total_chunks": total,
        "completed_chunks": done,
        "remaining_chunks": remaining,
        "percent": round(100.0 * done / total, 2) if total else 100.0,
        "counts": counts,
        "eta_seconds": eta_seconds,
    }


def run_progress(state: RunState, manifest: ChunkManifest | None = None) -> dict[str, object]:
    report: dict[str, object] = {
        "match_id": state.match_id,
        "run_dir": state.run_dir,
        "stages": {record.name: record.status.value for record in state.stages},
    }
    if manifest is not None:
        report["chunks"] = chunk_progress(manifest)
    return report
