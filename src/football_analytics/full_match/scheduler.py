"""Sequential, resumable chunk scheduling."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .chunking import iter_chunk_frames
from .manifest import (
    CHUNK_MANIFEST_NAME,
    CHUNK_STATUS_PARQUET_NAME,
    atomic_write_json,
    save_chunk_status_parquet,
    save_model,
)
from .resume import mark_chunk_result, should_run_chunk
from .schemas import (
    ChunkManifest,
    ChunkRecord,
    ChunkStatus,
    Fingerprints,
    MatchManifest,
    utc_now_iso,
)

ChunkProcessor = Callable[[Path, ChunkRecord], dict[str, Any]]


def default_chunk_processor(video_path: Path, record: ChunkRecord) -> dict[str, Any]:
    """Infrastructure-only processing: stream-decode and summarize the chunk.

    This intentionally produces no detections; model stages are reported as
    unavailable rather than fabricated.
    """
    frames = 0
    intensity_sum = 0.0
    for _, image in iter_chunk_frames(video_path, record):
        frames += 1
        intensity_sum += float(image.mean())
    return {
        "camera_id": record.camera_id,
        "chunk_index": record.chunk_index,
        "start_seconds": record.start_seconds,
        "end_seconds": record.end_seconds,
        "frames_decoded": frames,
        "mean_intensity": (intensity_sum / frames) if frames else None,
        "model_outputs": None,
        "model_stage_status": "NOT_AVAILABLE",
    }


class ChunkScheduler:
    """Run chunks camera-by-camera with resume, retry, and atomic persistence."""

    def __init__(
        self,
        run_dir: Path,
        match_manifest: MatchManifest,
        chunk_manifest: ChunkManifest,
        fingerprints: Fingerprints,
        retry_limit: int = 3,
        fail_fast: bool = True,
        processor: ChunkProcessor | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.match_manifest = match_manifest
        self.chunk_manifest = chunk_manifest
        self.fingerprints = fingerprints
        self.retry_limit = retry_limit
        self.fail_fast = fail_fast
        self.processor = processor or default_chunk_processor

    def _persist(self) -> None:
        save_model(self.run_dir / CHUNK_MANIFEST_NAME, self.chunk_manifest)
        save_chunk_status_parquet(
            self.run_dir / CHUNK_STATUS_PARQUET_NAME, self.chunk_manifest.records
        )

    def _selected(
        self, from_chunk: int | None, until_chunk: int | None
    ) -> list[ChunkRecord]:
        selected = []
        for record in self.chunk_manifest.records:
            if from_chunk is not None and record.chunk_index < from_chunk:
                continue
            if until_chunk is not None and record.chunk_index > until_chunk:
                continue
            selected.append(record)
        return selected

    def _run_one(self, record: ChunkRecord) -> None:
        camera = self.match_manifest.camera(record.camera_id)
        record.status = ChunkStatus.RUNNING
        record.updated_at = utc_now_iso()
        self._persist()

        started = time.monotonic()
        try:
            payload = self.processor(Path(camera.path), record)
        except Exception as exc:  # noqa: BLE001 - failure is recorded, not hidden
            mark_chunk_result(
                record,
                ok=False,
                fingerprints=self.fingerprints,
                retry_limit=self.retry_limit,
                error=f"{type(exc).__name__}: {exc}",
                wall_seconds=time.monotonic() - started,
            )
            self._persist()
            return

        wall = time.monotonic() - started
        if int(payload.get("frames_decoded") or 0) <= 0:
            record.attempts += 1
            record.status = ChunkStatus.INVALID_INPUT
            record.error = "no frames decodable in chunk range"
            record.wall_seconds = wall
            record.updated_at = utc_now_iso()
            self._persist()
            return

        result_path = (
            self.run_dir
            / "chunks"
            / record.camera_id
            / f"chunk_{record.chunk_index:05d}.json"
        )
        atomic_write_json(result_path, payload)
        mark_chunk_result(
            record,
            ok=True,
            fingerprints=self.fingerprints,
            retry_limit=self.retry_limit,
            wall_seconds=wall,
            result_path=str(result_path.relative_to(self.run_dir)),
        )
        self._persist()

    def run(
        self, from_chunk: int | None = None, until_chunk: int | None = None
    ) -> dict[str, Any]:
        selected = self._selected(from_chunk, until_chunk)
        executed = 0
        skipped = 0
        for record in selected:
            if not should_run_chunk(record, self.fingerprints, self.retry_limit):
                skipped += 1
                continue
            self._run_one(record)
            executed += 1
            if self.fail_fast and record.status in (
                ChunkStatus.FAILED,
                ChunkStatus.INVALID_INPUT,
            ):
                break
        counts = self.chunk_manifest.counts_by_status()
        failed = counts[ChunkStatus.FAILED.value] + counts[ChunkStatus.INVALID_INPUT.value]
        return {
            "selected_chunks": len(selected),
            "executed_chunks": executed,
            "skipped_chunks": skipped,
            "counts": counts,
            "ok": failed == 0 and counts[ChunkStatus.RETRY.value] == 0,
        }
