"""Fingerprints, resume decisions, retry accounting, and invalidation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from football_analytics.utils.hashing import sha256_file

from .schemas import (
    ChunkManifest,
    ChunkRecord,
    ChunkStatus,
    Fingerprints,
    MatchManifest,
    utc_now_iso,
)

DEFAULT_RETRY_LIMIT = 3


def _canonical_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_fingerprint(config: dict[str, Any]) -> str:
    return _canonical_hash(config)


def model_fingerprint(model_files: dict[str, Path] | None = None) -> str:
    """Fingerprint model weights by content hash; stable marker when absent."""
    if not model_files:
        return _canonical_hash({"models": "none"})
    hashes = {
        name: (sha256_file(Path(path)) if Path(path).is_file() else "missing")
        for name, path in sorted(model_files.items())
    }
    return _canonical_hash(hashes)


def input_fingerprints(manifest: MatchManifest) -> dict[str, str]:
    return {camera.camera_id: camera.sha256 for camera in manifest.cameras}


def build_fingerprints(
    config: dict[str, Any],
    manifest: MatchManifest,
    model_files: dict[str, Path] | None = None,
) -> Fingerprints:
    return Fingerprints(
        config=config_fingerprint(config),
        model=model_fingerprint(model_files),
        inputs=input_fingerprints(manifest),
    )


def should_run_chunk(
    record: ChunkRecord,
    fingerprints: Fingerprints,
    retry_limit: int = DEFAULT_RETRY_LIMIT,
) -> bool:
    """Resume policy: skip verified PASS work, retry failures up to the limit."""
    expected = fingerprints.combined(record.camera_id)
    if record.status == ChunkStatus.PASS and record.fingerprint == expected:
        return False
    if record.status == ChunkStatus.SKIPPED:
        return False
    if record.status == ChunkStatus.INVALID_INPUT:
        return False
    if (
        record.status in (ChunkStatus.FAILED, ChunkStatus.RETRY)
        and record.attempts >= retry_limit
    ):
        return False
    return True


def invalidate_stale_chunks(
    manifest: ChunkManifest, fingerprints: Fingerprints
) -> list[ChunkRecord]:
    """Mark completed chunks INVALIDATED when config/model/input identity changed."""
    invalidated: list[ChunkRecord] = []
    for record in manifest.records:
        if record.status not in (ChunkStatus.PASS, ChunkStatus.FAILED, ChunkStatus.RETRY):
            continue
        if record.fingerprint == fingerprints.combined(record.camera_id):
            continue
        record.status = ChunkStatus.INVALIDATED
        record.attempts = 0
        record.error = "fingerprint mismatch (config/model/input changed)"
        record.updated_at = utc_now_iso()
        invalidated.append(record)
    return invalidated


def mark_chunk_result(
    record: ChunkRecord,
    ok: bool,
    fingerprints: Fingerprints,
    retry_limit: int = DEFAULT_RETRY_LIMIT,
    error: str | None = None,
    wall_seconds: float | None = None,
    result_path: str | None = None,
) -> ChunkRecord:
    record.attempts += 1
    record.wall_seconds = wall_seconds
    record.updated_at = utc_now_iso()
    if ok:
        record.status = ChunkStatus.PASS
        record.fingerprint = fingerprints.combined(record.camera_id)
        record.error = None
        record.result_path = result_path
    else:
        record.error = error or "chunk processing failed"
        record.fingerprint = None
        record.status = (
            ChunkStatus.RETRY if record.attempts < retry_limit else ChunkStatus.FAILED
        )
    return record
