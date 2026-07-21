"""Validate manifests, checksums, artifacts, and completion state of a run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from football_analytics.utils.hashing import sha256_file

from .manifest import (
    CHUNK_MANIFEST_NAME,
    MATCH_MANIFEST_NAME,
    RUN_STATE_NAME,
    atomic_write_json,
    load_model,
)
from .schemas import ChunkManifest, ChunkStatus, MatchManifest, RunState, utc_now_iso
from .video_probe import ProbeError, probe_camera


def _load_run_documents(
    run_dir: Path, errors: list[str]
) -> tuple[RunState | None, MatchManifest | None, ChunkManifest | None]:
    documents: list[Any] = []
    for name, model_type in (
        (RUN_STATE_NAME, RunState),
        (MATCH_MANIFEST_NAME, MatchManifest),
        (CHUNK_MANIFEST_NAME, ChunkManifest),
    ):
        path = run_dir / name
        if not path.is_file():
            errors.append(f"missing manifest: {name}")
            documents.append(None)
            continue
        try:
            documents.append(load_model(path, model_type))
        except (ValueError, OSError) as exc:
            errors.append(f"invalid {name}: {exc}")
            documents.append(None)
    return documents[0], documents[1], documents[2]


def validate_full_match_run(
    run_dir: Path,
    verify_checksums: bool = True,
    open_media: bool = False,
    strict: bool = False,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Validate schema conformance, artifact presence, and input integrity."""
    run_dir = Path(run_dir)
    errors: list[str] = []
    warnings: list[str] = []

    state, match_manifest, chunk_manifest = _load_run_documents(run_dir, errors)

    if match_manifest is not None:
        for camera in match_manifest.cameras:
            source = Path(camera.path)
            if not source.is_file():
                warnings.append(f"camera source missing on disk: {camera.path}")
                continue
            if verify_checksums:
                actual = sha256_file(source)
                if actual != camera.sha256:
                    errors.append(
                        f"checksum mismatch for camera {camera.camera_id}: "
                        f"manifest={camera.sha256[:12]}..., actual={actual[:12]}..."
                    )
            if open_media:
                try:
                    probe = probe_camera(source)
                    if not probe.decodable:
                        errors.append(f"camera {camera.camera_id} failed frame decoding")
                except ProbeError as exc:
                    errors.append(f"camera {camera.camera_id} probe failed: {exc}")

    incomplete = 0
    if chunk_manifest is not None:
        for record in chunk_manifest.records:
            if record.status == ChunkStatus.PASS:
                if not record.result_path:
                    errors.append(
                        f"chunk {record.camera_id}/{record.chunk_index} is PASS "
                        "without a result artifact"
                    )
                elif not (run_dir / record.result_path).is_file():
                    errors.append(
                        f"chunk artifact missing: {record.result_path}"
                    )
            elif record.status in (ChunkStatus.SKIPPED, ChunkStatus.INVALID_INPUT):
                warnings.append(
                    f"chunk {record.camera_id}/{record.chunk_index} "
                    f"is {record.status.value}"
                )
            else:
                incomplete += 1
        if incomplete:
            warnings.append(f"{incomplete} chunk(s) are not complete yet")

    if state is not None:
        for stage in state.stages:
            if stage.status.value in ("FAILED",):
                errors.append(f"stage {stage.name} is FAILED: {stage.reason or ''}".strip())
            elif stage.status.value in ("SKIPPED", "INVALIDATED", "PENDING", "RUNNING"):
                warnings.append(f"stage {stage.name} is {stage.status.value}")

    failed = bool(errors) or (strict and bool(warnings))
    report = {
        "status": "FAILED" if failed else "PASS",
        "generated_at": utc_now_iso(),
        "run_dir": str(run_dir),
        "strict": strict,
        "checksums_verified": verify_checksums,
        "media_opened": open_media,
        "errors": errors,
        "warnings": warnings,
    }
    if report_path is not None:
        atomic_write_json(Path(report_path), report)
        report["report_path"] = str(report_path)
    return report
