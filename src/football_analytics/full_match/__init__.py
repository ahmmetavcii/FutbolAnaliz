"""Resumable full-match processing: public entry points.

The functions in this module orchestrate real infrastructure work (probing,
chunking, scheduling, consolidation, validation). Heavy model stages are not
bundled here; when they are unavailable the run records them as SKIPPED /
NOT_AVAILABLE instead of fabricating detections.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from football_analytics.utils.hashing import sha256_file
from football_analytics.video.ffprobe import probe_video

from .chunking import iter_chunk_frames, plan_chunks, validate_chunk_seconds
from .consolidation import consolidate_run
from .health import check_disk_guard, estimate_run_bytes, free_disk_bytes
from .manifest import (
    CHUNK_MANIFEST_NAME,
    MATCH_MANIFEST_NAME,
    RUN_STATE_NAME,
    atomic_write_json,
    atomic_write_parquet,
    load_model,
    save_model,
)
from .progress import chunk_progress, run_progress
from .recompute import recompute_after_manual_correction
from .resume import (
    build_fingerprints,
    invalidate_stale_chunks,
    mark_chunk_result,
    should_run_chunk,
)
from .scheduler import ChunkScheduler, default_chunk_processor
from .schemas import (
    ALLOWED_CAMERA_COUNTS,
    DEFAULT_CHUNK_SECONDS,
    MAX_CHUNK_SECONDS,
    MIN_CHUNK_SECONDS,
    CameraRole,
    CameraSpec,
    ChunkManifest,
    ChunkRecord,
    ChunkStatus,
    Fingerprints,
    MatchManifest,
    RunState,
    StageStatus,
    default_stage_records,
    stages_from,
    utc_now_iso,
)
from .validation import validate_full_match_run
from .video_probe import ProbeError, probe_camera

__all__ = [
    "prepare_full_match",
    "synchronize_cameras",
    "sync_cameras",
    "calibrate_cameras",
    "calibrate_run",
    "run_full_match",
    "orchestrate_full_match",
    "resume_full_match",
    "validate_full_match_run",
    "recompute_after_manual_correction",
    # Schemas and building blocks used by tests and adapters.
    "MatchManifest",
    "CameraSpec",
    "CameraRole",
    "ChunkManifest",
    "ChunkRecord",
    "ChunkStatus",
    "StageStatus",
    "RunState",
    "Fingerprints",
    "ChunkScheduler",
    "probe_camera",
    "plan_chunks",
    "iter_chunk_frames",
    "chunk_progress",
    "run_progress",
    "consolidate_run",
    "check_disk_guard",
    "estimate_run_bytes",
    "free_disk_bytes",
    "atomic_write_json",
    "atomic_write_parquet",
    "build_fingerprints",
    "should_run_chunk",
    "invalidate_stale_chunks",
    "mark_chunk_result",
    "default_chunk_processor",
    "validate_chunk_seconds",
    "DEFAULT_CHUNK_SECONDS",
    "MIN_CHUNK_SECONDS",
    "MAX_CHUNK_SECONDS",
    "ALLOWED_CAMERA_COUNTS",
    "ProbeError",
]

SYNC_ARTIFACT = "sync.json"
CALIBRATION_DIR = "calibration"

_DEFAULT_ROLES: dict[int, tuple[CameraRole, ...]] = {
    1: (CameraRole.TACTICAL_FULL,),
    2: (CameraRole.TACTICAL_LEFT, CameraRole.TACTICAL_RIGHT),
    4: (
        CameraRole.TACTICAL_LEFT,
        CameraRole.TACTICAL_RIGHT,
        CameraRole.GOAL_LEFT,
        CameraRole.GOAL_RIGHT,
    ),
}


def _resolve_roles(count: int, config: dict[str, Any] | None) -> list[CameraRole]:
    configured = ((config or {}).get("cameras") or {}).get("roles")
    if configured:
        if len(configured) != count:
            raise ValueError(
                f"config lists {len(configured)} camera roles for {count} inputs"
            )
        return [CameraRole(role) for role in configured]
    return list(_DEFAULT_ROLES[count])


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def prepare_full_match(
    inputs: Sequence[Path] | None = None,
    camera_inputs: Sequence[Path] | None = None,
    camera_ids: Sequence[str] | None = None,
    config: dict[str, Any] | None = None,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    match_id: str | None = None,
    force: bool = False,
    chunk_seconds: float | None = None,
) -> dict[str, Any]:
    """Probe camera inputs and write resumable match + chunk manifests."""
    videos = [Path(item) for item in (inputs or camera_inputs or [])]
    if not videos:
        raise ValueError("at least one camera input is required")
    if len(videos) not in ALLOWED_CAMERA_COUNTS:
        raise ValueError(
            f"exactly {ALLOWED_CAMERA_COUNTS} cameras are supported, got {len(videos)}"
        )
    if output_dir is None or match_id is None:
        raise ValueError("output_dir and match_id are required")

    output_dir = Path(output_dir)
    manifest_path = output_dir / MATCH_MANIFEST_NAME
    if manifest_path.exists() and not force:
        raise ValueError(
            f"preparation already exists at {output_dir}; pass force=True to replace"
        )

    config = config or {}
    full_match_cfg = config.get("full_match") or {}
    seconds = validate_chunk_seconds(
        chunk_seconds if chunk_seconds is not None else full_match_cfg.get("chunk_seconds")
    )
    ids = list(camera_ids or [f"camera_{index + 1}" for index in range(len(videos))])
    if len(ids) != len(videos):
        raise ValueError("camera_ids must match the number of inputs")
    roles = _resolve_roles(len(videos), config)

    cameras: list[CameraSpec] = []
    for camera_id, role, path in zip(ids, roles, videos):
        probe = probe_camera(path)
        if not probe.decodable:
            raise ProbeError(f"camera {camera_id} failed first/middle/last frame checks")
        cameras.append(
            CameraSpec(
                camera_id=camera_id,
                role=role,
                path=str(Path(path).resolve()),
                sha256=sha256_file(Path(path)),
                probe=probe,
            )
        )

    manifest = MatchManifest(
        match_id=match_id,
        chunk_seconds=seconds,
        profile=full_match_cfg.get("profile"),
        cameras=cameras,
    )
    disk = check_disk_guard(output_dir, estimate_run_bytes(manifest))

    records: list[ChunkRecord] = []
    for camera in manifest.cameras:
        records.extend(plan_chunks(camera.camera_id, camera.probe, seconds))
    chunk_manifest = ChunkManifest(
        match_id=match_id, chunk_seconds=seconds, records=records
    )

    save_model(manifest_path, manifest)
    save_model(output_dir / CHUNK_MANIFEST_NAME, chunk_manifest)
    report = {
        "status": "PASS",
        "match_id": match_id,
        "prepared_dir": str(output_dir),
        "config_path": str(config_path) if config_path else None,
        "chunk_seconds": seconds,
        "cameras": [
            {
                "camera_id": camera.camera_id,
                "role": camera.role.value,
                "duration_seconds": camera.probe.duration_seconds,
                "sha256": camera.sha256,
            }
            for camera in manifest.cameras
        ],
        "total_chunks": len(records),
        "disk_guard": disk,
    }
    atomic_write_json(output_dir / "prepare_report.json", report)
    return report


prepare_run = prepare_full_match


# ---------------------------------------------------------------------------
# synchronization
# ---------------------------------------------------------------------------


def _extract_audio_envelope(
    path: Path, seconds: float, rate: int = 8000
) -> np.ndarray | None:
    """Decode the first `seconds` of mono audio to PCM without loading video."""
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-t",
        str(seconds),
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    if len(completed.stdout) < 2 * rate:  # under one second of audio
        return None
    signal = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float64)
    return np.abs(signal)


def _audio_offset_seconds(
    reference: np.ndarray, other: np.ndarray, rate: int, max_offset_seconds: float
) -> float:
    max_lag = int(max_offset_seconds * rate)
    length = min(len(reference), len(other))
    ref = reference[:length] - reference[:length].mean()
    sig = other[:length] - other[:length].mean()
    best_lag, best_score = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1, max(1, rate // 100)):
        if lag >= 0:
            a, b = ref[lag:], sig[: length - lag]
        else:
            a, b = ref[: length + lag], sig[-lag:]
        if len(a) < rate:
            continue
        score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        if score > best_score:
            best_score, best_lag = score, lag
    return best_lag / rate


def _load_offsets_file(path: Path) -> dict[str, float]:
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)
    offsets = payload.get("offsets", payload) if isinstance(payload, dict) else None
    if not isinstance(offsets, dict):
        raise ValueError(f"offsets file must map camera_id -> seconds: {path}")
    return {str(key): float(value) for key, value in offsets.items()}


def synchronize_cameras(
    prepared_dir: Path,
    method: str = "audio",
    reference_camera: str | None = None,
    offsets_path: Path | None = None,
    output_path: Path | None = None,
    max_offset_seconds: float = 30.0,
) -> dict[str, Any]:
    """Estimate or apply per-camera time offsets relative to a reference camera."""
    prepared_dir = Path(prepared_dir)
    manifest: MatchManifest = load_model(
        prepared_dir / MATCH_MANIFEST_NAME, MatchManifest
    )
    reference = reference_camera or manifest.cameras[0].camera_id
    manifest.camera(reference)  # raises KeyError for unknown reference
    target = Path(output_path) if output_path else prepared_dir / SYNC_ARTIFACT

    offsets: dict[str, float] = {reference: 0.0}
    failures: list[str] = []

    if method == "manual":
        if offsets_path is None:
            raise ValueError("offsets_path is required for manual synchronization")
        provided = _load_offsets_file(Path(offsets_path))
        for camera in manifest.cameras:
            if camera.camera_id not in provided:
                failures.append(f"no manual offset for camera {camera.camera_id}")
                continue
            value = provided[camera.camera_id]
            if abs(value) > max_offset_seconds:
                failures.append(
                    f"offset {value}s for {camera.camera_id} exceeds "
                    f"max_offset_seconds={max_offset_seconds}"
                )
                continue
            offsets[camera.camera_id] = value
    elif method == "timecode":
        times: dict[str, float] = {}
        for camera in manifest.cameras:
            probe = probe_video(Path(camera.path))
            start = float((probe.get("format") or {}).get("start_time") or 0.0)
            times[camera.camera_id] = start
        for camera in manifest.cameras:
            offsets[camera.camera_id] = times[camera.camera_id] - times[reference]
    elif method == "audio":
        if len(manifest.cameras) > 1:
            window = min(120.0, max_offset_seconds * 4)
            rate = 8000
            ref_signal = _extract_audio_envelope(
                Path(manifest.camera(reference).path), window, rate
            )
            if ref_signal is None:
                failures.append(f"reference camera {reference} has no decodable audio")
            else:
                for camera in manifest.cameras:
                    if camera.camera_id == reference:
                        continue
                    signal = _extract_audio_envelope(Path(camera.path), window, rate)
                    if signal is None:
                        failures.append(
                            f"camera {camera.camera_id} has no decodable audio"
                        )
                        continue
                    offsets[camera.camera_id] = _audio_offset_seconds(
                        ref_signal, signal, rate, max_offset_seconds
                    )
    else:
        raise ValueError(f"unsupported synchronization method: {method}")

    status = "PASS" if not failures and len(offsets) == len(manifest.cameras) else "FAILED"
    result = {
        "status": status,
        "method": method,
        "reference_camera": reference,
        "offsets_seconds": offsets,
        "max_offset_seconds": max_offset_seconds,
        "failures": failures,
        "generated_at": utc_now_iso(),
        "artifact": str(target),
    }
    if status == "PASS":
        atomic_write_json(target, result)
    return result


sync_cameras = synchronize_cameras


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def calibrate_cameras(
    prepared_dir: Path,
    provider: str = "auto",
    camera_ids: Sequence[str] | None = None,
    manual_calibration: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Apply verified manual calibration; report model providers honestly.

    Automatic providers (sn_calibration, pnlcalib) require model runtimes that
    this package does not bundle, so they are reported as NOT_AVAILABLE rather
    than producing invented homographies.
    """
    prepared_dir = Path(prepared_dir)
    manifest: MatchManifest = load_model(
        prepared_dir / MATCH_MANIFEST_NAME, MatchManifest
    )
    target_dir = Path(output_dir) if output_dir else prepared_dir / CALIBRATION_DIR
    selected = list(camera_ids) if camera_ids else [c.camera_id for c in manifest.cameras]
    for camera_id in selected:
        manifest.camera(camera_id)  # raises KeyError for unknown ids

    per_camera: dict[str, dict[str, Any]] = {}

    if provider == "manual" or (provider == "auto" and manual_calibration is not None):
        if manual_calibration is None:
            raise ValueError("manual_calibration file is required for manual provider")
        payload = json.loads(Path(manual_calibration).read_text(encoding="utf-8"))
        entries = payload.get("cameras", payload)
        if not isinstance(entries, dict):
            raise ValueError("manual calibration must map camera_id -> calibration")
        for camera_id in selected:
            if camera_id not in entries:
                per_camera[camera_id] = {
                    "status": "FAILED",
                    "reason": "no manual calibration entry",
                }
                continue
            artifact = target_dir / f"{camera_id}.json"
            if artifact.exists() and not force:
                per_camera[camera_id] = {
                    "status": "PASS",
                    "artifact": str(artifact),
                    "reason": "existing calibration kept (force=False)",
                }
                continue
            atomic_write_json(
                artifact,
                {
                    "camera_id": camera_id,
                    "provider": "manual",
                    "calibration": entries[camera_id],
                    "generated_at": utc_now_iso(),
                },
            )
            per_camera[camera_id] = {"status": "PASS", "artifact": str(artifact)}
    elif provider == "auto":
        for camera_id in selected:
            artifact = target_dir / f"{camera_id}.json"
            if artifact.is_file():
                per_camera[camera_id] = {"status": "PASS", "artifact": str(artifact)}
            else:
                per_camera[camera_id] = {
                    "status": "NOT_AVAILABLE",
                    "reason": "no existing calibration and no model provider bundled",
                }
    elif provider in ("metadata", "sn_calibration", "pnlcalib"):
        for camera_id in selected:
            per_camera[camera_id] = {
                "status": "NOT_AVAILABLE",
                "reason": f"provider {provider!r} requires an external runtime "
                "that is not bundled with football_analytics.full_match",
            }
    else:
        raise ValueError(f"unsupported calibration provider: {provider}")

    statuses = {entry["status"] for entry in per_camera.values()}
    overall = "PASS" if statuses == {"PASS"} else ("FAILED" if "FAILED" in statuses else "NOT_AVAILABLE")
    return {
        "status": overall,
        "provider": provider,
        "calibration_dir": str(target_dir),
        "cameras": per_camera,
    }


calibrate_run = calibrate_cameras


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _resolve_adapter(chunk_pipeline_config: Path) -> Any:
    """Load the opt-in per-chunk pipeline adapter from its YAML descriptor."""
    payload = yaml.safe_load(Path(chunk_pipeline_config).read_text(encoding="utf-8")) or {}
    dotted = (payload.get("adapter") or {}).get("callable") if isinstance(payload, dict) else None
    if not dotted:
        raise ValueError(
            f"chunk pipeline config {chunk_pipeline_config} must define adapter.callable"
        )
    module_name, _, attribute = str(dotted).rpartition(".")
    if not module_name:
        raise ValueError(f"adapter.callable must be a dotted path, got {dotted!r}")
    import importlib

    module = importlib.import_module(module_name)
    adapter = getattr(module, attribute, None)
    if not callable(adapter):
        raise ValueError(f"adapter {dotted} is not callable")
    return adapter


def run_full_match(
    prepared_dir: Path,
    run_dir: Path,
    config: dict[str, Any] | None = None,
    config_path: Path | None = None,
    synchronize: bool = True,
    calibrate: bool = True,
    from_chunk: int | None = None,
    until_chunk: int | None = None,
    chunk_pipeline_config: Path | None = None,
    retry_limit: int = 3,
    model_files: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Orchestrate a prepared full match with resume, retry, and invalidation."""
    prepared_dir = Path(prepared_dir)
    run_dir = Path(run_dir)
    config = config or {}
    full_match_cfg = config.get("full_match") or {}

    match_manifest: MatchManifest = load_model(
        prepared_dir / MATCH_MANIFEST_NAME, MatchManifest
    )
    disk = check_disk_guard(run_dir, estimate_run_bytes(match_manifest))
    fingerprints = build_fingerprints(config, match_manifest, model_files)

    state_path = run_dir / RUN_STATE_NAME
    chunk_path = run_dir / CHUNK_MANIFEST_NAME
    invalidated = 0
    if state_path.is_file() and chunk_path.is_file():
        state: RunState = load_model(state_path, RunState)
        chunk_manifest: ChunkManifest = load_model(chunk_path, ChunkManifest)
        state.fingerprints = fingerprints
        invalidated = len(invalidate_stale_chunks(chunk_manifest, fingerprints))
        if invalidated:
            for name in stages_from("chunks"):
                state.set_stage(
                    name, StageStatus.INVALIDATED, reason="config/model fingerprint changed"
                )
    else:
        chunk_manifest = load_model(prepared_dir / CHUNK_MANIFEST_NAME, ChunkManifest)
        state = RunState(
            match_id=match_manifest.match_id,
            prepared_dir=str(prepared_dir),
            run_dir=str(run_dir),
            retry_limit=retry_limit,
            fail_fast=bool(full_match_cfg.get("fail_fast", True)),
            chunk_pipeline_adapter=str(chunk_pipeline_config) if chunk_pipeline_config else None,
            fingerprints=fingerprints,
            stages=default_stage_records(),
        )

    save_model(run_dir / MATCH_MANIFEST_NAME, match_manifest)
    state.set_stage("prepare", StageStatus.PASS, reason=f"prepared at {prepared_dir}")

    # --- sync stage -------------------------------------------------------
    if not synchronize:
        state.set_stage("sync", StageStatus.SKIPPED, reason="disabled by caller")
    elif len(match_manifest.cameras) == 1:
        sync_result = {
            "status": "PASS",
            "method": "trivial",
            "reference_camera": match_manifest.cameras[0].camera_id,
            "offsets_seconds": {match_manifest.cameras[0].camera_id: 0.0},
            "generated_at": utc_now_iso(),
        }
        atomic_write_json(run_dir / SYNC_ARTIFACT, sync_result)
        state.set_stage("sync", StageStatus.PASS, reason="single camera, zero offset")
    elif (prepared_dir / SYNC_ARTIFACT).is_file():
        payload = json.loads((prepared_dir / SYNC_ARTIFACT).read_text(encoding="utf-8"))
        atomic_write_json(run_dir / SYNC_ARTIFACT, payload)
        state.set_stage("sync", StageStatus.PASS, reason="sync artifact from preparation")
    else:
        state.set_stage(
            "sync",
            StageStatus.SKIPPED,
            reason="multi-camera sync artifact missing; run sync_cameras.py",
        )

    # --- calibration stage ------------------------------------------------
    if not calibrate:
        state.set_stage("calibration", StageStatus.SKIPPED, reason="disabled by caller")
    else:
        calibration_dir = prepared_dir / CALIBRATION_DIR
        missing = [
            camera.camera_id
            for camera in match_manifest.cameras
            if not (calibration_dir / f"{camera.camera_id}.json").is_file()
        ]
        if not missing:
            state.set_stage(
                "calibration", StageStatus.PASS, reason=f"artifacts in {calibration_dir}"
            )
        else:
            state.set_stage(
                "calibration",
                StageStatus.SKIPPED,
                reason=f"no calibration for cameras {missing}; "
                "no automatic provider is bundled",
            )

    # --- chunk processing stage --------------------------------------------
    processor = None
    if chunk_pipeline_config is not None:
        processor = _resolve_adapter(Path(chunk_pipeline_config))
        state.chunk_pipeline_adapter = str(chunk_pipeline_config)
    state.set_stage("chunks", StageStatus.RUNNING)
    save_model(state_path, state)

    scheduler = ChunkScheduler(
        run_dir=run_dir,
        match_manifest=match_manifest,
        chunk_manifest=chunk_manifest,
        fingerprints=fingerprints,
        retry_limit=retry_limit,
        fail_fast=state.fail_fast,
        processor=processor,
    )
    chunk_result = scheduler.run(from_chunk=from_chunk, until_chunk=until_chunk)
    state.set_stage(
        "chunks",
        StageStatus.PASS if chunk_result["ok"] else StageStatus.FAILED,
        reason=None if chunk_result["ok"] else "one or more chunks failed",
    )

    # --- model stages (honest reporting) ------------------------------------
    model_reason = (
        "delegated to chunk pipeline adapter"
        if processor is not None
        else "model stage not available; infrastructure-only run (no fabricated outputs)"
    )
    for name in ("detection", "tracking", "events"):
        state.set_stage(
            name,
            StageStatus.PASS if processor is not None and chunk_result["ok"] else StageStatus.SKIPPED,
            reason=model_reason,
        )

    # --- consolidation -------------------------------------------------------
    summary = consolidate_run(run_dir, chunk_manifest)
    state.set_stage("consolidation", StageStatus.PASS)
    state.set_stage(
        "export", StageStatus.SKIPPED, reason="run export_full_match_results.py"
    )
    save_model(state_path, state)

    return {
        "status": "PASS" if chunk_result["ok"] else "FAILED",
        "match_id": match_manifest.match_id,
        "run_dir": str(run_dir),
        "prepared_dir": str(prepared_dir),
        "config_path": str(config_path) if config_path else None,
        "invalidated_chunks": invalidated,
        "chunks": chunk_result,
        "progress": run_progress(state, chunk_manifest),
        "consolidation": summary,
        "disk_guard": disk,
    }


orchestrate_full_match = run_full_match


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


def resume_full_match(
    run_dir: Path,
    rerun_from_stage: str | None = None,
    from_chunk: int | None = None,
    repair_manifests: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Resume a run: skip PASS chunks, retry failures, honor invalidations."""
    run_dir = Path(run_dir)
    state: RunState = load_model(run_dir / RUN_STATE_NAME, RunState)
    match_manifest: MatchManifest = load_model(
        run_dir / MATCH_MANIFEST_NAME, MatchManifest
    )
    chunk_manifest: ChunkManifest = load_model(
        run_dir / CHUNK_MANIFEST_NAME, ChunkManifest
    )
    fingerprints = state.fingerprints

    repaired = 0
    if repair_manifests:
        for record in chunk_manifest.records:
            if record.status == ChunkStatus.RUNNING:
                record.status = ChunkStatus.RETRY
                record.error = "interrupted while RUNNING; repaired for resume"
                record.updated_at = utc_now_iso()
                repaired += 1

    invalidated_stages: list[str] = []
    if rerun_from_stage:
        invalidated_stages = list(stages_from(rerun_from_stage))
        for name in invalidated_stages:
            state.set_stage(name, StageStatus.INVALIDATED, reason="rerun requested")
        if "chunks" in invalidated_stages:
            for record in chunk_manifest.records:
                if record.status in (
                    ChunkStatus.PASS,
                    ChunkStatus.FAILED,
                    ChunkStatus.RETRY,
                ):
                    record.status = ChunkStatus.INVALIDATED
                    record.attempts = 0
                    record.updated_at = utc_now_iso()

    pending = [
        record
        for record in chunk_manifest.records
        if (from_chunk is None or record.chunk_index >= from_chunk)
        and should_run_chunk(record, fingerprints, state.retry_limit)
    ]

    if dry_run:
        return {
            "status": "PASS",
            "mode": "dry_run",
            "run_dir": str(run_dir),
            "repaired_chunks": repaired,
            "invalidated_stages": invalidated_stages,
            "chunks_to_run": len(pending),
            "counts": chunk_manifest.counts_by_status(),
        }

    save_model(run_dir / CHUNK_MANIFEST_NAME, chunk_manifest)
    save_model(run_dir / RUN_STATE_NAME, state)

    scheduler = ChunkScheduler(
        run_dir=run_dir,
        match_manifest=match_manifest,
        chunk_manifest=chunk_manifest,
        fingerprints=fingerprints,
        retry_limit=state.retry_limit,
        fail_fast=state.fail_fast,
    )
    chunk_result = scheduler.run(from_chunk=from_chunk)
    state.set_stage(
        "chunks",
        StageStatus.PASS if chunk_result["ok"] else StageStatus.FAILED,
        reason=None if chunk_result["ok"] else "one or more chunks failed",
    )
    summary = consolidate_run(run_dir, chunk_manifest)
    state.set_stage("consolidation", StageStatus.PASS)
    save_model(run_dir / RUN_STATE_NAME, state)

    return {
        "status": "PASS" if chunk_result["ok"] else "FAILED",
        "run_dir": str(run_dir),
        "repaired_chunks": repaired,
        "invalidated_stages": invalidated_stages,
        "chunks": chunk_result,
        "progress": run_progress(state, chunk_manifest),
        "consolidation": summary,
    }


resume_run = resume_full_match
