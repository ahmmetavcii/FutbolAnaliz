"""Recompute derived stages after operator corrections."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from .consolidation import consolidate_run
from .manifest import (
    CHUNK_MANIFEST_NAME,
    RUN_STATE_NAME,
    atomic_write_json,
    load_model,
    save_model,
)
from .schemas import ChunkManifest, RunState, StageStatus, stages_from, utc_now_iso


def _load_corrections(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"corrections must be a mapping: {path}")
    return payload


def recompute_after_manual_correction(
    run_dir: Path,
    corrections_path: Path,
    from_stage: str = "events",
    output_run_dir: Path | None = None,
    in_place: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply operator corrections and invalidate/recompute derived stages.

    Only infrastructure recomputation (consolidation) is performed here; model
    stages that are unavailable stay INVALIDATED rather than being fabricated.
    """
    run_dir = Path(run_dir)
    if bool(output_run_dir) == bool(in_place):
        raise ValueError("choose exactly one of output_run_dir or in_place")

    corrections = _load_corrections(Path(corrections_path))
    invalidated_stages = list(stages_from(from_stage))

    if dry_run:
        return {
            "status": "PASS",
            "mode": "dry_run",
            "run_dir": str(run_dir),
            "target_run_dir": str(output_run_dir or run_dir),
            "from_stage": from_stage,
            "stages_to_invalidate": invalidated_stages,
            "correction_keys": sorted(corrections),
        }

    if output_run_dir is not None:
        target = Path(output_run_dir)
        if target.exists():
            raise ValueError(f"output run directory already exists: {target}")
        shutil.copytree(run_dir, target)
    else:
        target = run_dir

    state: RunState = load_model(target / RUN_STATE_NAME, RunState)
    chunk_manifest: ChunkManifest = load_model(target / CHUNK_MANIFEST_NAME, ChunkManifest)

    stamp = utc_now_iso().replace(":", "").replace("+", "Z")
    correction_record = target / "corrections" / f"correction_{stamp}.json"
    atomic_write_json(
        correction_record,
        {
            "applied_at": utc_now_iso(),
            "source": str(corrections_path),
            "from_stage": from_stage,
            "corrections": corrections,
        },
    )

    for name in invalidated_stages:
        state.set_stage(name, StageStatus.INVALIDATED, reason="manual correction")

    # Consolidation is infrastructure work we can honestly redo right now.
    summary = consolidate_run(target, chunk_manifest)
    state.set_stage("consolidation", StageStatus.PASS, reason="recomputed after correction")
    for name in invalidated_stages:
        if name in ("consolidation",):
            continue
        record = state.stage(name)
        if record.status == StageStatus.INVALIDATED:
            record.reason = "manual correction; model stage recompute not available"
    save_model(target / RUN_STATE_NAME, state)

    return {
        "status": "PASS",
        "run_dir": str(run_dir),
        "target_run_dir": str(target),
        "from_stage": from_stage,
        "invalidated_stages": invalidated_stages,
        "correction_record": str(correction_record.relative_to(target)),
        "consolidation": summary,
    }
