"""Disk guard and resource estimate helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

from .schemas import MatchManifest

# Empirical planning constant: per-chunk JSON/parquet artifacts are small
# relative to source video; reserve a conservative fraction of input size.
ARTIFACT_FRACTION_OF_INPUT = 0.10
MIN_ESTIMATE_BYTES = 64 * 1024 * 1024


class DiskGuardError(RuntimeError):
    """Not enough free disk space to proceed safely."""


def free_disk_bytes(path: Path) -> int:
    target = Path(path)
    while not target.exists():
        parent = target.parent
        if parent == target:
            break
        target = parent
    return shutil.disk_usage(target).free


def estimate_run_bytes(manifest: MatchManifest) -> int:
    """Estimate artifact footprint for a run from probed input sizes."""
    total_input = sum(
        (camera.probe.size_bytes if camera.probe else 0) for camera in manifest.cameras
    )
    return max(MIN_ESTIMATE_BYTES, int(total_input * ARTIFACT_FRACTION_OF_INPUT))


def check_disk_guard(
    output_dir: Path,
    required_bytes: int,
    min_free_bytes: int = 1024 * 1024 * 1024,
) -> dict[str, int | bool | str]:
    """Verify free space covers the estimate plus a safety floor."""
    free = free_disk_bytes(output_dir)
    needed = int(required_bytes) + int(min_free_bytes)
    report = {
        "output_dir": str(output_dir),
        "free_bytes": free,
        "required_bytes": int(required_bytes),
        "min_free_bytes": int(min_free_bytes),
        "ok": free >= needed,
    }
    if not report["ok"]:
        raise DiskGuardError(
            f"insufficient disk space at {output_dir}: free={free} bytes, "
            f"required={needed} bytes (estimate {required_bytes} + floor {min_free_bytes})"
        )
    return report
