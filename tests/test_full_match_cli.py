from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "prepare_full_match.py",
    "sync_cameras.py",
    "calibrate_cameras.py",
    "run_full_match.py",
    "resume_full_match.py",
    "validate_full_match_run.py",
    "export_full_match_results.py",
    "recompute_after_manual_correction.py",
)
PROFILES = ("single_camera", "dual_camera", "four_camera", "low_memory")


def test_full_match_commands_expose_help_without_runtime_packages() -> None:
    for name in SCRIPTS:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / name), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        assert completed.returncode == 0, f"{name}: {completed.stderr}"
        assert "usage:" in completed.stdout.lower()


def test_full_match_profiles_enforce_conservative_runtime_defaults() -> None:
    for profile in PROFILES:
        path = ROOT / "configs" / "full_match" / f"{profile}.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["full_match"]["chunk_seconds"] == 120
        assert config["cameras"]["processing"] == "sequential"
        assert config["inference"]["batch_size"] == 1
        assert config["inference"]["workers"] == 0
        assert config["inference"]["heavy_models"] == "sequential"
        assert config["resources"]["max_vram_gb"] > 0
        assert config["resources"]["max_ram_gb"] > 0
