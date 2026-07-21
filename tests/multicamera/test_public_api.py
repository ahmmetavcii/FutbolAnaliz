"""The CLI scripts resolve entry points from football_analytics.multicamera."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def test_scripts_resolve_public_entry_points():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    cli = importlib.import_module("_full_match_cli")
    sync = cli.resolve_api("football_analytics.multicamera", ("synchronize_cameras", "sync_cameras"))
    calibrate = cli.resolve_api(
        "football_analytics.multicamera", ("calibrate_cameras", "calibrate_run")
    )
    assert callable(sync)
    assert callable(calibrate)


def test_all_exports_importable():
    module = importlib.import_module("football_analytics.multicamera")
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == []
