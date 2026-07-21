"""Regression checks against the real football.mp4 panel-driven analysis.

These tests only run when the real single-button panel flow has completed
on this machine (they never fabricate results).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MATCH_DIR = Path("/mnt/c/football_data/results/football-panel")

def _finished() -> bool:
    path = MATCH_DIR / "analysis_state.json"
    if not path.is_file():
        return False
    try:
        status = json.loads(path.read_text(encoding="utf-8")).get("status")
    except (json.JSONDecodeError, OSError):
        return False
    return status in {"COMPLETED", "FAILED"}


pytestmark = pytest.mark.skipif(
    not _finished(),
    reason="real panel run not present or still in progress on this machine",
)


def _state() -> dict:
    return json.loads((MATCH_DIR / "analysis_state.json").read_text(encoding="utf-8"))


def test_panel_run_completed_via_single_flow():
    state = _state()
    assert state["status"] == "COMPLETED", state.get("error")
    assert state["run_id"] == "football-panel"
    assert state.get("elapsed_seconds", 0) > 0


def test_panel_run_wrote_heartbeats():
    heartbeat = json.loads((MATCH_DIR / "run" / "heartbeat.json").read_text())
    for field in (
        "timestamp", "pid", "current_stage", "processed_frames",
        "total_frames", "status",
    ):
        assert field in heartbeat


def test_panel_run_produced_real_outputs():
    output = MATCH_DIR / "output"
    for name in (
        "annotated_match.mp4",
        "tactical_map.mp4",
        "full_match_report.xlsx",
        "player_summary.csv",
        "team_summary.csv",
        "quality_report.json",
        "detections.parquet",
        "local_tracks.parquet",
    ):
        assert (output / name).is_file(), name


def test_panel_run_model_stages_claimed_only_with_real_models():
    quality = json.loads((MATCH_DIR / "output" / "quality_report.json").read_text())
    if quality.get("model_stages_claimed"):
        assert quality.get("detections", 0) > 0
        assert quality.get("tracks", quality.get("global_players", 0)) > 0


def test_panel_run_scheduler_state_consistent():
    run_state = json.loads((MATCH_DIR / "run" / "run_state.json").read_text())
    stages = {stage["name"]: stage["status"] for stage in run_state["stages"]}
    assert stages.get("chunks") == "PASS"
    chunk_manifest = json.loads((MATCH_DIR / "run" / "chunk_manifest.json").read_text())
    statuses = {record["status"] for record in chunk_manifest["records"]}
    assert statuses <= {"PASS", "SKIPPED"}
