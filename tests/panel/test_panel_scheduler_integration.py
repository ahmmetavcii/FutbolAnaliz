"""Panel <-> real scheduler integration: command construction and progress."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps"))

import full_match_panel as panel  # noqa: E402


@pytest.fixture()
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    uploads = tmp_path / "uploads"
    results = tmp_path / "results"
    uploads.mkdir()
    results.mkdir()
    monkeypatch.setattr(panel, "UPLOAD_ROOT", uploads)
    monkeypatch.setattr(panel, "RESULTS_ROOT", results)
    return uploads, results


def test_prepare_command_targets_real_scheduler(sandbox) -> None:
    uploads, results = sandbox
    video = uploads / "m1" / "camera_1.mp4"
    video.parent.mkdir()
    video.write_bytes(b"x")
    command = panel.build_prepare_command("m1", [video], "m1/prepared")
    assert str(panel.PREPARE_FULL_MATCH_SCRIPT) in command
    assert "--match-id" in command and "m1" in command
    assert str(results / "m1" / "prepared") in command


def test_run_command_uses_real_adapter_by_default(sandbox) -> None:
    _, results = sandbox
    (results / "m1").mkdir()
    command = panel.build_full_match_run_command("m1/prepared", "m1/run")
    assert str(panel.RUN_FULL_MATCH_SCRIPT) in command
    assert "--chunk-pipeline-config" in command
    adapter = command[command.index("--chunk-pipeline-config") + 1]
    assert adapter.endswith("existing_pipeline_adapter.yaml")


def test_resume_command_repairs_manifests(sandbox) -> None:
    _, results = sandbox
    (results / "m1").mkdir()
    command = panel.build_resume_command("m1/run", rerun_from_stage="events")
    assert str(panel.RESUME_FULL_MATCH_SCRIPT) in command
    assert "--repair-manifests" in command
    assert command[command.index("--rerun-from-stage") + 1] == "events"


def test_read_run_progress_reports_real_state(sandbox) -> None:
    _, results = sandbox
    run_dir = results / "m1" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "chunk_manifest.json").write_text(
        json.dumps(
            {
                "records": [
                    {"camera_id": "camera_1", "chunk_index": 0, "status": "PASS",
                     "attempts": 1, "wall_seconds": 12.5},
                    {"camera_id": "camera_1", "chunk_index": 1, "status": "PENDING",
                     "attempts": 0, "wall_seconds": None},
                ]
            }
        )
    )
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "match_id": "m1",
                "stages": [
                    {"name": "prepare", "status": "PASS"},
                    {"name": "chunks", "status": "RUNNING"},
                ],
            }
        )
    )
    progress = panel.read_run_progress("m1/run")
    assert progress["exists"] is True
    assert progress["match_id"] == "m1"
    assert progress["stages"] == {"prepare": "PASS", "chunks": "RUNNING"}
    assert progress["chunks"]["total"] == 2
    assert progress["chunks"]["counts"] == {"PASS": 1, "PENDING": 1}
    assert progress["chunks"]["percent"] == 50.0
