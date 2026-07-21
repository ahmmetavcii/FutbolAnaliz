from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_analytics.full_match import (
    prepare_full_match,
    recompute_after_manual_correction,
    run_full_match,
    validate_full_match_run,
)


@pytest.fixture()
def completed_run(tmp_path: Path, make_video, base_config) -> Path:
    video = make_video(tmp_path / "cam.mp4", seconds=35.0, fps=10.0)
    prepared_dir = tmp_path / "prepared"
    run_dir = tmp_path / "run"
    prepare_full_match(
        inputs=[video],
        camera_ids=["cam_1"],
        config=base_config,
        output_dir=prepared_dir,
        match_id="match-v",
    )
    run_full_match(prepared_dir=prepared_dir, run_dir=run_dir, config=base_config)
    return run_dir


def test_validate_passes_for_completed_run(completed_run: Path, tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report = validate_full_match_run(
        run_dir=completed_run,
        verify_checksums=True,
        open_media=True,
        report_path=report_path,
    )
    assert report["status"] == "PASS"
    assert report["errors"] == []
    assert report_path.is_file()


def test_validate_detects_missing_chunk_artifact(completed_run: Path) -> None:
    chunk_files = sorted((completed_run / "chunks" / "cam_1").glob("*.json"))
    chunk_files[0].unlink()
    report = validate_full_match_run(run_dir=completed_run, verify_checksums=False)
    assert report["status"] == "FAILED"
    assert any("chunk artifact missing" in error for error in report["errors"])


def test_validate_strict_promotes_warnings(completed_run: Path) -> None:
    report = validate_full_match_run(
        run_dir=completed_run, verify_checksums=False, strict=True
    )
    # Infrastructure-only runs carry SKIPPED model stages as warnings.
    assert report["warnings"]
    assert report["status"] == "FAILED"


def test_recompute_after_manual_correction_in_place(
    completed_run: Path, tmp_path: Path
) -> None:
    corrections = tmp_path / "corrections.json"
    corrections.write_text(
        json.dumps({"track_reassignments": [{"from": 3, "to": 7}]}), encoding="utf-8"
    )
    result = recompute_after_manual_correction(
        run_dir=completed_run,
        corrections_path=corrections,
        from_stage="events",
        in_place=True,
    )
    assert result["status"] == "PASS"
    assert "events" in result["invalidated_stages"]
    assert "export" in result["invalidated_stages"]
    record_path = completed_run / result["correction_record"]
    assert record_path.is_file()
    summary = result["consolidation"]
    assert summary["corrections_applied"] == [record_path.name]
