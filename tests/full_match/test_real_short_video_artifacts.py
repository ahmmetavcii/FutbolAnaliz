"""Regression checks against the real short-video model run.

These tests validate genuine artifacts produced by the real detection/
tracking pipeline on football.mp4. They are skipped when the run directory
is absent (e.g. fresh checkout) so CI without the workspace stays green.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

RUN_DIR = Path("/home/ahmet/workspace/full_match_runs/run_football_short_model")

pytestmark = pytest.mark.skipif(
    not (RUN_DIR / "quality_report.json").is_file(),
    reason="real short-video model run not present on this machine",
)

ALLOWED_ROLES = {
    "outfield_player",
    "goalkeeper",
    "referee",
    "assistant_referee",
    "fourth_official",
    "substitute",
    "staff",
    "unknown_person",
}
OFFICIAL_ROLES = {"referee", "assistant_referee", "fourth_official"}


@pytest.fixture(scope="module")
def quality() -> dict:
    return json.loads((RUN_DIR / "quality_report.json").read_text())


@pytest.fixture(scope="module")
def run_report() -> dict:
    return json.loads((RUN_DIR / "run_report.json").read_text())


def test_real_detection_artifact(quality: dict) -> None:
    frame = pd.read_parquet(RUN_DIR / "detections.parquet")
    assert len(frame) > 0
    assert {"frame_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "detection_confidence", "source_model"} <= set(frame.columns)
    assert len(frame) == quality["detections"]


def test_real_tracking_artifact(quality: dict) -> None:
    frame = pd.read_parquet(RUN_DIR / "local_tracks.parquet")
    assert len(frame) > 0
    assert {"camera_id", "track_id", "frame_id", "source_tracker"} <= set(frame.columns)
    assert frame["track_id"].nunique() == quality["unique_tracks"]


def test_calibration_artifact(quality: dict) -> None:
    payload = json.loads((RUN_DIR / "camera_calibrations.json").read_text())
    camera = payload["cameras"]["camera_1"]
    assert camera["frames"] > 0
    assert 0.0 <= camera["valid_ratio"] <= 1.0
    assert camera["valid_ratio"] == pytest.approx(quality["calibration_valid_ratio"])
    assert camera["homography_sample"] is not None


def test_non_synthetic_marker(run_report: dict) -> None:
    """Model outputs must trace back to a real pipeline run with a real model."""
    pipeline = run_report["pipeline_run_report"]
    assert pipeline["status"] == "PASS"
    assert pipeline["model"]["name"]
    assert Path(pipeline["model"]["path"]).name.endswith(".pt")
    assert pipeline["detection_metrics"]["detections"] > 0
    assert pipeline["tracking_metrics"]["bytetrack"]["track_rows"] > 0


def test_model_stages_claimed_consistency(quality: dict, run_report: dict) -> None:
    claimed = quality["model_stages_claimed"]
    assert claimed == run_report["model_stages_claimed"]
    if claimed:
        assert quality["detections"] > 0
        assert quality["track_rows"] > 0
    else:
        assert quality["detections"] == 0


def test_role_team_consistency() -> None:
    roles = pd.read_parquet(RUN_DIR / "role_predictions.parquet")
    assert set(roles["role"]) <= ALLOWED_ROLES
    players = pd.read_csv(RUN_DIR / "player_summary.csv")
    # Player summaries only ever carry team-member roles.
    assert set(players["role"]) <= {"outfield_player", "goalkeeper", "substitute"}


def test_referee_excluded_from_team_totals() -> None:
    players = pd.read_csv(RUN_DIR / "player_summary.csv")
    assert not players["role"].isin(OFFICIAL_ROLES).any()
    officials = pd.read_csv(RUN_DIR / "officials_summary.csv")
    if len(officials):
        assert officials["team_id"].isna().all()
        assert not officials["counts_toward_team_totals"].any()
    # Team totals must be reproducible from countable players only.
    teams = pd.read_csv(RUN_DIR / "team_summary.csv")
    countable = players[players["counts_toward_team_totals"]]
    for row in teams.itertuples(index=False):
        members = countable[countable["team_id"] == row.team_id]
        assert row.player_count == len(members)


def test_global_identity_schema() -> None:
    mapping = pd.read_parquet(RUN_DIR / "global_identity_map.parquet")
    players = pd.read_parquet(RUN_DIR / "global_players.parquet")
    assert {"camera_id", "local_track_id", "global_id", "unresolved"} <= set(mapping.columns)
    assert {"global_id", "n_local_tracks", "team_id", "role", "unresolved"} <= set(players.columns)
    # One local track never maps to two global identities (no impossible duplicates).
    assert not mapping.duplicated(["camera_id", "local_track_id"]).any()
    assert set(mapping["global_id"]) <= set(players["global_id"])


def test_empty_event_honesty(quality: dict) -> None:
    events = pd.read_parquet(RUN_DIR / "events.parquet")
    assert len(events) == 0
    assert quality["confirmed_events"] == 0
    assert quality["candidate_events"] == 0
    assert quality["events_reason"] == "no_supported_event_detected"
    assert any("events" in note for note in quality["empty_or_limited_artifacts"])


def test_xlsx_reopens_with_all_sheets() -> None:
    from football_analytics.export.excel_exporter import validate_excel_workbook

    result = validate_excel_workbook(RUN_DIR / "full_match_report.xlsx")
    assert result["validated"] is True
    assert result["rows_per_sheet"]["Player Summary"] > 0


@pytest.mark.parametrize("name", ["annotated_match.mp4", "tactical_map.mp4"])
def test_mp4_decodes(name: str) -> None:
    from football_analytics.export.video_exporter import validate_video_export

    result = validate_video_export(RUN_DIR / name)
    assert result["ffprobe"]["duration_seconds"] > 0
    assert result["opencv"]["opened"] is True


def test_resume_without_duplicate_processing() -> None:
    from football_analytics.full_match import resume_full_match

    orchestrated = RUN_DIR / "orchestrated"
    if not (orchestrated / "run_state.json").is_file():
        pytest.skip("orchestrated run dir missing")
    report = resume_full_match(run_dir=orchestrated, dry_run=True)
    assert report["status"] == "PASS"
    assert report["chunks_to_run"] == 0
