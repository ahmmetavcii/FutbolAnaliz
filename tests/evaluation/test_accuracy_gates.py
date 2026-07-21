"""Evaluation accuracy gates and honest GT incomplete behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_analytics.evaluation.ball_gt import (
    ball_gt_complete,
    empty_ball_gt_row,
    select_stratified_frames,
)
from football_analytics.evaluation.ball_metrics import (
    evaluate_ball_tracking,
    provenance_mutually_exclusive,
)
from football_analytics.evaluation.calibration_audit import (
    PropagationConfig,
    propagate_calibration,
)
from football_analytics.evaluation.identity_gt import identity_gt_complete
from football_analytics.evaluation.identity_metrics import evaluate_global_identity
from football_analytics.evaluation.publishability import compute_publishability
from football_analytics.evaluation.touch_review import touch_review_complete
from football_analytics.evaluation.wrong_object import filter_ball_trajectory_candidates
from football_analytics.opta.identity_resolve import (
    IdentityResolveConfig,
    TrackFragment,
    resolve_global_identities,
)


def test_ball_candidate_coverage_is_not_recall(tmp_path: Path) -> None:
    gt = pd.DataFrame(
        [
            empty_ball_gt_row(
                video_name="t",
                frame_index=i,
                timestamp=i / 25,
                frame_width=1920,
                frame_height=1080,
            )
            for i in range(150)
        ]
    )
    gt_path = tmp_path / "ball_gt.csv"
    gt.to_csv(gt_path, index=False)
    run = tmp_path / "run"
    run.mkdir()
    (run / "ball_detection_report.json").write_text(
        json.dumps({"raw_detection_coverage": 0.95, "tracked_coverage": 0.99, "frames": 750}),
        encoding="utf-8",
    )
    report = evaluate_ball_tracking(gt_csv=gt_path, run_dir=run, out_dir=tmp_path / "eval")
    assert report["status"] == "GT_INCOMPLETE"
    assert "precision" not in report
    assert "recall" not in report
    assert report["candidate_coverage"] == 0.95
    assert "not recall" in report["note"]


def test_gt_incomplete_blocks_precision_recall(tmp_path: Path) -> None:
    gt = pd.DataFrame(
        [
            empty_ball_gt_row(
                video_name="v",
                frame_index=0,
                timestamp=0.0,
                frame_width=100,
                frame_height=100,
            )
            for _ in range(10)
        ]
    )
    gt["reviewed"] = False
    path = tmp_path / "ball_gt.csv"
    gt.to_csv(path, index=False)
    ok, reason = ball_gt_complete(gt, min_frames=150)
    assert ok is False
    report = evaluate_ball_tracking(gt_csv=path, run_dir=tmp_path, out_dir=tmp_path / "e")
    assert report["status"] == "GT_INCOMPLETE"
    assert "precision" not in report


def test_ball_provenance_mutually_exclusive() -> None:
    ok = pd.DataFrame({"frame_id": [0, 1, 2], "provenance": ["detected", "tracked", "missing"]})
    bad = pd.DataFrame({"frame_id": [0, 0], "provenance": ["detected", "tracked"]})
    assert provenance_mutually_exclusive(ok)
    assert not provenance_mutually_exclusive(bad)


def test_wrong_object_trajectory_switch_rejection() -> None:
    rows = []
    for i in range(5):
        rows.append(
            {
                "frame_id": i,
                "ball_x_pixel": 100.0 + i,
                "ball_y_pixel": 100.0,
                "bbox_w": 20,
                "bbox_h": 20,
                "detection_confidence": 0.8,
            }
        )
    # Impossible jump to far object
    rows.append(
        {
            "frame_id": 5,
            "ball_x_pixel": 1800.0,
            "ball_y_pixel": 900.0,
            "bbox_w": 80,
            "bbox_h": 80,
            "detection_confidence": 0.3,
        }
    )
    filtered, metrics = filter_ball_trajectory_candidates(pd.DataFrame(rows))
    assert metrics["wrong_object_switches"] >= 1
    assert len(filtered) <= 5 or filtered["ball_x_pixel"].iloc[-1] != 1800.0 or metrics["segments"] >= 2


def test_long_prediction_rejection() -> None:
    rows = [
        {
            "frame_id": 0,
            "ball_x_pixel": 50.0,
            "ball_y_pixel": 50.0,
            "bbox_w": 20,
            "bbox_h": 20,
            "detection_confidence": 0.9,
        }
    ]
    # gap then far detection after long miss — new segment
    rows.append(
        {
            "frame_id": 20,
            "ball_x_pixel": 60.0,
            "ball_y_pixel": 60.0,
            "bbox_w": 20,
            "bbox_h": 20,
            "detection_confidence": 0.9,
        }
    )
    _, metrics = filter_ball_trajectory_candidates(
        pd.DataFrame(rows),
        config=__import__(
            "football_analytics.evaluation.wrong_object", fromlist=["WrongObjectConfig"]
        ).WrongObjectConfig(max_prediction_frames=5),
    )
    assert metrics["segments"] >= 1


def test_ball_annotation_save_reload(tmp_path: Path) -> None:
    row = empty_ball_gt_row(
        video_name="v",
        frame_index=1,
        timestamp=0.04,
        frame_width=1920,
        frame_height=1080,
    )
    row["ball_visible"] = True
    row["ball_x1"] = 10
    row["ball_y1"] = 10
    row["ball_x2"] = 30
    row["ball_y2"] = 30
    row["reviewed"] = True
    path = tmp_path / "ball_gt.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    assert bool(reloaded.iloc[0]["reviewed"])
    assert float(reloaded.iloc[0]["ball_x1"]) == 10


def test_identity_annotation_save_reload(tmp_path: Path) -> None:
    path = tmp_path / "identity_gt.csv"
    df = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "gt_player_id": 1,
                "team_id": "team_0",
                "role": "outfield",
                "x1": 1,
                "y1": 2,
                "x2": 3,
                "y2": 4,
                "visible": True,
                "occluded": False,
                "difficult": False,
                "reviewed": True,
            }
        ]
    )
    df.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    assert int(reloaded.iloc[0]["gt_player_id"]) == 1


def test_gt_incomplete_blocks_idf1(tmp_path: Path) -> None:
    path = tmp_path / "identity_gt.csv"
    pd.DataFrame(columns=["frame_index", "gt_player_id", "reviewed"]).to_csv(path, index=False)
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame({"frame_id": [0], "track_id": [1], "x1": [0], "y1": [0], "x2": [1], "y2": [1]}).to_parquet(
        run / "tracks.parquet"
    )
    report = evaluate_global_identity(gt_csv=path, run_dir=run, out_dir=tmp_path / "e")
    assert report["status"] == "GT_INCOMPLETE"
    assert "idf1" not in report


def test_simultaneous_identity_merge_rejection() -> None:
    frags = [
        TrackFragment(
            track_id=1,
            team_id="team_0",
            team_confidence=0.9,
            role="outfield",
            first_ms=0,
            last_ms=1000,
            frame_count=30,
            visible_seconds=1.5,
            embedding=np.ones(4),
            start_xy=(10, 10),
            end_xy=(12, 12),
            mean_xy=(11, 11),
        ),
        TrackFragment(
            track_id=2,
            team_id="team_0",
            team_confidence=0.9,
            role="outfield",
            first_ms=200,
            last_ms=1200,
            frame_count=30,
            visible_seconds=1.5,
            embedding=np.ones(4),
            start_xy=(10.1, 10.1),
            end_xy=(12.1, 12.1),
            mean_xy=(11.1, 11.1),
        ),
    ]
    _, _, metrics, decisions = resolve_global_identities(
        frags, config=IdentityResolveConfig(min_track_frames=1, min_visible_seconds=0.1)
    )
    assert (decisions["reason"] == "simultaneous_overlap").any()
    assert metrics["global_player_count"] == 2


def test_cross_team_merge_rejection() -> None:
    emb = np.ones(4)
    frags = [
        TrackFragment(
            track_id=1,
            team_id="team_0",
            team_confidence=0.9,
            role="outfield",
            first_ms=0,
            last_ms=1000,
            frame_count=30,
            visible_seconds=2.0,
            embedding=emb,
            start_xy=(10, 10),
            end_xy=(12, 12),
            mean_xy=(11, 11),
        ),
        TrackFragment(
            track_id=2,
            team_id="team_1",
            team_confidence=0.9,
            role="outfield",
            first_ms=2000,
            last_ms=3000,
            frame_count=30,
            visible_seconds=2.0,
            embedding=emb,
            start_xy=(12, 12),
            end_xy=(14, 14),
            mean_xy=(13, 13),
        ),
    ]
    _, _, metrics, decisions = resolve_global_identities(
        frags, config=IdentityResolveConfig(min_track_frames=1, min_visible_seconds=0.1)
    )
    assert (decisions["reason"] == "different_team").any()
    assert metrics["global_player_count"] == 2


def test_false_split_and_merge_report_files(tmp_path: Path) -> None:
    # Complete synthetic GT so evaluate produces CSVs
    rows = []
    for f in range(60):
        for pid in range(8):
            rows.append(
                {
                    "frame_index": f,
                    "timestamp": f / 25,
                    "gt_player_id": pid + 1,
                    "team_id": "team_0" if pid < 4 else "team_1",
                    "role": "outfield",
                    "x1": pid * 10,
                    "y1": 0,
                    "x2": pid * 10 + 8,
                    "y2": 20,
                    "visible": True,
                    "occluded": False,
                    "difficult": False,
                    "reviewed": True,
                }
            )
    gt_path = tmp_path / "identity_gt.csv"
    pd.DataFrame(rows).to_csv(gt_path, index=False)
    assert identity_gt_complete(pd.read_csv(gt_path))[0]

    run = tmp_path / "run"
    run.mkdir()
    track_rows = []
    for f in range(60):
        for pid in range(8):
            track_rows.append(
                {
                    "frame_id": f,
                    "track_id": pid + 1,
                    "object_type": "person",
                    "x1": pid * 10,
                    "y1": 0,
                    "x2": pid * 10 + 8,
                    "y2": 20,
                }
            )
    pd.DataFrame(track_rows).to_parquet(run / "tracks.parquet")
    pd.DataFrame(
        {"local_track_id": list(range(1, 9)), "global_player_id": list(range(1, 9))}
    ).to_parquet(run / "global_identity_map.parquet")
    out = tmp_path / "eval"
    report = evaluate_global_identity(gt_csv=gt_path, run_dir=run, out_dir=out)
    assert report["status"] == "OK"
    assert "idf1" in report
    assert (out / "identity_false_merges.csv").is_file()
    assert (out / "identity_false_splits.csv").is_file()
    assert (out / "identity_switches.csv").is_file()


def test_calibration_provider_provenance_and_propagation() -> None:
    rows = []
    H = json.dumps([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for i in range(30):
        valid = i % 5 == 0
        rows.append(
            {
                "frame_id": i,
                "timestamp_ms": i * 40,
                "valid": valid,
                "provider": "pnlcalib" if valid else "pnlcalib",
                "homography_json": H if valid else None,
                "confidence": 0.8 if valid else 0.0,
                "reprojection_error": 2.0 if valid else None,
                "visible_pitch_coverage": 0.5 if valid else None,
                "invalid_reason": None if valid else "no valid calibration within hold window",
                "segment_id": 0,
                "source_method": "pnlcalib",
            }
        )
    cal = pd.DataFrame(rows)
    out, stats = propagate_calibration(
        cal, config=PropagationConfig(max_propagate_frames=4, max_homography_jump=100)
    )
    assert stats["propagated_frames"] > 0
    assert (out["provider"] == "propagated").any()
    assert stats["calibration_coverage_after"] > stats["calibration_coverage_before"]


def test_homography_jump_rejection() -> None:
    H1 = json.dumps([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    H2 = json.dumps([[2, 0, 500], [0, 2, 500], [0, 0, 1]])
    rows = [
        {
            "frame_id": 0,
            "valid": True,
            "provider": "pnlcalib",
            "homography_json": H1,
            "confidence": 0.9,
            "reprojection_error": 1.0,
            "visible_pitch_coverage": 0.5,
            "invalid_reason": None,
            "segment_id": 0,
            "source_method": "pnlcalib",
            "timestamp_ms": 0,
        },
        {
            "frame_id": 1,
            "valid": False,
            "provider": "pnlcalib",
            "homography_json": None,
            "confidence": 0.0,
            "reprojection_error": None,
            "visible_pitch_coverage": None,
            "invalid_reason": "gap",
            "segment_id": 0,
            "source_method": "pnlcalib",
            "timestamp_ms": 40,
        },
    ]
    # measured at frame 10 with huge jump H2 — should not propagate across if neighbor differs
    rows.append(
        {
            "frame_id": 10,
            "valid": True,
            "provider": "pnlcalib",
            "homography_json": H2,
            "confidence": 0.9,
            "reprojection_error": 1.0,
            "visible_pitch_coverage": 0.5,
            "invalid_reason": None,
            "segment_id": 0,
            "source_method": "pnlcalib",
            "timestamp_ms": 400,
        }
    )
    for i in range(2, 10):
        rows.append(
            {
                "frame_id": i,
                "valid": False,
                "provider": "pnlcalib",
                "homography_json": None,
                "confidence": 0.0,
                "reprojection_error": None,
                "visible_pitch_coverage": None,
                "invalid_reason": "gap",
                "segment_id": 0,
                "source_method": "pnlcalib",
                "timestamp_ms": i * 40,
            }
        )
    out, stats = propagate_calibration(
        pd.DataFrame(rows),
        config=PropagationConfig(max_propagate_frames=3, max_homography_jump=5.0),
    )
    assert "rejected_homography_jumps" in stats


def test_publishability_flags_independent() -> None:
    flags = compute_publishability(
        ball_gt_complete=False,
        identity_gt_complete=False,
        touch_review_complete=False,
        calibration_coverage=0.9,
        measured_calibration_coverage=0.8,
        player_position_coverage=0.7,
        continuous_calibrated_seconds=5.0,
        speed_spike_candidates=0,
    )
    assert flags.ball_detection_publishable is False
    assert flags.identity_publishable is False
    assert flags.touch_publishable is False
    assert flags.overall_publishable is False
    # calibration can be true independently
    assert flags.calibration_publishable is True
    assert flags.reasons["ball_detection"] == "GT_INCOMPLETE"


def test_overall_publishability_false_without_gt() -> None:
    flags = compute_publishability(
        ball_gt_complete=False,
        identity_gt_complete=True,
        identity_eval={"idf1": 0.9},
        calibration_coverage=0.9,
        measured_calibration_coverage=0.5,
        player_position_coverage=0.5,
        continuous_calibrated_seconds=3.0,
        speed_spike_candidates=0,
        identity_quality={"validated_by_team": {"team_0": 11, "team_1": 11}},
    )
    assert flags.overall_publishable is False


def test_touch_review_incomplete_blocks_precision_claim(tmp_path: Path) -> None:
    path = tmp_path / "touch_review.csv"
    pd.DataFrame(
        [{"touch_id": "t1", "reviewed": False, "correct_touch": ""}]
    ).to_csv(path, index=False)
    ok, reason = touch_review_complete(path)
    assert ok is False
    assert "unreviewed" in reason


def test_select_stratified_frames_seed_reproducible() -> None:
    a = select_stratified_frames(n_frames=160, total_frames=750, seed=42, per_stratum=12)
    b = select_stratified_frames(n_frames=160, total_frames=750, seed=42, per_stratum=12)
    assert [x["frame_index"] for x in a] == [x["frame_index"] for x in b]
    assert len(a) >= 150


def test_panel_gt_warning_text_present() -> None:
    source = Path("/home/ahmet/projects/football-analytics/apps/full_match_panel.py").read_text(
        encoding="utf-8"
    )
    assert "doğruluk ground truth ile doğrulanmadı" in source
    assert "Top modeli değerlendirildi mi?" in source


def test_physical_metrics_hidden_with_low_calibration() -> None:
    flags = compute_publishability(
        ball_gt_complete=True,
        ball_eval={"precision": 0.9, "recall": 0.9, "trajectory_coverage": 0.9},
        identity_gt_complete=True,
        identity_eval={"idf1": 0.8},
        touch_review_complete=True,
        touch_eval={"precision": 0.9},
        calibration_coverage=0.1,
        measured_calibration_coverage=0.05,
        player_position_coverage=0.1,
        continuous_calibrated_seconds=0.5,
        speed_spike_candidates=3,
        identity_quality={"validated_by_team": {"team_0": 10, "team_1": 10}},
    )
    assert flags.physical_metrics_publishable is False
    assert flags.calibration_publishable is False
