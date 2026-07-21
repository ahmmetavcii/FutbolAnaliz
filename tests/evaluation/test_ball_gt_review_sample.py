"""Tests for 50-frame ball GT review sample + safe save/counter."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from football_analytics.evaluation.ball_gt import empty_ball_gt_row, load_ball_gt_csv, review_sample_complete
from football_analytics.evaluation.ball_gt_annotate import (
    first_unreviewed_sample_index,
    recount,
    save_current_annotation,
)
from football_analytics.evaluation.ball_metrics import evaluate_ball_tracking
from football_analytics.evaluation.ball_review_sample import (
    create_balanced_review_sample,
    sample_reviewed_count,
    write_review_sample,
)
from football_analytics.evaluation.boolean_utils import (
    apply_boolean_dtype,
    is_true,
    normalize_boolean,
)


def _make_gt(n: int = 165, reviewed_frames: set[int] | None = None) -> pd.DataFrame:
    reviewed_frames = reviewed_frames or set()
    rows = []
    for i in range(n):
        row = empty_ball_gt_row(
            video_name="football",
            frame_index=i * 3,  # sparse ids like real stratified sample
            timestamp=i * 0.12,
            frame_width=1920,
            frame_height=1080,
        )
        fid = int(row["frame_index"])
        if fid in reviewed_frames:
            row["reviewed"] = True
            row["ball_visible"] = True
            row["ball_x1"] = 10
            row["ball_y1"] = 10
            row["ball_x2"] = 30
            row["ball_y2"] = 30
        else:
            row["reviewed"] = False
        rows.append(row)
    return apply_boolean_dtype(pd.DataFrame(rows))


def _sample_from_gt(gt: pd.DataFrame, size: int = 50) -> pd.DataFrame:
    return create_balanced_review_sample(ball_gt=gt, sample_size=size, seed=7)


def test_normalize_boolean_false_string_not_true() -> None:
    assert normalize_boolean("False") is False
    assert normalize_boolean("false") is False
    assert is_true("False") is False
    assert normalize_boolean(True) is True
    assert normalize_boolean(1) is True
    assert normalize_boolean(0) is False
    assert normalize_boolean("") is False
    assert normalize_boolean(None) is pd.NA
    assert is_true(None) is False
    assert is_true(float("nan")) is False


def test_review_sample_exactly_50_unique() -> None:
    gt = _make_gt(165)
    sample = _sample_from_gt(gt, 50)
    assert len(sample) == 50
    assert sample["frame_idx"].nunique() == 50
    assert sample_reviewed_count(sample, gt) == 0  # auto reviewed in sample csv is 0


def test_automatic_reviewed_count_zero_in_sample_csv(tmp_path: Path) -> None:
    gt = _make_gt(165)
    sample = _sample_from_gt(gt, 50)
    path = tmp_path / "review_sample.csv"
    write_review_sample(sample, path)
    loaded = pd.read_csv(path)
    assert int((loaded["reviewed"].astype(str).str.lower() == "true").sum()) == 0


def test_save_increments_0_to_1(tmp_path: Path) -> None:
    gt = _make_gt(165)
    sample = _sample_from_gt(gt, 50)
    gt_path = tmp_path / "ball_gt.csv"
    gt.to_csv(gt_path, index=False)
    gt = load_ball_gt_csv(gt_path)
    fid = int(sample.iloc[0]["frame_idx"])
    assert recount(sample, gt)["sample_reviewed"] == 0
    gt2, res = save_current_annotation(
        gt=gt,
        sample=sample,
        gt_path=gt_path,
        frame_idx=fid,
        bbox=(10, 10, 40, 40),
        ball_visible=True,
    )
    assert res["sample_reviewed"] == 1
    assert res["sample_size"] == 50
    assert res["csv_verified"] is True
    assert recount(sample, gt2)["sample_reviewed"] == 1


def test_second_save_same_frame_stays_1(tmp_path: Path) -> None:
    gt = _make_gt(165)
    sample = _sample_from_gt(gt, 50)
    gt_path = tmp_path / "ball_gt.csv"
    gt.to_csv(gt_path, index=False)
    gt = load_ball_gt_csv(gt_path)
    fid = int(sample.iloc[0]["frame_idx"])
    gt, _ = save_current_annotation(
        gt=gt, sample=sample, gt_path=gt_path, frame_idx=fid, bbox=(1, 1, 2, 2), ball_visible=True
    )
    gt, res = save_current_annotation(
        gt=gt, sample=sample, gt_path=gt_path, frame_idx=fid, bbox=(1, 1, 3, 3), ball_visible=True
    )
    assert res["sample_reviewed"] == 1


def test_save_19_to_20(tmp_path: Path) -> None:
    gt = _make_gt(165)
    sample = _sample_from_gt(gt, 50)
    gt_path = tmp_path / "ball_gt.csv"
    # pre-mark first 19 sample frames reviewed in gt
    for fid in sample["frame_idx"].tolist()[:19]:
        idx = gt.index[gt["frame_index"] == int(fid)][0]
        gt.at[idx, "reviewed"] = True
        gt.at[idx, "ball_visible"] = True
        gt.at[idx, "ball_x1"] = 1
        gt.at[idx, "ball_y1"] = 1
        gt.at[idx, "ball_x2"] = 2
        gt.at[idx, "ball_y2"] = 2
    gt = apply_boolean_dtype(gt)
    gt.to_csv(gt_path, index=False)
    gt = load_ball_gt_csv(gt_path)
    assert recount(sample, gt)["sample_reviewed"] == 19
    fid = int(sample["frame_idx"].tolist()[19])
    gt, res = save_current_annotation(
        gt=gt, sample=sample, gt_path=gt_path, frame_idx=fid, bbox=(5, 5, 15, 15), ball_visible=True
    )
    assert res["sample_reviewed"] == 20


def test_csv_reload_preserves_reviewed(tmp_path: Path) -> None:
    gt = _make_gt(20)
    sample = _sample_from_gt(gt, 10)
    gt_path = tmp_path / "ball_gt.csv"
    gt.to_csv(gt_path, index=False)
    gt = load_ball_gt_csv(gt_path)
    fid = int(sample.iloc[0]["frame_idx"])
    save_current_annotation(
        gt=gt, sample=sample, gt_path=gt_path, frame_idx=fid, bbox=(1, 1, 2, 2), ball_visible=True
    )
    reloaded = load_ball_gt_csv(gt_path)
    idx = reloaded.index[reloaded["frame_index"] == fid][0]
    assert is_true(reloaded.at[idx, "reviewed"]) is True


def test_v_clears_bbox_and_sets_reviewed(tmp_path: Path) -> None:
    gt = _make_gt(165)
    sample = _sample_from_gt(gt, 50)
    gt_path = tmp_path / "ball_gt.csv"
    gt.to_csv(gt_path, index=False)
    gt = load_ball_gt_csv(gt_path)
    fid = int(sample.iloc[0]["frame_idx"])
    # first put a box
    gt, _ = save_current_annotation(
        gt=gt, sample=sample, gt_path=gt_path, frame_idx=fid, bbox=(1, 1, 2, 2), ball_visible=True
    )
    gt, res = save_current_annotation(
        gt=gt, sample=sample, gt_path=gt_path, frame_idx=fid, ball_visible=False, clear_bbox=True
    )
    idx = gt.index[gt["frame_index"] == fid][0]
    assert is_true(gt.at[idx, "reviewed"])
    assert is_true(gt.at[idx, "ball_visible"]) is False
    assert pd.isna(gt.at[idx, "ball_x1"])
    assert res["sample_reviewed"] == 1


def test_save_failure_does_not_advance(tmp_path: Path) -> None:
    gt = _make_gt(165)
    sample = _sample_from_gt(gt, 50)
    gt_path = tmp_path / "ball_gt.csv"
    gt.to_csv(gt_path, index=False)
    gt = load_ball_gt_csv(gt_path)
    fid = int(sample.iloc[0]["frame_idx"])
    with pytest.raises(RuntimeError):
        save_current_annotation(
            gt=gt, sample=sample, gt_path=gt_path, frame_idx=fid, ball_visible=True
        )
    # still 0 reviewed
    assert recount(sample, load_ball_gt_csv(gt_path))["sample_reviewed"] == 0


def test_sample_completion_ignores_outside_frames(tmp_path: Path) -> None:
    gt = _make_gt(165)
    sample = _sample_from_gt(gt, 50)
    # mark a non-sample frame reviewed
    outside = [f for f in gt["frame_index"].tolist() if int(f) not in set(sample["frame_idx"])][0]
    idx = gt.index[gt["frame_index"] == outside][0]
    gt.at[idx, "reviewed"] = True
    gt.at[idx, "ball_visible"] = True
    gt.at[idx, "ball_x1"] = 1
    gt.at[idx, "ball_y1"] = 1
    gt.at[idx, "ball_x2"] = 2
    gt.at[idx, "ball_y2"] = 2
    assert sample_reviewed_count(sample, apply_boolean_dtype(gt)) == 0


def test_evaluator_49_incomplete(tmp_path: Path) -> None:
    gt = _make_gt(165)
    sample = _sample_from_gt(gt, 50)
    for fid in sample["frame_idx"].tolist()[:49]:
        idx = gt.index[gt["frame_index"] == int(fid)][0]
        gt.at[idx, "reviewed"] = True
        gt.at[idx, "ball_visible"] = True
        gt.at[idx, "ball_x1"] = 1
        gt.at[idx, "ball_y1"] = 1
        gt.at[idx, "ball_x2"] = 2
        gt.at[idx, "ball_y2"] = 2
    gt = apply_boolean_dtype(gt)
    gt_path = tmp_path / "ball_gt.csv"
    sample_path = tmp_path / "review_sample.csv"
    gt.to_csv(gt_path, index=False)
    write_review_sample(sample, sample_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "ball_detection_report.json").write_text(
        json.dumps({"raw_detection_coverage": 0.9, "tracked_coverage": 0.95, "frames": 750})
    )
    report = evaluate_ball_tracking(
        gt_csv=gt_path, run_dir=run, out_dir=tmp_path / "e", review_sample_csv=sample_path
    )
    assert report["status"] == "GT_INCOMPLETE"
    assert "precision" not in report
    assert report["manually_reviewed"] == 49
    assert report["remaining"] == 1


def test_evaluator_50_produces_metrics(tmp_path: Path) -> None:
    gt = _make_gt(165)
    sample = _sample_from_gt(gt, 50)
    for fid in sample["frame_idx"].tolist():
        idx = gt.index[gt["frame_index"] == int(fid)][0]
        gt.at[idx, "reviewed"] = True
        gt.at[idx, "ball_visible"] = True
        gt.at[idx, "ball_x1"] = 10
        gt.at[idx, "ball_y1"] = 10
        gt.at[idx, "ball_x2"] = 30
        gt.at[idx, "ball_y2"] = 30
    gt = apply_boolean_dtype(gt)
    gt_path = tmp_path / "ball_gt.csv"
    sample_path = tmp_path / "review_sample.csv"
    gt.to_csv(gt_path, index=False)
    write_review_sample(sample, sample_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "ball_detection_report.json").write_text(
        json.dumps({"raw_detection_coverage": 0.9, "tracked_coverage": 0.95, "frames": 750})
    )
    # matching preds
    pred_rows = []
    for fid in sample["frame_idx"].tolist():
        pred_rows.append(
            {
                "frame_id": int(fid),
                "ball_x_pixel": 20.0,
                "ball_y_pixel": 20.0,
                "bbox_w": 20,
                "bbox_h": 20,
                "detection_confidence": 0.8,
            }
        )
    pd.DataFrame(pred_rows).to_parquet(run / "football_ball_detections.parquet", index=False)
    report = evaluate_ball_tracking(
        gt_csv=gt_path, run_dir=run, out_dir=tmp_path / "e", review_sample_csv=sample_path
    )
    assert report["status"] == "OK"
    assert "precision" in report
    assert "recall" in report
    assert "f1" in report
    assert report["manually_reviewed"] == 50
    assert report["automatic_reviewed_count"] == 0


def test_resume_first_unreviewed() -> None:
    gt = _make_gt(165)
    sample = _sample_from_gt(gt, 50)
    # review first 3 in sample order
    ordered = sample.sort_values("sample_order")
    for fid in ordered["frame_idx"].tolist()[:3]:
        idx = gt.index[gt["frame_index"] == int(fid)][0]
        gt.at[idx, "reviewed"] = True
        gt.at[idx, "ball_visible"] = False
    gt = apply_boolean_dtype(gt)
    i = first_unreviewed_sample_index(ordered, gt)
    assert int(ordered.iloc[i]["frame_idx"]) == int(ordered.iloc[3]["frame_idx"])


def test_review_sample_complete_helper() -> None:
    gt = _make_gt(165)
    sample = _sample_from_gt(gt, 50)
    ok, reason = review_sample_complete(sample, gt, sample_size=50)
    assert ok is False
    assert "unreviewed" in reason
