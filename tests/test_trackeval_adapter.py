from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from football_analytics.evaluation import (
    DEFAULT_TRACKEVAL_ROOT,
    canonical_gt_to_mot,
    canonical_tracks_to_mot,
    export_soccernet_gs_predictions,
    parse_soccernet_gs_predictions,
    run_trackeval,
)


def _gt_rows() -> list[dict[str, object]]:
    rows = []
    for frame in range(6):
        for track_id, x in ((1, 10.0 + frame), (2, 40.0 + frame)):
            rows.append(
                {
                    "frame_id": frame,
                    "track_id": track_id,
                    "object_type": "person",
                    "class_id": 0,
                    "bbox_x1": x,
                    "bbox_y1": 20.0,
                    "bbox_x2": x + 10.0,
                    "bbox_y2": 40.0,
                }
            )
    return rows


def _prediction_rows(scenario: str) -> list[dict[str, object]]:
    rows = []
    for row in _gt_rows():
        prediction = dict(row)
        prediction["tracking_confidence"] = 0.9
        if scenario == "id_switch" and int(row["frame_id"]) >= 3:
            prediction["track_id"] = 3 - int(row["track_id"])
        if scenario == "localization":
            prediction["bbox_x1"] = float(row["bbox_x1"]) + 2.0
            prediction["bbox_x2"] = float(row["bbox_x2"]) + 2.0
        if scenario == "fn" and row["track_id"] == 2 and int(row["frame_id"]) >= 3:
            continue
        rows.append(prediction)
    if scenario == "fp":
        for frame in range(6):
            rows.append(
                {
                    "frame_id": frame,
                    "track_id": 99,
                    "object_type": "person",
                    "class_id": 0,
                    "bbox_x1": 80.0,
                    "bbox_y1": 20.0,
                    "bbox_x2": 90.0,
                    "bbox_y2": 40.0,
                    "tracking_confidence": 0.7,
                }
            )
    return rows


def test_mot_conversion_from_parquet_and_csv(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.parquet"
    gt_path = tmp_path / "gt.csv"
    pq.write_table(pa.Table.from_pylist(_prediction_rows("perfect")), tracks_path)
    with gt_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_gt_rows()[0]))
        writer.writeheader()
        writer.writerows(_gt_rows())

    tracks_mot = canonical_tracks_to_mot(tracks_path, tmp_path / "tracker.txt")
    gt_mot = canonical_gt_to_mot(gt_path, tmp_path / "gt.txt")

    tracker_fields = tracks_mot.read_text().splitlines()[0].split(",")
    gt_fields = gt_mot.read_text().splitlines()[0].split(",")
    assert tracker_fields[:2] == ["1", "1"]
    assert tracker_fields[4:7] == ["10.0", "20.0", "0.9"]
    assert gt_fields[:2] == ["1", "1"]
    assert gt_fields[6:9] == ["1", "1", "1.0"]


def test_prediction_score_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tracking_confidence"):
        canonical_tracks_to_mot(_gt_rows(), tmp_path / "tracks.txt")
    with pytest.raises(ValueError, match="ground_truth is required"):
        run_trackeval(_prediction_rows("perfect"), None, tmp_path / "results")


@pytest.mark.skipif(not DEFAULT_TRACKEVAL_ROOT.is_dir(), reason="SoccerNet TrackEval absent")
def test_real_trackeval_synthetic_scenarios(tmp_path: Path) -> None:
    results = {}
    for scenario in ("perfect", "id_switch", "fp", "fn", "localization"):
        results[scenario] = run_trackeval(
            pa.Table.from_pylist(_prediction_rows(scenario)),
            pa.Table.from_pylist(_gt_rows()),
            tmp_path / scenario,
            sequence_name="synthetic",
            tracker_name=scenario,
        )

    perfect = results["perfect"].metrics
    assert perfect["HOTA"] == pytest.approx(1.0)
    assert perfect["MOTA"] == pytest.approx(1.0)
    assert perfect["IDF1"] == pytest.approx(1.0)
    assert perfect["CLR_FP"] == perfect["CLR_FN"] == perfect["IDSW"] == 0
    assert results["id_switch"].metrics["IDSW"] > 0
    assert results["id_switch"].metrics["IDF1"] < perfect["IDF1"]
    assert results["fp"].metrics["CLR_FP"] == 6
    assert results["fn"].metrics["CLR_FN"] == 3
    assert results["localization"].metrics["MOTP"] < perfect["MOTP"]

    for result in results.values():
        assert result.summary_json.is_file()
        assert result.summary_csv.is_file()
        assert (result.native_output_dir / "pedestrian_summary.txt").is_file()
        assert (result.native_output_dir / "pedestrian_detailed.csv").is_file()


def test_soccernet_gs_prediction_round_trip(tmp_path: Path) -> None:
    source = {
        "predictions": [
            {
                "image_id": "1000000001",
                "track_id": 7,
                "category_id": 1,
                "supercategory": "object",
                "bbox_image": {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0},
                "bbox_pitch": {
                    "x_bottom_left": 10.0,
                    "y_bottom_left": 20.0,
                    "x_bottom_middle": 11.0,
                    "y_bottom_middle": 20.0,
                    "x_bottom_right": 12.0,
                    "y_bottom_right": 20.0,
                },
                "attributes": {"role": "player", "team": "left", "jersey": "9"},
                "confidence": 0.87,
            }
        ]
    }
    table = parse_soccernet_gs_predictions(source)
    output = export_soccernet_gs_predictions(table, tmp_path / "predictions.json")
    exported = json.loads(output.read_text())

    assert exported == source


def test_soccernet_gs_never_fakes_confidence() -> None:
    source = {
        "predictions": [
            {
                "image_id": "frame",
                "track_id": 1,
                "category_id": 1,
                "supercategory": "object",
                "attributes": {},
            }
        ]
    }
    with pytest.raises(ValueError, match="confidence is required"):
        parse_soccernet_gs_predictions(source)
