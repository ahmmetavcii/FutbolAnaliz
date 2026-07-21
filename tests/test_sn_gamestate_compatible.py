from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from football_analytics.integrations.sn_gamestate_compatible import (
    ExportInputs,
    PITCH_KEYS,
    export_predictions,
    write_predictions,
)


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _inputs(tmp_path: Path) -> ExportInputs:
    detections = _write(
        tmp_path / "detections.parquet",
        [
            {"frame_id": 1, "detection_id": "d1", "detection_confidence": 0.91},
            {"frame_id": 2, "detection_id": "d2", "detection_confidence": 0.81},
        ],
    )
    tracks = _write(
        tmp_path / "tracks.parquet",
        [
            {
                "frame_id": 1,
                "timestamp_ms": 40.0,
                "track_id": 7,
                "detection_id": "d1",
                "bbox_x1": 10.0,
                "bbox_y1": 20.0,
                "bbox_x2": 30.0,
                "bbox_y2": 40.0,
                "tracking_confidence": 0.8,
            },
            {
                "frame_id": 2,
                "timestamp_ms": 80.0,
                "track_id": 8,
                "detection_id": "d2",
                "bbox_x1": 11.0,
                "bbox_y1": 21.0,
                "bbox_x2": 31.0,
                "bbox_y2": 41.0,
                "tracking_confidence": 0.7,
            },
        ],
    )
    identities = _write(
        tmp_path / "track_identities.parquet",
        [
            {
                "frame_id": 1,
                "track_id": 7,
                "role": "player",
                "team_id": "team_0",
                "valid": True,
            },
            {
                "frame_id": 2,
                "track_id": 8,
                "role": "unknown",
                "team_id": "unmapped",
                "valid": False,
            },
        ],
    )
    calibration = _write(
        tmp_path / "calibration.parquet",
        [
            {
                "frame_id": 1,
                "valid": True,
                "homography_json": json.dumps(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                ),
            },
            {"frame_id": 2, "valid": False, "homography_json": None},
        ],
    )
    game_state = _write(
        tmp_path / "game_state.parquet",
        [
            {
                "match_id": "video-a",
                "frame_id": 1,
                "timestamp_ms": 40.0,
                "track_id": 7,
                "role": "player",
                "team_id": "team_0",
                "x_field": 20.0,
                "y_field": 40.0,
                "valid": True,
            },
            {
                "match_id": "video-a",
                "frame_id": 2,
                "timestamp_ms": 80.0,
                "track_id": 8,
                "role": "unknown",
                "team_id": None,
                "x_field": None,
                "y_field": None,
                "valid": False,
            },
        ],
    )
    return ExportInputs(detections, tracks, identities, calibration, game_state)


def test_export_exact_fields_geometry_and_nulls(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    jersey_path = tmp_path / "jerseys.csv"
    jersey_path.write_text("track_id,jersey\n7,9\n", encoding="utf-8")
    inputs = ExportInputs(
        inputs.detections,
        inputs.tracks,
        inputs.track_identities,
        inputs.calibration,
        inputs.game_state,
        jersey_path,
    )

    payload = export_predictions(inputs)
    predictions = payload["predictions"]
    assert len(predictions) == 2
    assert list(predictions[0]) == [
        "bbox_pitch",
        "bbox_image",
        "category_id",
        "image_id",
        "video_id",
        "track_id",
        "supercategory",
        "attributes",
        "id",
        "confidence",
        "frame_id",
        "timestamp_ms",
    ]
    assert predictions[0]["bbox_image"] == {"x": 10.0, "y": 20.0, "w": 20.0, "h": 20.0}
    assert predictions[0]["bbox_pitch"] == {
        "x_bottom_left": 10.0,
        "y_bottom_left": 40.0,
        "x_bottom_right": 30.0,
        "y_bottom_right": 40.0,
        "x_bottom_middle": 20.0,
        "y_bottom_middle": 40.0,
    }
    assert predictions[0]["attributes"] == {
        "role": "player",
        "jersey": "9",
        "team": "left",
    }
    assert predictions[0]["video_id"] == "video-a"
    assert predictions[0]["image_id"] == "1"
    assert predictions[0]["confidence"] == 0.8
    assert predictions[1]["attributes"]["jersey"] is None
    assert predictions[1]["attributes"]["team"] is None
    assert predictions[1]["bbox_pitch"] == {key: None for key in PITCH_KEYS}


def test_write_predictions_is_valid_strict_json(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "predictions.json"
    summary = write_predictions(_inputs(tmp_path), output, video_id="override")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert summary == {
        "output": str(output),
        "predictions": 2,
        "first_frame_id": 1,
        "last_frame_id": 2,
        "pitch_valid": 1,
    }
    assert payload["predictions"][0]["video_id"] == "override"


def test_rejects_invalid_bbox(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    table = pq.read_table(inputs.tracks)
    rows = table.to_pylist()
    rows[0]["bbox_x2"] = rows[0]["bbox_x1"]
    _write(inputs.tracks, rows)

    with pytest.raises(ValueError, match="invalid bbox"):
        export_predictions(inputs)


def test_rejects_nonfinite_pitch_on_valid_row(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    rows = pq.read_table(inputs.game_state).to_pylist()
    rows[0]["x_field"] = float("nan")
    _write(inputs.game_state, rows)

    with pytest.raises(ValueError, match="valid row has non-finite pitch"):
        export_predictions(inputs)


def test_rejects_unordered_frames(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    rows = list(reversed(pq.read_table(inputs.tracks).to_pylist()))
    _write(inputs.tracks, rows)

    with pytest.raises(ValueError, match="not ordered by frame_id"):
        export_predictions(inputs)
