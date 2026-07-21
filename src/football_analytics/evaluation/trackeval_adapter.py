"""Clean-room adapters from football-analytics artifacts to SoccerNet TrackEval."""

from __future__ import annotations

import csv
import importlib
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

DEFAULT_TRACKEVAL_ROOT = Path("/home/ahmet/projects/soccernet/sn-trackeval")

_BOX_COLUMNS = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
_BASE_COLUMNS = ("frame_id", "track_id", *_BOX_COLUMNS)


@dataclass(frozen=True)
class EvaluationResult:
    """Locations and flattened combined-sequence metrics from one evaluation."""

    metrics: dict[str, int | float]
    summary_json: Path
    summary_csv: Path
    native_output_dir: Path


def _records(source: Any) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pq.read_table(path).to_pylist()
        if suffix == ".csv":
            return pa_csv.read_csv(path).to_pylist()
        raise ValueError(f"Unsupported table format {suffix!r}; expected .parquet or .csv")
    if isinstance(source, pa.Table):
        return source.to_pylist()
    if hasattr(source, "to_dict"):
        try:
            records = source.to_dict(orient="records")
        except TypeError:
            records = source.to_dict("records")
        if isinstance(records, list):
            return [dict(row) for row in records]
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        return [dict(row) for row in source]
    raise TypeError("Expected a parquet/CSV path, PyArrow table, DataFrame, or row sequence")


def _require_columns(rows: Sequence[Mapping[str, Any]], columns: Iterable[str], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} is empty")
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _finite_number(value: Any, name: str, row_number: int) -> float:
    if value is None:
        raise ValueError(f"row {row_number}: {name} must not be null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"row {row_number}: {name} must be finite")
    return number


def _mot_rows(source: Any, *, is_gt: bool, frame_offset: int) -> list[list[int | float]]:
    rows = _records(source)
    label = "ground truth" if is_gt else "tracks"
    required = _BASE_COLUMNS if is_gt else (*_BASE_COLUMNS, "tracking_confidence")
    _require_columns(rows, required, label)

    mot_rows: list[list[int | float]] = []
    seen: set[tuple[int, int]] = set()
    for index, row in enumerate(rows, start=1):
        if row.get("object_type", "person") != "person":
            continue
        frame = int(_finite_number(row["frame_id"], "frame_id", index)) + frame_offset
        track_id = int(_finite_number(row["track_id"], "track_id", index))
        if frame < 1:
            raise ValueError(f"row {index}: converted MOT frame must be at least 1")
        if track_id < 0:
            raise ValueError(f"row {index}: track_id must be non-negative")
        key = (frame, track_id)
        if key in seen:
            raise ValueError(f"row {index}: duplicate frame/track identity {key}")
        seen.add(key)

        x1, y1, x2, y2 = (
            _finite_number(row[column], column, index) for column in _BOX_COLUMNS
        )
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            raise ValueError(f"row {index}: bounding box must have positive width and height")

        if is_gt:
            # MOT GT column 7 is an inclusion mark, not a detector confidence.
            mark = int(row.get("gt_mark", row.get("mark", 1)))
            # Canonical class_id is model-specific (commonly 0 for person);
            # MOTChallenge reserves class 1 for pedestrians.
            class_id = 1
            visibility = _finite_number(row.get("visibility", 1.0), "visibility", index)
            mot_rows.append([frame, track_id, x1, y1, width, height, mark, class_id, visibility])
        else:
            score = _finite_number(row["tracking_confidence"], "tracking_confidence", index)
            mot_rows.append([frame, track_id, x1, y1, width, height, score, -1, -1, -1])

    if not mot_rows:
        raise ValueError(f"{label} contains no person rows")
    return sorted(mot_rows, key=lambda row: (row[0], row[1]))


def _write_mot(rows: Sequence[Sequence[int | float]], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    return path


def canonical_tracks_to_mot(
    tracks: Any, destination: str | Path, *, frame_offset: int = 1
) -> Path:
    """Write canonical tracks parquet/CSV/DataFrame as MOTChallenge predictions."""

    return _write_mot(
        _mot_rows(tracks, is_gt=False, frame_offset=frame_offset),
        destination,
    )


def canonical_gt_to_mot(
    ground_truth: Any, destination: str | Path, *, frame_offset: int = 1
) -> Path:
    """Write canonical GT parquet/CSV/DataFrame in MOTChallenge GT format."""

    return _write_mot(
        _mot_rows(ground_truth, is_gt=True, frame_offset=frame_offset),
        destination,
    )


def _load_trackeval(trackeval_root: str | Path) -> Any:
    root = Path(trackeval_root).resolve()
    if not (root / "trackeval" / "__init__.py").is_file():
        raise FileNotFoundError(f"TrackEval package not found under {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("trackeval")
    module_file = Path(module.__file__).resolve()
    if root not in module_file.parents:
        raise RuntimeError(f"Imported TrackEval from {module_file}, expected {root}")
    return module


def _scalar(value: Any) -> int | float:
    if hasattr(value, "shape") and getattr(value, "shape", ()) != ():
        value = value.mean()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, int):
        return value
    return float(value)


def _flatten_metrics(combined: Mapping[str, Mapping[str, Any]]) -> dict[str, int | float]:
    flattened: dict[str, int | float] = {}
    for metric_name in ("HOTA", "CLEAR", "Identity", "Count"):
        for field, value in combined[metric_name].items():
            if field in flattened:
                flattened[f"{metric_name}.{field}"] = _scalar(value)
            else:
                flattened[field] = _scalar(value)
    return flattened


def _write_summary(
    metrics: Mapping[str, int | float], output_dir: Path, sequence_name: str, tracker_name: str
) -> tuple[Path, Path]:
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    payload = {"sequence": sequence_name, "tracker": tracker_name, "metrics": dict(metrics)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "tracker", *metrics.keys()])
        writer.writeheader()
        writer.writerow({"sequence": sequence_name, "tracker": tracker_name, **metrics})
    return json_path, csv_path


def run_trackeval(
    tracks: Any,
    ground_truth: Any,
    output_dir: str | Path,
    *,
    sequence_name: str = "sequence",
    tracker_name: str = "tracker",
    trackeval_root: str | Path = DEFAULT_TRACKEVAL_ROOT,
    frame_offset: int = 1,
) -> EvaluationResult:
    """Run real SoccerNet TrackEval HOTA, CLEAR, Identity, and Count metrics.

    Ground truth is mandatory. Tracker confidence is mandatory and is never
    fabricated. The output directory receives summary JSON/CSV plus TrackEval's
    native ``pedestrian_summary.txt`` and ``pedestrian_detailed.csv`` files.
    """

    if ground_truth is None:
        raise ValueError("ground_truth is required; predictions cannot be evaluated alone")
    if not sequence_name or "/" in sequence_name or "\\" in sequence_name:
        raise ValueError("sequence_name must be a non-empty path-safe name")
    if not tracker_name or "/" in tracker_name or "\\" in tracker_name:
        raise ValueError("tracker_name must be a non-empty path-safe name")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trackeval = _load_trackeval(trackeval_root)

    with tempfile.TemporaryDirectory(prefix="football-analytics-trackeval-") as temporary:
        root = Path(temporary)
        gt_file = root / "gt" / sequence_name / "gt" / "gt.txt"
        tracker_file = root / "trackers" / tracker_name / "data" / f"{sequence_name}.txt"
        gt_rows = _mot_rows(ground_truth, is_gt=True, frame_offset=frame_offset)
        tracker_rows = _mot_rows(tracks, is_gt=False, frame_offset=frame_offset)
        _write_mot(gt_rows, gt_file)
        _write_mot(tracker_rows, tracker_file)
        sequence_length = max(int(row[0]) for row in [*gt_rows, *tracker_rows])

        evaluator = trackeval.Evaluator(
            {
                "USE_PARALLEL": False,
                "BREAK_ON_ERROR": True,
                "PRINT_RESULTS": False,
                "PRINT_CONFIG": False,
                "TIME_PROGRESS": False,
                "OUTPUT_SUMMARY": True,
                "OUTPUT_DETAILED": True,
                "PLOT_CURVES": False,
            }
        )
        dataset = trackeval.datasets.MotChallenge2DBox(
            {
                "GT_FOLDER": str(root / "gt"),
                "TRACKERS_FOLDER": str(root / "trackers"),
                "OUTPUT_FOLDER": str(root / "native"),
                "TRACKERS_TO_EVAL": [tracker_name],
                "CLASSES_TO_EVAL": ["pedestrian"],
                "BENCHMARK": "MOT17",
                "SPLIT_TO_EVAL": "train",
                "SEQ_INFO": {sequence_name: sequence_length},
                "SKIP_SPLIT_FOL": True,
                "PRINT_CONFIG": False,
            }
        )
        metric_config = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
        metrics = [
            trackeval.metrics.HOTA(metric_config),
            trackeval.metrics.CLEAR(metric_config),
            trackeval.metrics.Identity(metric_config),
        ]
        raw_results, messages = evaluator.evaluate([dataset], metrics)
        if messages["MotChallenge2DBox"][tracker_name] != "Success":
            raise RuntimeError(messages["MotChallenge2DBox"][tracker_name])
        combined = raw_results["MotChallenge2DBox"][tracker_name]["COMBINED_SEQ"]["pedestrian"]
        flattened = _flatten_metrics(combined)

        native_source = root / "native" / tracker_name
        native_output = output / "trackeval"
        if native_output.exists():
            shutil.rmtree(native_output)
        shutil.copytree(native_source, native_output)

    json_path, csv_path = _write_summary(flattened, output, sequence_name, tracker_name)
    return EvaluationResult(flattened, json_path, csv_path, native_output)


def _load_json(source: Any) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("SoccerNet GS JSON root must be an object")
    return data


def parse_soccernet_gs_predictions(source: Any) -> pa.Table:
    """Parse the prediction schema consumed by ``datasets/soccernet_gs.py``."""

    data = _load_json(source)
    predictions = data.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("SoccerNet GS JSON must contain a predictions list")
    rows: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions, start=1):
        if "confidence" not in prediction or prediction["confidence"] is None:
            raise ValueError(f"prediction {index}: confidence is required")
        bbox_image = prediction.get("bbox_image") or {}
        bbox_pitch = prediction.get("bbox_pitch") or {}
        attributes = prediction.get("attributes") or {}
        rows.append(
            {
                "image_id": prediction.get("image_id"),
                "track_id": prediction.get("track_id"),
                "category_id": prediction.get("category_id"),
                "supercategory": prediction.get("supercategory"),
                "confidence": prediction["confidence"],
                "bbox_x": bbox_image.get("x"),
                "bbox_y": bbox_image.get("y"),
                "bbox_w": bbox_image.get("w"),
                "bbox_h": bbox_image.get("h"),
                "pitch_x_bottom_left": bbox_pitch.get("x_bottom_left"),
                "pitch_y_bottom_left": bbox_pitch.get("y_bottom_left"),
                "pitch_x_bottom_middle": bbox_pitch.get("x_bottom_middle"),
                "pitch_y_bottom_middle": bbox_pitch.get("y_bottom_middle"),
                "pitch_x_bottom_right": bbox_pitch.get("x_bottom_right"),
                "pitch_y_bottom_right": bbox_pitch.get("y_bottom_right"),
                "role": attributes.get("role"),
                "team": attributes.get("team"),
                "jersey": attributes.get("jersey"),
            }
        )
    return pa.Table.from_pylist(rows)


def export_soccernet_gs_predictions(rows: Any, destination: str | Path) -> Path:
    """Export flat rows to the SoccerNet GS tracker prediction JSON schema."""

    records = _records(rows)
    required = ("image_id", "track_id", "category_id", "supercategory", "confidence")
    _require_columns(records, required, "SoccerNet GS predictions")
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(records, start=1):
        confidence = _finite_number(row["confidence"], "confidence", index)
        predictions.append(
            {
                "image_id": row["image_id"],
                "track_id": int(row["track_id"]),
                "category_id": int(row["category_id"]),
                "supercategory": row["supercategory"],
                "bbox_image": {
                    "x": row.get("bbox_x"),
                    "y": row.get("bbox_y"),
                    "w": row.get("bbox_w"),
                    "h": row.get("bbox_h"),
                },
                "bbox_pitch": {
                    "x_bottom_left": row.get("pitch_x_bottom_left"),
                    "y_bottom_left": row.get("pitch_y_bottom_left"),
                    "x_bottom_middle": row.get("pitch_x_bottom_middle"),
                    "y_bottom_middle": row.get("pitch_y_bottom_middle"),
                    "x_bottom_right": row.get("pitch_x_bottom_right"),
                    "y_bottom_right": row.get("pitch_y_bottom_right"),
                },
                "attributes": {
                    "role": row.get("role"),
                    "team": row.get("team"),
                    "jersey": row.get("jersey"),
                },
                "confidence": confidence,
            }
        )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"predictions": predictions}, indent=2) + "\n", encoding="utf-8")
    return path
