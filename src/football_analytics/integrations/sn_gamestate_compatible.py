"""Clean-room SoccerNet Game State Reconstruction JSON exporter.

This module only implements the public prediction interchange shape.  It does
not import or depend on SoccerNet or TrackLab code.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

PITCH_KEYS = (
    "x_bottom_left",
    "y_bottom_left",
    "x_bottom_right",
    "y_bottom_right",
    "x_bottom_middle",
    "y_bottom_middle",
)


@dataclass(frozen=True)
class ExportInputs:
    detections: Path
    tracks: Path
    track_identities: Path
    calibration: Path
    game_state: Path
    jersey_predictions: Path | None = None


def _read_parquet(path: Path, required: set[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Input parquet does not exist: {path}")
    table = pq.read_table(path)
    missing = sorted(required.difference(table.column_names))
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    if table.num_rows == 0:
        raise ValueError(f"{path.name} must be nonempty")
    rows = table.to_pylist()
    frame_ids = [row["frame_id"] for row in rows]
    if any(frame_id is None for frame_id in frame_ids):
        raise ValueError(f"{path.name} has null frame_id")
    if frame_ids != sorted(frame_ids):
        raise ValueError(f"{path.name} is not ordered by frame_id")
    return rows


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_bbox(row: dict[str, Any]) -> None:
    values = [row[name] for name in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
    if not all(_finite_number(value) for value in values):
        raise ValueError(
            f"tracks.parquet has non-finite bbox at frame={row['frame_id']} "
            f"track={row['track_id']}"
        )
    x1, y1, x2, y2 = (float(value) for value in values)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"tracks.parquet has invalid bbox at frame={row['frame_id']} "
            f"track={row['track_id']}"
        )


def _unique_index(
    rows: Iterable[dict[str, Any]],
    key_names: tuple[str, ...],
    source_name: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[name] for name in key_names)
        if key in result:
            raise ValueError(f"{source_name} has duplicate key {key_names}={key}")
        result[key] = row
    return result


def _project(
    homography: list[list[float]], x_pixel: float, y_pixel: float
) -> tuple[float, float] | None:
    if (
        len(homography) != 3
        or any(len(row) != 3 for row in homography)
        or not all(_finite_number(value) for row in homography for value in row)
    ):
        return None
    denominator = (
        float(homography[2][0]) * x_pixel
        + float(homography[2][1]) * y_pixel
        + float(homography[2][2])
    )
    if not math.isfinite(denominator) or abs(denominator) < 1e-12:
        return None
    x_field = (
        float(homography[0][0]) * x_pixel
        + float(homography[0][1]) * y_pixel
        + float(homography[0][2])
    ) / denominator
    y_field = (
        float(homography[1][0]) * x_pixel
        + float(homography[1][1]) * y_pixel
        + float(homography[1][2])
    ) / denominator
    if not (_finite_number(x_field) and _finite_number(y_field)):
        return None
    return float(x_field), float(y_field)


def _pitch_bbox(
    track: dict[str, Any],
    game: dict[str, Any] | None,
    calibration: dict[str, Any] | None,
) -> dict[str, float | None]:
    null_pitch = {key: None for key in PITCH_KEYS}
    if not game or not calibration or game.get("valid") is not True:
        return null_pitch
    if calibration.get("valid") is not True:
        return null_pitch
    x_middle, y_middle = game.get("x_field"), game.get("y_field")
    if not (_finite_number(x_middle) and _finite_number(y_middle)):
        raise ValueError(
            "game_state.parquet valid row has non-finite pitch coordinates at "
            f"frame={track['frame_id']} track={track['track_id']}"
        )
    raw_homography = calibration.get("homography_json")
    if not isinstance(raw_homography, str):
        raise ValueError(
            f"calibration.parquet valid row lacks homography at frame={track['frame_id']}"
        )
    try:
        homography = json.loads(raw_homography)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"calibration.parquet has invalid homography at frame={track['frame_id']}"
        ) from exc
    left = _project(homography, float(track["bbox_x1"]), float(track["bbox_y2"]))
    right = _project(homography, float(track["bbox_x2"]), float(track["bbox_y2"]))
    if left is None or right is None:
        raise ValueError(
            f"Valid pitch row projects non-finite bbox corners at frame={track['frame_id']} "
            f"track={track['track_id']}"
        )
    return {
        "x_bottom_left": left[0],
        "y_bottom_left": left[1],
        "x_bottom_right": right[0],
        "y_bottom_right": right[1],
        "x_bottom_middle": float(x_middle),
        "y_bottom_middle": float(y_middle),
    }


def _normalize_jersey(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return str(int(value))
    text = str(value).strip()
    return text or None


def _jersey_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Jersey predictions do not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() != ".json":
        raise ValueError("Jersey predictions must be CSV or JSON")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
        return payload["predictions"]
    if isinstance(payload, dict):
        return [{"track_id": key, "jersey": value} for key, value in payload.items()]
    raise ValueError("Unsupported jersey JSON shape")


def _load_jerseys(
    path: Path | None,
) -> tuple[dict[tuple[int, int], str | None], dict[int, str | None]]:
    by_frame: dict[tuple[int, int], str | None] = {}
    by_track: dict[int, str | None] = {}
    if path is None:
        return by_frame, by_track
    for record in _jersey_records(path):
        if not isinstance(record, dict) or record.get("track_id") is None:
            raise ValueError("Each jersey prediction must contain track_id")
        try:
            track_id = int(record["track_id"])
            frame_id = int(record["frame_id"]) if record.get("frame_id") not in (None, "") else None
        except (TypeError, ValueError) as exc:
            raise ValueError("Jersey track_id/frame_id must be integers") from exc
        jersey_value = next(
            (
                record[name]
                for name in ("jersey", "jersey_number", "number")
                if name in record
            ),
            None,
        )
        jersey = _normalize_jersey(jersey_value)
        target: dict[Any, str | None]
        key: Any
        if frame_id is None:
            target, key = by_track, track_id
        else:
            target, key = by_frame, (frame_id, track_id)
        if key in target and target[key] != jersey:
            raise ValueError(f"Conflicting jersey predictions for {key}")
        target[key] = jersey
    return by_frame, by_track


def _official_team(team_id: Any) -> str | None:
    if team_id is None:
        return None
    return {
        "left": "left",
        "right": "right",
        "team_0": "left",
        "team_1": "right",
    }.get(str(team_id))


def export_predictions(
    inputs: ExportInputs,
    *,
    video_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Read canonical parquet artifacts and return a GSR-compatible payload."""

    _read_parquet(
        inputs.detections,
        {"frame_id", "detection_id", "detection_confidence"},
    )
    tracks = _read_parquet(
        inputs.tracks,
        {
            "frame_id",
            "timestamp_ms",
            "track_id",
            "detection_id",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "tracking_confidence",
        },
    )
    identities = _read_parquet(
        inputs.track_identities,
        {"frame_id", "track_id", "role", "team_id", "valid"},
    )
    calibrations = _read_parquet(
        inputs.calibration,
        {"frame_id", "valid", "homography_json"},
    )
    game_states = _read_parquet(
        inputs.game_state,
        {
            "match_id",
            "frame_id",
            "timestamp_ms",
            "track_id",
            "role",
            "team_id",
            "x_field",
            "y_field",
            "valid",
        },
    )

    identity_by_key = _unique_index(
        identities, ("frame_id", "track_id"), "track_identities.parquet"
    )
    calibration_by_frame = _unique_index(
        calibrations, ("frame_id",), "calibration.parquet"
    )
    game_by_key = _unique_index(game_states, ("frame_id", "track_id"), "game_state.parquet")
    jersey_by_frame, jersey_by_track = _load_jerseys(inputs.jersey_predictions)

    match_ids = {str(row["match_id"]) for row in game_states if row.get("match_id") is not None}
    if video_id is None:
        if len(match_ids) != 1:
            raise ValueError("game_state.parquet must contain one non-null match_id or set video_id")
        resolved_video_id = next(iter(match_ids))
    else:
        resolved_video_id = str(video_id)

    predictions: list[dict[str, Any]] = []
    for track in tracks:
        _validate_bbox(track)
        frame_id = int(track["frame_id"])
        track_id = int(track["track_id"])
        key = (frame_id, track_id)
        game = game_by_key.get(key)
        identity = identity_by_key.get(key)
        source = game if game is not None else identity
        role = source.get("role") if source is not None else None
        team = _official_team(source.get("team_id")) if source is not None else None
        confidence = track.get("tracking_confidence")
        if confidence is not None and not _finite_number(confidence):
            raise ValueError(f"Non-finite tracking confidence at frame={frame_id} track={track_id}")
        timestamp_ms = track.get("timestamp_ms")
        if timestamp_ms is not None and not _finite_number(timestamp_ms):
            raise ValueError(f"Non-finite timestamp at frame={frame_id} track={track_id}")
        jersey = jersey_by_frame.get(key, jersey_by_track.get(track_id))
        prediction = {
            "bbox_pitch": _pitch_bbox(track, game, calibration_by_frame.get((frame_id,))),
            "bbox_image": {
                "x": float(track["bbox_x1"]),
                "y": float(track["bbox_y1"]),
                "w": float(track["bbox_x2"]) - float(track["bbox_x1"]),
                "h": float(track["bbox_y2"]) - float(track["bbox_y1"]),
            },
            "category_id": 1.0,
            "image_id": str(frame_id),
            "video_id": resolved_video_id,
            "track_id": track_id,
            "supercategory": "object",
            "attributes": {"role": role, "jersey": jersey, "team": team},
            "id": str(len(predictions)),
            "confidence": float(confidence) if confidence is not None else None,
            "frame_id": frame_id,
            "timestamp_ms": float(timestamp_ms) if timestamp_ms is not None else None,
        }
        predictions.append(prediction)

    if not predictions:
        raise ValueError("Export produced no predictions")
    frame_ids = [prediction["frame_id"] for prediction in predictions]
    if frame_ids != sorted(frame_ids):
        raise ValueError("Export predictions are not frame ordered")
    return {"predictions": predictions}


def write_predictions(
    inputs: ExportInputs,
    output: Path,
    *,
    video_id: str | None = None,
) -> dict[str, Any]:
    """Export and atomically write JSON; return a compact result summary."""

    payload = export_predictions(inputs, video_id=video_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, separators=(",", ":"))
        handle.write("\n")
    temporary.replace(output)
    predictions = payload["predictions"]
    return {
        "output": str(output),
        "predictions": len(predictions),
        "first_frame_id": predictions[0]["frame_id"],
        "last_frame_id": predictions[-1]["frame_id"],
        "pitch_valid": sum(
            prediction["bbox_pitch"]["x_bottom_middle"] is not None
            for prediction in predictions
        ),
    }
