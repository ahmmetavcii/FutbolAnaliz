"""Clean-room planar calibration adapter for sn-calibration-style consumers.

This module does not import or reproduce code from sn-calibration.  It defines a
small, JSON-safe interoperability contract from the observable coordinate
conventions: a 105 x 68 metre pitch centred at the halfway spot and a
pitch-to-image homography.

PnLCalib's canonical homography maps image pixels to corner-origin pitch metres.
The adapter retains that matrix and additionally emits both directions in the
centred pitch convention.  A planar homography cannot uniquely recover a full
camera model (intrinsics, extrinsics, or distortion), so none is fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
CONTRACT_NAME = "sn-calibration-compatible-planar"
CONTRACT_VERSION = "1.0.0"
ORIENTATIONS = frozenset(("left_to_right", "right_to_left"))
_DETERMINANT_EPSILON = 1e-12
_INVERSE_TOLERANCE = 1e-8
__all__ = [
    "adapt_calibration",
    "calibration_to_artifact",
    "calibration_to_json",
    "load_canonical_calibration",
    "validate_artifact",
    "validate_json_artifact",
    "write_calibration_artifact",
    "write_json_artifact",
]


def load_canonical_calibration(
    source: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """Load a canonical calibration artifact without mutating the input."""
    if isinstance(source, pd.DataFrame):
        return source.copy(deep=True)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"calibration artifact not found: {path}")
    return pd.read_parquet(path)


def calibration_to_artifact(
    source: str | Path | pd.DataFrame,
    *,
    frame_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Convert canonical rows to the JSON-safe compatibility contract.

    ``frame_ids=None`` converts every row.  Explicit frame IDs retain caller
    order, must be unique, and must all exist in the source.
    """
    frame = load_canonical_calibration(source)
    _require_columns(frame)
    selected = _select_rows(frame, frame_ids)
    records = [_adapt_row(row) for row in selected.to_dict(orient="records")]
    artifact: dict[str, Any] = {
        "contract": CONTRACT_NAME,
        "contract_version": CONTRACT_VERSION,
        "pitch": {
            "length_m": PITCH_LENGTH_M,
            "width_m": PITCH_WIDTH_M,
            "origin": "center_mark",
            "x_range_m": [-PITCH_LENGTH_M / 2.0, PITCH_LENGTH_M / 2.0],
            "y_range_m": [-PITCH_WIDTH_M / 2.0, PITCH_WIDTH_M / 2.0],
        },
        "homography_conventions": {
            "canonical_image_to_pitch_105x68": (
                "image_pixels_to_corner_origin_pitch_metres"
            ),
            "image_to_pitch": "image_pixels_to_centered_pitch_metres",
            "inverse_homography": "centered_pitch_metres_to_image_pixels",
        },
        "frames": records,
    }
    artifact["validation"] = validate_artifact(artifact)
    return artifact


def calibration_to_json(
    source: str | Path | pd.DataFrame,
    *,
    frame_ids: Iterable[int] | None = None,
    indent: int | None = 2,
) -> str:
    """Return a strict JSON document for the selected calibration rows."""
    return json.dumps(
        calibration_to_artifact(source, frame_ids=frame_ids),
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
    )


def write_calibration_artifact(
    source: str | Path | pd.DataFrame,
    output_path: str | Path,
    *,
    frame_ids: Iterable[int] | None = None,
) -> Path:
    """Build and atomically write a strict JSON artifact."""
    artifact = calibration_to_artifact(source, frame_ids=frame_ids)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
    return path


def validate_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Validate frame states, finite matrices, and inverse round trips.

    Returns a JSON-safe report instead of raising, allowing callers to inspect
    bad frames.  ``contract_valid`` means every frame obeys the contract; it
    does not mean every calibration is usable.
    """
    errors: list[dict[str, Any]] = []
    frames = artifact.get("frames")
    if not isinstance(frames, list):
        return {
            "contract_valid": False,
            "frame_count": 0,
            "valid_calibration_count": 0,
            "invalid_calibration_count": 0,
            "errors": [{"frame_id": None, "reason": "frames must be a list"}],
        }

    seen: set[int] = set()
    valid_count = 0
    for record in frames:
        if not isinstance(record, Mapping):
            errors.append({"frame_id": None, "reason": "frame must be an object"})
            continue
        frame_id = _int_or_none(record.get("frame_id"))
        if frame_id is None:
            errors.append({"frame_id": None, "reason": "frame_id must be an integer"})
            continue
        if frame_id in seen:
            errors.append({"frame_id": frame_id, "reason": "duplicate frame_id"})
        seen.add(frame_id)
        errors.extend(_validate_record(record, frame_id))
        if record.get("valid") is True:
            valid_count += 1

    return {
        "contract_valid": not errors,
        "frame_count": len(frames),
        "valid_calibration_count": valid_count,
        "invalid_calibration_count": len(frames) - valid_count,
        "errors": errors,
    }


def _adapt_row(row: Mapping[str, Any]) -> dict[str, Any]:
    frame_id = _required_int(row.get("frame_id"), "frame_id")
    confidence = _finite_float(row.get("confidence"))
    reprojection_error = _finite_float(row.get("reprojection_error"))
    coverage = _finite_float(row.get("visible_pitch_coverage"))
    orientation = str(row.get("orientation") or "")
    source_valid = _strict_bool(row.get("valid"))
    reasons: list[str] = []

    if not source_valid:
        reasons.append(_text_or_none(row.get("invalid_reason")) or "source calibration invalid")
    if orientation not in ORIENTATIONS:
        reasons.append("orientation is missing or unsupported")
    if confidence is None or not 0.0 <= confidence <= 1.0:
        reasons.append("confidence must be finite and within [0, 1]")
    if source_valid and (reprojection_error is None or reprojection_error < 0.0):
        reasons.append("reprojection_error must be finite and non-negative")
    if source_valid and (coverage is None or not 0.0 <= coverage <= 1.0):
        reasons.append("visible_pitch_coverage must be finite and within [0, 1]")
    if not _same_number(row.get("pitch_length_m"), PITCH_LENGTH_M):
        reasons.append("pitch_length_m must be 105")
    if not _same_number(row.get("pitch_width_m"), PITCH_WIDTH_M):
        reasons.append("pitch_width_m must be 68")

    canonical = _parse_homography(row.get("homography_json"))
    centered: NDArray[np.float64] | None = None
    inverse: NDArray[np.float64] | None = None
    if source_valid:
        if canonical is None:
            reasons.append("canonical homography is malformed, non-finite, or non-invertible")
        else:
            corner_to_center = np.array(
                [
                    [1.0, 0.0, -PITCH_LENGTH_M / 2.0],
                    [0.0, 1.0, -PITCH_WIDTH_M / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            centered = _normalize_homography(corner_to_center @ canonical)
            inverse = _invert_homography(centered)
            if centered is None or inverse is None:
                reasons.append("transformed homography is non-finite or non-invertible")
                centered = None
                inverse = None
            elif not _inverse_round_trip(centered, inverse):
                reasons.append("homography inverse failed round-trip validation")
                centered = None
                inverse = None

    valid = source_valid and not reasons
    if not valid:
        canonical = None
        centered = None
        inverse = None

    return {
        "frame_id": frame_id,
        "timestamp_ms": _finite_float(row.get("timestamp_ms")),
        "run_id": _text_or_none(row.get("run_id")),
        "match_id": _text_or_none(row.get("match_id")),
        "source_provider": _text_or_none(row.get("provider"))
        or _text_or_none(row.get("source_method")),
        "pitch_length_m": PITCH_LENGTH_M,
        "pitch_width_m": PITCH_WIDTH_M,
        "orientation": orientation or None,
        "confidence": confidence,
        "reprojection_error": reprojection_error,
        "visible_pitch_coverage": coverage,
        "canonical_homography_image_to_pitch_105x68": _matrix_list(canonical),
        "homography": _matrix_list(centered),
        "inverse_homography": _matrix_list(inverse),
        "valid": valid,
        "invalid_reason": "; ".join(dict.fromkeys(reasons)) if reasons else None,
    }


def _validate_record(record: Mapping[str, Any], frame_id: int) -> list[dict[str, Any]]:
    problems: list[str] = []
    valid = record.get("valid")
    if type(valid) is not bool:
        problems.append("valid must be a boolean")
        valid = False
    orientation = record.get("orientation")
    if orientation not in ORIENTATIONS:
        problems.append("orientation is missing or unsupported")
    confidence = _finite_float(record.get("confidence"))
    if confidence is None or not 0.0 <= confidence <= 1.0:
        problems.append("confidence must be finite and within [0, 1]")

    matrix_keys = (
        "canonical_homography_image_to_pitch_105x68",
        "homography",
        "inverse_homography",
    )
    if valid:
        matrices = [_parse_homography(record.get(key)) for key in matrix_keys]
        if any(matrix is None for matrix in matrices):
            problems.append("valid frame must contain three finite invertible homographies")
        else:
            canonical, centered, inverse = matrices
            assert canonical is not None and centered is not None and inverse is not None
            translation = np.array(
                [
                    [1.0, 0.0, -PITCH_LENGTH_M / 2.0],
                    [0.0, 1.0, -PITCH_WIDTH_M / 2.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            expected = _normalize_homography(translation @ canonical)
            if expected is None or not np.allclose(
                expected, centered, rtol=_INVERSE_TOLERANCE, atol=_INVERSE_TOLERANCE
            ):
                problems.append("centered homography does not match 105x68 translation")
            if not _inverse_round_trip(centered, inverse):
                problems.append("inverse homography fails round-trip validation")
        error = _finite_float(record.get("reprojection_error"))
        coverage = _finite_float(record.get("visible_pitch_coverage"))
        if error is None or error < 0.0:
            problems.append("valid frame has invalid reprojection_error")
        if coverage is None or not 0.0 <= coverage <= 1.0:
            problems.append("valid frame has invalid visible_pitch_coverage")
        if record.get("invalid_reason") is not None:
            problems.append("valid frame cannot have invalid_reason")
    else:
        if not _text_or_none(record.get("invalid_reason")):
            problems.append("invalid frame must have invalid_reason")
        if any(record.get(key) is not None for key in matrix_keys):
            problems.append("invalid frame homographies must be null")

    return [{"frame_id": frame_id, "reason": reason} for reason in problems]


def _require_columns(frame: pd.DataFrame) -> None:
    required = {
        "frame_id",
        "valid",
        "homography_json",
        "orientation",
        "pitch_length_m",
        "pitch_width_m",
        "confidence",
        "reprojection_error",
        "visible_pitch_coverage",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"canonical calibration missing columns: {missing}")
    if frame["frame_id"].duplicated().any():
        duplicates = frame.loc[frame["frame_id"].duplicated(), "frame_id"].tolist()
        raise ValueError(f"canonical calibration has duplicate frame IDs: {duplicates}")


def _select_rows(
    frame: pd.DataFrame, frame_ids: Iterable[int] | None
) -> pd.DataFrame:
    if frame_ids is None:
        return frame
    requested = [_required_int(value, "frame_ids") for value in frame_ids]
    if len(set(requested)) != len(requested):
        raise ValueError("frame_ids must be unique")
    indexed = frame.set_index("frame_id", drop=False)
    missing = [value for value in requested if value not in indexed.index]
    if missing:
        raise KeyError(f"frame IDs not found: {missing}")
    return indexed.loc[requested].reset_index(drop=True)


def _parse_homography(raw: Any) -> NDArray[np.float64] | None:
    if raw is None or _is_missing(raw):
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    try:
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return _normalize_homography(matrix)


def _normalize_homography(
    matrix: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None
    determinant = float(np.linalg.det(matrix))
    scale = float(matrix[2, 2])
    if abs(determinant) <= _DETERMINANT_EPSILON or abs(scale) <= _DETERMINANT_EPSILON:
        return None
    normalized = matrix / scale
    return normalized if np.all(np.isfinite(normalized)) else None


def _invert_homography(
    matrix: NDArray[np.float64] | None,
) -> NDArray[np.float64] | None:
    if matrix is None:
        return None
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    return _normalize_homography(inverse)


def _inverse_round_trip(
    matrix: NDArray[np.float64], inverse: NDArray[np.float64]
) -> bool:
    identity = matrix @ inverse
    if abs(float(identity[2, 2])) <= _DETERMINANT_EPSILON:
        return False
    identity = identity / identity[2, 2]
    return bool(
        np.all(np.isfinite(identity))
        and np.allclose(
            identity,
            np.eye(3),
            rtol=_INVERSE_TOLERANCE,
            atol=_INVERSE_TOLERANCE,
        )
    )


def _matrix_list(matrix: NDArray[np.float64] | None) -> list[list[float]] | None:
    return matrix.tolist() if matrix is not None else None


def _finite_float(raw: Any) -> float | None:
    if raw is None or _is_missing(raw):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _same_number(raw: Any, expected: float) -> bool:
    value = _finite_float(raw)
    return value is not None and bool(np.isclose(value, expected, atol=1e-9, rtol=0.0))


def _strict_bool(raw: Any) -> bool:
    return isinstance(raw, (bool, np.bool_)) and bool(raw)


def _required_int(raw: Any, name: str) -> int:
    value = _int_or_none(raw)
    if value is None:
        raise ValueError(f"{name} must contain integers")
    return value


def _int_or_none(raw: Any) -> int | None:
    if isinstance(raw, (bool, np.bool_)) or raw is None:
        return None
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    if isinstance(raw, (float, np.floating)) and np.isfinite(raw) and raw.is_integer():
        return int(raw)
    return None


def _text_or_none(raw: Any) -> str | None:
    if raw is None or _is_missing(raw):
        return None
    value = str(raw).strip()
    return value or None


def _is_missing(raw: Any) -> bool:
    try:
        missing = pd.isna(raw)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


# Concise aliases for callers that treat this module as a JSON adapter boundary.
adapt_calibration = calibration_to_artifact
validate_json_artifact = validate_artifact
write_json_artifact = write_calibration_artifact
