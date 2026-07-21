from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_analytics.integrations.sn_calibration_compatible import (
    calibration_to_json,
    calibration_to_artifact,
    validate_artifact,
    write_calibration_artifact,
)

RUN_CALIBRATION = Path(
    "/home/ahmet/workspace/runs/run_20260718_033654_77a8a7/calibration.parquet"
)


def _canonical_frame(*, homography: object | None = None, valid: bool = True) -> pd.DataFrame:
    matrix = homography if homography is not None else np.eye(3).tolist()
    return pd.DataFrame(
        [
            {
                "frame_id": 7,
                "timestamp_ms": 280.0,
                "run_id": "run-test",
                "match_id": "match-test",
                "source_method": "pnlcalib",
                "provider": "pnlcalib",
                "valid": valid,
                "homography_json": json.dumps(matrix) if matrix is not None else None,
                "orientation": "left_to_right",
                "pitch_length_m": 105.0,
                "pitch_width_m": 68.0,
                "confidence": 0.8 if valid else 0.0,
                "reprojection_error": 1.25 if valid else np.nan,
                "visible_pitch_coverage": 0.4 if valid else np.nan,
                "invalid_reason": None if valid else "source calibration invalid",
            }
        ]
    )


def test_adapter_translates_corner_origin_and_emits_inverse() -> None:
    artifact = calibration_to_artifact(_canonical_frame())
    record = artifact["frames"][0]

    assert artifact["pitch"]["length_m"] == 105.0
    assert artifact["pitch"]["width_m"] == 68.0
    assert record["valid"] is True
    np.testing.assert_allclose(
        record["homography"],
        [[1.0, 0.0, -52.5], [0.0, 1.0, -34.0], [0.0, 0.0, 1.0]],
    )
    np.testing.assert_allclose(
        np.asarray(record["homography"])
        @ np.asarray(record["inverse_homography"]),
        np.eye(3),
        atol=1e-10,
    )
    assert artifact["validation"]["contract_valid"] is True


@pytest.mark.parametrize(
    ("homography", "reason"),
    [
        ([[1, 0, 0], [0, 0, 0], [0, 0, 1]], "non-invertible"),
        ([[1, 0, 0], [0, float("inf"), 0], [0, 0, 1]], "non-finite"),
    ],
)
def test_adapter_rejects_unsafe_homographies(homography: object, reason: str) -> None:
    record = calibration_to_artifact(_canonical_frame(homography=homography))["frames"][0]

    assert record["valid"] is False
    assert reason in record["invalid_reason"]
    assert record["homography"] is None
    assert record["inverse_homography"] is None


def test_invalid_source_row_is_json_safe_and_preserved(tmp_path: Path) -> None:
    frame = _canonical_frame(valid=False)
    frame.loc[0, "homography_json"] = None
    output = write_calibration_artifact(frame, tmp_path / "adapter.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    record = payload["frames"][0]

    assert record["valid"] is False
    assert record["invalid_reason"] == "source calibration invalid"
    assert record["reprojection_error"] is None
    assert record["visible_pitch_coverage"] is None
    assert payload["validation"]["contract_valid"] is True


def test_artifact_validator_detects_inverse_tampering() -> None:
    artifact = calibration_to_artifact(_canonical_frame())
    artifact["frames"][0]["inverse_homography"][0][2] += 1.0

    report = validate_artifact(artifact)

    assert report["contract_valid"] is False
    assert any("inverse" in item["reason"] for item in report["errors"])


def test_json_api_emits_strict_parseable_json() -> None:
    payload = json.loads(calibration_to_json(_canonical_frame(), indent=None))

    assert payload["frames"][0]["frame_id"] == 7
    assert payload["validation"]["contract_valid"] is True


def test_frame_selection_preserves_requested_order() -> None:
    frame = pd.concat(
        [
            _canonical_frame().assign(frame_id=frame_id)
            for frame_id in (1, 2, 3, 4, 5)
        ],
        ignore_index=True,
    )

    artifact = calibration_to_artifact(frame, frame_ids=[5, 1, 3])

    assert [item["frame_id"] for item in artifact["frames"]] == [5, 1, 3]
    with pytest.raises(KeyError, match="99"):
        calibration_to_artifact(frame, frame_ids=[99])


@pytest.mark.skipif(not RUN_CALIBRATION.is_file(), reason="reference run is unavailable")
def test_reference_run_selected_five_frames() -> None:
    selected = [0, 155, 156, 374, 749]
    artifact = calibration_to_artifact(RUN_CALIBRATION, frame_ids=selected)

    assert [item["frame_id"] for item in artifact["frames"]] == selected
    assert artifact["validation"] == {
        "contract_valid": True,
        "frame_count": 5,
        "valid_calibration_count": 4,
        "invalid_calibration_count": 1,
        "errors": [],
    }


@pytest.mark.skipif(not RUN_CALIBRATION.is_file(), reason="reference run is unavailable")
def test_reference_run_all_750_frames(tmp_path: Path) -> None:
    output = write_calibration_artifact(RUN_CALIBRATION, tmp_path / "all-frames.json")
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["validation"] == {
        "contract_valid": True,
        "frame_count": 750,
        "valid_calibration_count": 682,
        "invalid_calibration_count": 68,
        "errors": [],
    }
    assert len({item["frame_id"] for item in artifact["frames"]}) == 750
