from __future__ import annotations

import pyarrow as pa

from football_analytics.contracts.schemas import (
    DETECTIONS_SCHEMA,
    MVP2_SCHEMAS,
    TRACKS_SCHEMA,
    validate_mvp2_columns,
)


def test_mvp1_schemas_remain_stable() -> None:
    assert DETECTIONS_SCHEMA.field("detection_id").type == pa.string()
    assert TRACKS_SCHEMA.field("foot_y_pixel").type == pa.float64()
    assert "run_id" not in TRACKS_SCHEMA.names


def test_all_mvp2_schemas_have_common_contract() -> None:
    common = {
        "schema_version",
        "run_id",
        "match_id",
        "frame_id",
        "timestamp_ms",
        "source_method",
        "confidence",
        "valid",
    }
    for name, schema in MVP2_SCHEMAS.items():
        assert common.issubset(schema.names), name
        validate_mvp2_columns(name, schema.names)


def test_unknown_values_are_nullable() -> None:
    for schema in MVP2_SCHEMAS.values():
        assert all(field.nullable for field in schema)
