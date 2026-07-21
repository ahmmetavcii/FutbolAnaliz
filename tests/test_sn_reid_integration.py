"""Tests for sn-reid integration and global-identity wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_analytics.contracts.schemas import (
    REID_EMBEDDINGS_SCHEMA,
    TRACK_REID_PROTOTYPES_SCHEMA,
    validate_mvp2_columns,
)
from football_analytics.full_match.postprocess import build_global_identity
from football_analytics.integrations.sn_reid_extractor import l2_normalize
from football_analytics.stages.reid import ReidStage
from football_analytics.utils.io import write_rows_with_schema


def test_l2_normalize_unit_rows():
    matrix = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    normalized = l2_normalize(matrix)
    assert normalized.shape == (2, 2)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)


def test_select_samples_respects_stride_and_cap():
    rows = []
    for frame_id in range(20):
        rows.append(
            {
                "frame_id": frame_id,
                "timestamp_ms": frame_id * 40.0,
                "track_id": 7,
                "object_type": "person",
                "bbox_x1": 10.0,
                "bbox_y1": 10.0,
                "bbox_x2": 50.0,
                "bbox_y2": 80.0,
                "tracking_confidence": 0.9,
            }
        )
    frame = pd.DataFrame(rows)
    selected = ReidStage._select_samples(
        frame,
        sample_stride=5,
        max_samples=3,
        min_area=36,
        image_width=1920,
        image_height=1080,
    )
    assert len(selected) == 3
    assert [item["frame_id"] for item in selected] == [0, 5, 10]


def test_build_global_identity_merges_with_reid_prototypes():
    tracks = pd.DataFrame(
        [
            {
                "frame_id": 0,
                "timestamp_ms": 0.0,
                "track_id": 1,
                "object_type": "person",
                "bbox_x1": 10,
                "bbox_y1": 10,
                "bbox_x2": 40,
                "bbox_y2": 80,
                "tracking_confidence": 0.9,
            },
            {
                "frame_id": 100,
                "timestamp_ms": 4000.0,
                "track_id": 42,
                "object_type": "person",
                "bbox_x1": 12,
                "bbox_y1": 12,
                "bbox_x2": 42,
                "bbox_y2": 82,
                "tracking_confidence": 0.9,
            },
        ]
    )
    identities = pd.DataFrame(
        [
            {
                "frame_id": 0,
                "track_id": 1,
                "team_id": "team_0",
                "team_confidence": 0.9,
            },
            {
                "frame_id": 100,
                "track_id": 42,
                "team_id": "team_0",
                "team_confidence": 0.9,
            },
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "frame_id": 0,
                "track_id": 1,
                "x_field": 30.0,
                "y_field": 30.0,
                "valid": True,
            },
            {
                "frame_id": 100,
                "track_id": 42,
                "x_field": 33.0,
                "y_field": 30.0,
                "valid": True,
            },
        ]
    )
    embedding = [1.0] + [0.0] * 15
    prototypes = pd.DataFrame(
        [
            {"track_id": 1, "embedding": embedding, "valid": True},
            {"track_id": 42, "embedding": embedding, "valid": True},
        ]
    )
    identity_map, global_players, registry = build_global_identity(
        tracks, identities, metrics, "camera_1", reid_prototypes=prototypes
    )
    assert len(identity_map) == 2
    assert identity_map["global_id"].nunique() == 1
    assert int(global_players.iloc[0]["n_local_tracks"]) == 2
    assert not bool(global_players.iloc[0]["unresolved"])
    assert set(identity_map["local_track_id"].tolist()) == {1, 42}


def test_reid_schema_roundtrip(tmp_path: Path):
    embedding = [0.1, 0.2, 0.3]
    rows = [
        {
            "schema_version": "2.0.0",
            "run_id": "run_test",
            "match_id": "match",
            "frame_id": 1,
            "timestamp_ms": 40.0,
            "source_method": "sn_reid/osnet",
            "confidence": 0.9,
            "valid": True,
            "track_id": 3,
            "bbox_x1": 1.0,
            "bbox_y1": 2.0,
            "bbox_x2": 10.0,
            "bbox_y2": 20.0,
            "embedding": embedding,
            "embedding_dim": 3,
            "model_name": "osnet_x1_0",
        }
    ]
    path = tmp_path / "reid_embeddings.parquet"
    write_rows_with_schema(path, rows, REID_EMBEDDINGS_SCHEMA)
    frame = pd.read_parquet(path)
    validate_mvp2_columns("reid_embeddings", list(frame.columns))
    assert list(frame.iloc[0]["embedding"]) == pytest.approx(embedding)

    proto_rows = [
        {
            "schema_version": "2.0.0",
            "run_id": "run_test",
            "match_id": "match",
            "frame_id": 1,
            "timestamp_ms": 40.0,
            "source_method": "sn_reid/osnet_mean",
            "confidence": 0.9,
            "valid": True,
            "track_id": 3,
            "n_samples": 1,
            "embedding": embedding,
            "embedding_dim": 3,
            "model_name": "osnet_x1_0",
        }
    ]
    proto_path = tmp_path / "track_reid_prototypes.parquet"
    write_rows_with_schema(proto_path, proto_rows, TRACK_REID_PROTOTYPES_SCHEMA)
    proto = pd.read_parquet(proto_path)
    validate_mvp2_columns("track_reid_prototypes", list(proto.columns))


@pytest.mark.skipif(
    not Path("/home/ahmet/models/sn-reid/osnet_x1_0_market1501.pth").is_file(),
    reason="OSNet Market1501 weights not downloaded",
)
def test_sn_reid_extractor_smoke():
    from football_analytics.integrations.sn_reid_extractor import SnReidExtractor

    extractor = SnReidExtractor(device="cuda:0", verbose=False)
    crop = np.random.randint(0, 255, (120, 60, 3), dtype=np.uint8)
    vectors = extractor.extract([crop, crop], assume_bgr=True)
    assert vectors.shape == (2, 512)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert float(np.dot(vectors[0], vectors[1])) > 0.99
