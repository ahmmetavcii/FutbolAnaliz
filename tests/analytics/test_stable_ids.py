"""Stable display-id attachment tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_analytics.opta.stable_ids import build_stable_display_map, display_id_lookup


def _emb(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=8)
    v = v / np.linalg.norm(v)
    return v.astype(float).tolist()


def test_stable_map_attaches_reid_fragment_to_global():
    tracks = pd.DataFrame(
        [
            {
                "track_id": 1,
                "object_type": "person",
                "timestamp_ms": 0.0,
                "frame_id": 0,
                "bbox_x1": 10,
                "bbox_y1": 10,
                "bbox_x2": 40,
                "bbox_y2": 80,
            },
            {
                "track_id": 1,
                "object_type": "person",
                "timestamp_ms": 1000.0,
                "frame_id": 25,
                "bbox_x1": 12,
                "bbox_y1": 10,
                "bbox_x2": 42,
                "bbox_y2": 80,
            },
            {
                "track_id": 99,
                "object_type": "person",
                "timestamp_ms": 3000.0,
                "frame_id": 75,
                "bbox_x1": 14,
                "bbox_y1": 12,
                "bbox_x2": 44,
                "bbox_y2": 82,
            },
            {
                "track_id": 99,
                "object_type": "person",
                "timestamp_ms": 4000.0,
                "frame_id": 100,
                "bbox_x1": 16,
                "bbox_y1": 12,
                "bbox_x2": 46,
                "bbox_y2": 84,
            },
        ]
    )
    emb = _emb(1)
    gmap = pd.DataFrame(
        [
            {
                "camera_id": "camera_1",
                "local_track_id": 1,
                "global_id": 7,
                "unresolved": False,
                "team_id": "team_0",
                "validated": True,
            }
        ]
    )
    greport = pd.DataFrame(
        [{"global_player_id": 7, "team_id": "team_0", "local_track_ids": "1", "validated": True}]
    )
    reid = pd.DataFrame(
        [
            {"track_id": 1, "embedding": emb, "valid": True},
            {"track_id": 99, "embedding": emb, "valid": True},
        ]
    )
    identities = pd.DataFrame(
        [
            {"track_id": 1, "team_id": "team_0", "frame_id": 0},
            {"track_id": 99, "team_id": "team_0", "frame_id": 75},
        ]
    )
    stable = build_stable_display_map(
        tracks, gmap, greport, reid, identities, reid_attach_threshold=0.5
    )
    lookup = display_id_lookup(stable)
    assert lookup[1] == lookup[99]
    assert int(stable.loc[stable.local_track_id == 99, "source"].iloc[0].startswith("reid_attach"))


def test_orphan_fragments_chain_to_one_display_id():
    """Two orphans of the same person (no global seed) must share one display ID."""
    tracks = pd.DataFrame(
        [
            {
                "track_id": 10,
                "object_type": "person",
                "timestamp_ms": t,
                "frame_id": i,
                "bbox_x1": 100,
                "bbox_y1": 50,
                "bbox_x2": 140,
                "bbox_y2": 160,
            }
            for i, t in enumerate([0.0, 400.0, 800.0])
        ]
        + [
            {
                "track_id": 11,
                "object_type": "person",
                "timestamp_ms": t,
                "frame_id": i + 40,
                "bbox_x1": 108,
                "bbox_y1": 52,
                "bbox_x2": 148,
                "bbox_y2": 162,
            }
            for i, t in enumerate([2000.0, 2400.0, 2800.0])
        ]
    )
    emb = _emb(42)
    reid = pd.DataFrame(
        [
            {"track_id": 10, "embedding": emb, "valid": True},
            {"track_id": 11, "embedding": emb, "valid": True},
        ]
    )
    identities = pd.DataFrame(
        [
            {"track_id": 10, "team_id": "team_1", "frame_id": 0},
            {"track_id": 11, "team_id": "team_1", "frame_id": 40},
        ]
    )
    stable = build_stable_display_map(
        tracks,
        None,
        None,
        reid,
        identities,
        reid_attach_threshold=0.7,
        proximity_gap_ms=5000.0,
        proximity_dist_px=80.0,
    )
    lookup = display_id_lookup(stable)
    assert lookup[10] == lookup[11]
    assert stable["display_id"].nunique() == 1


def test_cross_team_orphans_do_not_chain():
    tracks = pd.DataFrame(
        [
            {
                "track_id": tid,
                "object_type": "person",
                "timestamp_ms": t,
                "frame_id": i + tid * 10,
                "bbox_x1": 100,
                "bbox_y1": 50,
                "bbox_x2": 140,
                "bbox_y2": 160,
            }
            for tid, base in ((10, 0.0), (11, 2000.0))
            for i, t in enumerate([base, base + 400, base + 800])
        ]
    )
    emb = _emb(7)
    reid = pd.DataFrame(
        [
            {"track_id": 10, "embedding": emb, "valid": True},
            {"track_id": 11, "embedding": emb, "valid": True},
        ]
    )
    identities = pd.DataFrame(
        [
            {"track_id": 10, "team_id": "team_0", "frame_id": 0},
            {"track_id": 11, "team_id": "team_1", "frame_id": 40},
        ]
    )
    stable = build_stable_display_map(
        tracks, None, None, reid, identities, reid_attach_threshold=0.5
    )
    lookup = display_id_lookup(stable)
    assert lookup[10] != lookup[11]
