"""Overlay slot identity tests."""

from __future__ import annotations

import pandas as pd

from football_analytics.opta.overlay_slots import (
    build_overlay_slot_assignments,
    frame_display_lookup,
)


def test_overlay_slots_keep_id_across_track_break():
    """When local track_id changes but position continues, display_id stays."""
    rows = []
    # Track 1 frames 0-9
    for f in range(10):
        rows.append(
            {
                "frame_id": f,
                "track_id": 1,
                "object_type": "person",
                "bbox_x1": 100 + f,
                "bbox_y1": 40,
                "bbox_x2": 140 + f,
                "bbox_y2": 160,
            }
        )
    # Gap then track 99 continues nearby
    for f in range(12, 22):
        rows.append(
            {
                "frame_id": f,
                "track_id": 99,
                "object_type": "person",
                "bbox_x1": 100 + f,
                "bbox_y1": 40,
                "bbox_x2": 140 + f,
                "bbox_y2": 160,
            }
        )
    tracks = pd.DataFrame(rows)
    identities = pd.DataFrame(
        [
            {"track_id": 1, "team_id": "team_0", "frame_id": 0},
            {"track_id": 99, "team_id": "team_0", "frame_id": 12},
        ]
    )
    frame_map, track_map = build_overlay_slot_assignments(
        tracks, identities, match_dist_px=80.0, hold_frames=30
    )
    lookup = frame_display_lookup(frame_map)
    assert lookup[(0, 1)] == lookup[(12, 99)]
    assert frame_map["display_id"].nunique() == 1


def test_overlay_slots_do_not_merge_two_simultaneous_players():
    rows = []
    for f in range(8):
        rows.append(
            {
                "frame_id": f,
                "track_id": 1,
                "object_type": "person",
                "bbox_x1": 50,
                "bbox_y1": 40,
                "bbox_x2": 90,
                "bbox_y2": 160,
            }
        )
        rows.append(
            {
                "frame_id": f,
                "track_id": 2,
                "object_type": "person",
                "bbox_x1": 300,
                "bbox_y1": 40,
                "bbox_x2": 340,
                "bbox_y2": 160,
            }
        )
    tracks = pd.DataFrame(rows)
    identities = pd.DataFrame(
        [
            {"track_id": 1, "team_id": "team_1", "frame_id": 0},
            {"track_id": 2, "team_id": "team_1", "frame_id": 0},
        ]
    )
    frame_map, _ = build_overlay_slot_assignments(tracks, identities)
    lookup = frame_display_lookup(frame_map)
    assert lookup[(0, 1)] != lookup[(0, 2)]
    assert frame_map["display_id"].nunique() == 2
