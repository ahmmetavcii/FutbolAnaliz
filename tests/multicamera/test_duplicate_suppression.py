from __future__ import annotations

from football_analytics.multicamera import (
    find_impossible_shared_identities,
    find_intra_camera_duplicates,
    suppress_duplicates,
)


def test_same_camera_same_spot_duplicate_suppressed(make_obs):
    keeper = make_obs(
        camera_id="cam1",
        local_track_id=1,
        reference_time_seconds=10.0,
        pitch_xy_m=(10.0, 10.0),
        detection_confidence=0.9,
    )
    double_fire = make_obs(
        camera_id="cam1",
        local_track_id=2,
        reference_time_seconds=10.02,
        pitch_xy_m=(10.3, 10.2),
        detection_confidence=0.5,
    )
    kept, pairs = suppress_duplicates([keeper, double_fire])
    assert len(pairs) == 1
    assert pairs[0].kept is keeper
    assert pairs[0].suppressed is double_fire
    assert kept == [keeper]


def test_different_cameras_are_not_duplicates(make_obs):
    pairs = find_intra_camera_duplicates(
        [
            make_obs(camera_id="cam1", local_track_id=1, pitch_xy_m=(10.0, 10.0)),
            make_obs(camera_id="cam2", local_track_id=1, pitch_xy_m=(10.0, 10.0)),
        ]
    )
    assert pairs == []


def test_distant_same_camera_boxes_are_not_duplicates(make_obs):
    pairs = find_intra_camera_duplicates(
        [
            make_obs(camera_id="cam1", local_track_id=1, pitch_xy_m=(10.0, 10.0)),
            make_obs(camera_id="cam1", local_track_id=2, pitch_xy_m=(30.0, 10.0)),
        ]
    )
    assert pairs == []


def test_impossible_shared_jersey_flagged(make_obs):
    flags = find_impossible_shared_identities(
        [
            make_obs(
                camera_id="cam1",
                local_track_id=1,
                reference_time_seconds=10.0,
                pitch_xy_m=(0.0, 0.0),
                jersey_number=9,
                jersey_confidence=0.9,
            ),
            make_obs(
                camera_id="cam2",
                local_track_id=2,
                reference_time_seconds=10.03,
                pitch_xy_m=(40.0, 0.0),
                jersey_number=9,
                jersey_confidence=0.9,
            ),
        ]
    )
    assert len(flags) == 1
    assert flags[0]["reason"] == "impossible_simultaneous_duplicate"
    assert flags[0]["jersey_number"] == 9


def test_different_teams_sharing_jersey_not_flagged(make_obs):
    flags = find_impossible_shared_identities(
        [
            make_obs(
                camera_id="cam1",
                local_track_id=1,
                reference_time_seconds=10.0,
                pitch_xy_m=(0.0, 0.0),
                team_id=0,
                team_confidence=0.9,
                jersey_number=9,
                jersey_confidence=0.9,
            ),
            make_obs(
                camera_id="cam2",
                local_track_id=2,
                reference_time_seconds=10.03,
                pitch_xy_m=(40.0, 0.0),
                team_id=1,
                team_confidence=0.9,
                jersey_number=9,
                jersey_confidence=0.9,
            ),
        ]
    )
    assert flags == []
