from __future__ import annotations

import pytest

from football_analytics.multicamera import PlayerRole, fuse_observations, fuse_timeline


def test_position_is_confidence_weighted(make_obs):
    fused = fuse_observations(
        1,
        [
            make_obs(
                camera_id="cam1",
                reference_time_seconds=10.0,
                pitch_xy_m=(10.0, 0.0),
                detection_confidence=0.9,
            ),
            make_obs(
                camera_id="cam2",
                reference_time_seconds=10.0,
                pitch_xy_m=(20.0, 0.0),
                detection_confidence=0.1,
            ),
        ],
    )
    assert fused.pitch_xy_m[0] == pytest.approx(11.0)
    assert fused.source_cameras == ("cam1", "cam2")
    assert not fused.has_conflict


def test_most_confident_attribute_wins(make_obs):
    fused = fuse_observations(
        1,
        [
            make_obs(camera_id="cam1", team_id=0, team_confidence=0.6),
            make_obs(camera_id="cam2", team_id=0, team_confidence=0.9),
        ],
    )
    assert fused.team_id == 0
    assert fused.team_confidence == pytest.approx(0.9)
    assert not fused.has_conflict


def test_confident_disagreement_raises_conflict_flags(make_obs):
    fused = fuse_observations(
        1,
        [
            make_obs(
                camera_id="cam1",
                team_id=0,
                team_confidence=0.9,
                jersey_number=9,
                jersey_confidence=0.8,
                role=PlayerRole.OUTFIELD,
            ),
            make_obs(
                camera_id="cam2",
                team_id=1,
                team_confidence=0.8,
                jersey_number=10,
                jersey_confidence=0.9,
                role=PlayerRole.GOALKEEPER,
            ),
        ],
    )
    assert fused.has_conflict
    assert set(fused.conflicts) == {"team_conflict", "jersey_conflict", "role_conflict"}
    # Weighted selection still picks the most confident values.
    assert fused.team_id == 0
    assert fused.jersey_number == 10


def test_low_confidence_sources_do_not_vote(make_obs):
    fused = fuse_observations(
        1,
        [
            make_obs(camera_id="cam1", team_id=0, team_confidence=0.9),
            make_obs(camera_id="cam2", team_id=1, team_confidence=0.2),
        ],
    )
    assert fused.team_id == 0
    assert not fused.has_conflict


def test_missing_positions_yield_null_pitch(make_obs):
    fused = fuse_observations(1, [make_obs(pitch_xy_m=None)])
    assert fused.pitch_xy_m is None


def test_fuse_timeline_bins_by_time(make_obs):
    observations = [
        make_obs(camera_id="cam1", reference_time_seconds=0.05, pitch_xy_m=(1.0, 1.0)),
        make_obs(camera_id="cam2", reference_time_seconds=0.10, pitch_xy_m=(2.0, 2.0)),
        make_obs(camera_id="cam1", reference_time_seconds=5.0, pitch_xy_m=(3.0, 3.0)),
    ]
    timeline = fuse_timeline(1, observations, bin_seconds=0.2)
    assert len(timeline) == 2
    assert timeline[0].observation_count == 2
    assert timeline[1].observation_count == 1


def test_empty_observations_rejected():
    with pytest.raises(ValueError):
        fuse_observations(1, [])
