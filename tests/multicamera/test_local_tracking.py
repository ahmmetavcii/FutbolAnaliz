from __future__ import annotations

import numpy as np
import pytest

from football_analytics.multicamera import cosine_similarity, group_into_tracks


def test_track_key_and_grouping(make_obs):
    observations = [
        make_obs(camera_id="cam1", local_track_id=1, reference_time_seconds=1.0),
        make_obs(camera_id="cam1", local_track_id=1, reference_time_seconds=0.0),
        make_obs(camera_id="cam2", local_track_id=1, reference_time_seconds=0.5),
    ]
    tracks = group_into_tracks(observations)
    assert set(tracks) == {("cam1", 1), ("cam2", 1)}
    track = tracks[("cam1", 1)]
    assert track.start_time_seconds == 0.0
    assert track.end_time_seconds == 1.0


def test_track_rejects_foreign_observation(make_obs):
    tracks = group_into_tracks([make_obs(camera_id="cam1", local_track_id=1)])
    with pytest.raises(ValueError):
        tracks[("cam1", 1)].add(make_obs(camera_id="cam2", local_track_id=1))


def test_mean_embedding(make_obs):
    observations = [
        make_obs(reference_time_seconds=0.0, reid_embedding=(1.0, 0.0)),
        make_obs(reference_time_seconds=1.0, reid_embedding=(0.0, 1.0)),
        make_obs(reference_time_seconds=2.0),
    ]
    track = group_into_tracks(observations)[("cam1", 1)]
    assert np.allclose(track.mean_embedding(), [0.5, 0.5])


def test_bbox_must_be_positive(make_obs):
    with pytest.raises(ValueError):
        make_obs(bbox_xyxy=(10.0, 10.0, 5.0, 20.0))


def test_confidences_must_be_in_unit_interval(make_obs):
    with pytest.raises(ValueError):
        make_obs(team_confidence=1.5)


def test_cosine_similarity_degenerate_inputs():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0], [1.0, 0.0]) == 0.0
