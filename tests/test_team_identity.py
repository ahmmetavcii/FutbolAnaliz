import cv2
import numpy as np

from football_analytics.analytics.role_identity import PlayerRole, RoleIdentityTracker
from football_analytics.analytics.team_identity import (
    TeamIdentityAssigner,
    robust_pool,
    sample_upper_torso,
)


def _solid_feature(lab: tuple[float, float, float]) -> np.ndarray:
    return np.asarray((*lab, 10.0, 180.0, 160.0), dtype=np.float32)


def test_upper_torso_sampling_masks_pitch() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:] = (0, 180, 0)
    frame[28:52, 35:65] = (0, 0, 190)
    feature, fraction = sample_upper_torso(frame, (25, 10, 75, 80), min_pixels=8)
    assert feature is not None
    expected_lab = cv2.cvtColor(np.uint8([[[0, 0, 190]]]), cv2.COLOR_BGR2LAB)[0, 0]
    assert np.linalg.norm(feature[:3] - expected_lab) < 5
    assert 0.0 < fraction <= 1.0


def test_robust_temporal_pool_rejects_outlier() -> None:
    features = [_solid_feature((100 + offset, 150, 170)) for offset in (-1, 0, 1)]
    features.append(_solid_feature((250, 10, 10)))
    pooled = robust_pool(features)
    assert pooled is not None
    assert np.linalg.norm(pooled[:3] - np.array([100, 150, 170])) < 3


def test_two_team_fit_unknown_referee_and_scene_reset() -> None:
    assigner = TeamIdentityAssigner(
        min_samples=8,
        min_tracks=4,
        unknown_confidence_threshold=0.4,
    )
    for track_id, base in ((1, (70, 170, 170)), (2, (72, 169, 171)),
                           (3, (180, 95, 105)), (4, (182, 96, 104))):
        assigner.add_sample(track_id, _solid_feature(base))
        assigner.add_sample(track_id, _solid_feature(base) + np.array([1, 0, 0, 0, 0, 0]))
    assigner.add_sample(99, _solid_feature((120, 128, 128)), role=PlayerRole.REFEREE)
    assert assigner.fit()
    first = assigner.assign(1)
    second = assigner.assign(3)
    referee = assigner.assign(99, role=PlayerRole.REFEREE)
    assert first.team_id is not None
    assert second.team_id is not None
    assert first.team_id != second.team_id
    assert referee.is_unknown and referee.confidence == 0.0

    assigner.reset_scene()
    assert not assigner.fitted
    assert assigner.assign(1).is_unknown


def test_goalkeeper_role_is_retained_and_referee_excluded() -> None:
    roles = RoleIdentityTracker(smoothing=0.0)
    goalkeeper = roles.update(7, goalkeeper_confidence=0.9)
    retained = roles.update(7, goalkeeper_confidence=0.4)
    referee = roles.update(8, referee_confidence=0.9)
    assert goalkeeper.role is PlayerRole.GOALKEEPER
    assert retained.role is PlayerRole.GOALKEEPER
    assert referee.excluded_from_team_clustering
