import numpy as np

from football_analytics.analytics.kit_descriptor import (
    FAMILY_ORDER,
    compute_kit_family_fractions,
    extract_torso_bgr,
)
from football_analytics.analytics.role_identity import PlayerRole, RoleIdentityTracker
from football_analytics.analytics.team_identity import (
    FEATURE_DIM,
    TeamIdentityAssigner,
    robust_pool,
    sample_upper_torso,
)


def _family_feature(
    *,
    white: float = 0.0,
    gray: float = 0.0,
    black: float = 0.0,
    yellow: float = 0.0,
    orange: float = 0.0,
    green: float = 0.0,
) -> np.ndarray:
    row = np.zeros(FEATURE_DIM, dtype=np.float32)
    idx = {name: i for i, name in enumerate(FAMILY_ORDER)}
    row[idx["white"]] = white
    row[idx["gray"]] = gray
    row[idx["black"]] = black
    row[idx["yellow"]] = yellow
    row[idx["orange"]] = orange
    row[idx["green"]] = green
    total = float(row.sum())
    if total <= 0:
        row[idx["gray"]] = 1.0
        return row
    return (row / total).astype(np.float32)


def test_yellow_kit_survives_pitch_and_classifies_yellow() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:] = (40, 160, 40)  # grass
    frame[28:52, 35:65] = (20, 220, 220)  # yellow jersey
    feature, fraction = sample_upper_torso(frame, (25, 10, 75, 80), min_pixels=8)
    assert feature is not None
    assert feature.shape == (FEATURE_DIM,)
    assert float(feature[FAMILY_ORDER.index("yellow")]) > 0.4
    assert fraction > 0.05


def test_white_kit_family_on_pitch() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:] = (40, 160, 40)
    frame[28:52, 35:65] = (220, 220, 220)
    feature, _ = sample_upper_torso(frame, (25, 10, 75, 80), min_pixels=8)
    assert feature is not None
    white_gray = float(
        feature[FAMILY_ORDER.index("white")] + feature[FAMILY_ORDER.index("gray")]
    )
    assert white_gray > 0.5


def test_kit_torso_roi_matches_reference_fractions() -> None:
    frame = np.zeros((100, 80, 3), dtype=np.uint8)
    frame[:] = (0, 0, 255)
    torso = extract_torso_bgr(frame, (0, 0, 80, 100))
    assert torso is not None
    # 0.20–0.80 x, 0.15–0.65 y on 80x100 box → 48x50
    assert torso.shape[1] == 48
    assert torso.shape[0] == 50
    fracs = compute_kit_family_fractions(torso, mask_pitch=False)
    assert fracs is not None
    assert abs(float(fracs.sum()) - 1.0) < 1e-5


def test_robust_temporal_pool_rejects_outlier() -> None:
    base = _family_feature(yellow=0.8, green=0.2)
    features = [base, base.copy(), base.copy()]
    outlier = _family_feature(white=1.0)
    features.append(outlier)
    pooled = robust_pool(features)
    assert pooled is not None
    assert float(pooled[FAMILY_ORDER.index("yellow")]) > 0.5


def test_two_team_fit_unknown_referee_and_scene_reset() -> None:
    assigner = TeamIdentityAssigner(
        min_samples=8,
        min_tracks=4,
        unknown_confidence_threshold=0.35,
    )
    for track_id, feat in (
        (1, _family_feature(yellow=0.85, green=0.15)),
        (2, _family_feature(yellow=0.80, orange=0.20)),
        (3, _family_feature(white=0.70, gray=0.30)),
        (4, _family_feature(white=0.65, gray=0.35)),
    ):
        assigner.add_sample(track_id, feat)
        assigner.add_sample(track_id, feat)
    assigner.add_sample(
        99, _family_feature(black=0.9, gray=0.1), role=PlayerRole.REFEREE
    )
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


def test_dark_kits_excluded_from_team_assignment() -> None:
    assigner = TeamIdentityAssigner(
        min_samples=8,
        min_tracks=4,
        unknown_confidence_threshold=0.35,
        exclude_dark_kits=True,
    )
    for track_id, feat in (
        (1, _family_feature(yellow=0.9)),
        (2, _family_feature(yellow=0.85, orange=0.15)),
        (3, _family_feature(white=0.8, gray=0.2)),
        (4, _family_feature(white=0.75, gray=0.25)),
        (5, _family_feature(black=0.7, gray=0.3)),
    ):
        for _ in range(3):
            assigner.add_sample(track_id, feat)
    assert assigner.fit()
    yellow = assigner.assign(1)
    white = assigner.assign(3)
    dark = assigner.assign(5)
    assert yellow.team_id is not None
    assert white.team_id is not None
    assert yellow.team_id != white.team_id
    assert dark.is_unknown


def test_goalkeeper_role_is_retained_and_referee_excluded() -> None:
    roles = RoleIdentityTracker(smoothing=0.0)
    goalkeeper = roles.update(7, goalkeeper_confidence=0.9)
    retained = roles.update(7, goalkeeper_confidence=0.4)
    referee = roles.update(8, referee_confidence=0.9)
    assert goalkeeper.role is PlayerRole.GOALKEEPER
    assert retained.role is PlayerRole.GOALKEEPER
    assert referee.excluded_from_team_clustering
