"""Tests for ReID matching helpers and calibrated identity stitching."""

from __future__ import annotations

import numpy as np

from football_analytics.analytics.reid_matching import (
    calibrate_hard_negatives,
    cosine_similarity,
    relative_accept,
    robust_mean_embedding,
    torso_crop_xyxy,
)
from football_analytics.opta.identity_resolve import (
    IdentityResolveConfig,
    TrackFragment,
    resolve_global_identities,
)


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=32)
    return v / np.linalg.norm(v)


def test_torso_crop_inside_frame():
    box = torso_crop_xyxy((100, 100, 200, 300), image_width=640, image_height=480)
    assert box is not None
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= 640
    assert 0 <= y1 < y2 <= 480
    assert (y2 - y1) < (300 - 100)


def test_robust_mean_rejects_outlier():
    base = _emb(1)
    outlier = _emb(99)
    pooled, consistency = robust_mean_embedding([base, base, base, outlier])
    assert pooled is not None
    assert cosine_similarity(pooled, base) > 0.9
    assert consistency > 0.8


def test_hard_negative_calibration_raises_threshold():
    embeddings = {i: _emb(i) for i in range(6)}
    # Make simultaneous pairs look alike
    shared = _emb(0)
    for i in range(6):
        embeddings[i] = shared * 0.9 + _emb(i + 10) * 0.1
        embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
    intervals = {i: (0.0, 1000.0) for i in range(6)}
    teams = {i: "team_0" for i in range(6)}
    cal = calibrate_hard_negatives(
        embeddings, intervals, teams, base_merge=0.42, base_strong=0.55, margin=0.05
    )
    assert cal.pair_count >= 8
    assert cal.merge_threshold > 0.42


def test_relative_accept_requires_margin():
    ok, reason = relative_accept(0.80, 0.79, merge_threshold=0.7, strong_threshold=0.85, relative_margin=0.04)
    assert ok is False
    ok2, _ = relative_accept(0.90, 0.70, merge_threshold=0.7, strong_threshold=0.85, relative_margin=0.04)
    assert ok2 is True


def test_second_pass_merges_non_overlapping_same_embedding():
    emb = _emb(3)
    frags = [
        TrackFragment(1, "team_0", 0.9, "outfield", 0, 1000, 20, 1.0, emb, (10, 10), (12, 10), (11, 10)),
        TrackFragment(2, "team_0", 0.9, "outfield", 5000, 7000, 20, 2.0, emb, (14, 10), (18, 10), (16, 10)),
    ]
    gmap, report, metrics, _ = resolve_global_identities(
        frags,
        config=IdentityResolveConfig(
            reid_merge_threshold=0.5,
            reid_strong_threshold=0.6,
            enable_second_pass_gallery=True,
            max_gap_seconds_strong_reid=30.0,
        ),
    )
    assert gmap["global_id"].nunique() == 1
    assert int(report.iloc[0]["track_fragment_count"]) == 2
    assert metrics["merged_fragments"] >= 1
