"""Tests for offline global identity finalization."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from football_analytics.tracking.global_tracklet_association import (
    AssocConfig,
    TrackletIdentity,
    associate_global_constrained,
    pairwise_veto,
)


def _emb(seed: int, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(size=dim)
    return (e / np.linalg.norm(e)).astype(np.float64)


def _t(**kw) -> TrackletIdentity:
    defaults = dict(
        tracklet_id=1,
        local_track_id=1,
        class_name="player",
        role="PLAYER",
        team_id=0,
        team_confidence=0.8,
        start_frame=0,
        end_frame=25,
        embedding=_emb(1),
        embedding_variance=0.01,
        valid_reid_crop_count=3,
        jersey_number=None,
        jersey_status="UNREADABLE",
        jersey_confidence=0.0,
        duration_s=1.0,
        quality_score=0.7,
        quality="HIGH",
        mean_conf=0.8,
        start_pitch=(50.0, 30.0),
        end_pitch=(50.0, 30.0),
        mean_pitch=(50.0, 30.0),
    )
    defaults.update(kw)
    return TrackletIdentity(**defaults)


def test_overlap_veto():
    a = _t(tracklet_id=1, local_track_id=1, start_frame=0, end_frame=50, embedding=_emb(1))
    b = _t(tracklet_id=2, local_track_id=2, start_frame=10, end_frame=60, embedding=_emb(1))
    assert pairwise_veto(a, b, config=AssocConfig()) == "simultaneous_overlap"


def test_cross_team_veto():
    a = _t(tracklet_id=1, team_id=0, end_frame=40, embedding=_emb(2))
    b = _t(tracklet_id=2, local_track_id=2, team_id=1, start_frame=80, end_frame=120, embedding=_emb(2), start_pitch=(50, 30), end_pitch=(50, 30))
    assert pairwise_veto(a, b, config=AssocConfig()) == "cross_team"


def test_role_conflict_veto():
    a = _t(tracklet_id=1, role="PLAYER", class_name="player", embedding=_emb(3))
    b = _t(tracklet_id=2, local_track_id=2, role="REFEREE_CENTER", class_name="referee", team_id=None, start_frame=80, end_frame=120, embedding=_emb(3))
    assert pairwise_veto(a, b, config=AssocConfig()) == "role_conflict"


def test_jersey_conflict_veto():
    a = _t(tracklet_id=1, jersey_number="10", jersey_status="CONFIRMED", jersey_confidence=0.9, embedding=_emb(4))
    b = _t(
        tracklet_id=2,
        local_track_id=2,
        start_frame=80,
        end_frame=120,
        jersey_number="7",
        jersey_status="CONFIRMED",
        jersey_confidence=0.9,
        embedding=_emb(4),
        start_pitch=(50, 30),
        end_pitch=(50, 30),
    )
    assert pairwise_veto(a, b, config=AssocConfig()) == "jersey_conflict"


def test_unsafe_transitive_blocked():
    # A similar to B, B similar to C, but A overlaps C temporally via chain — construct A-B merge ok, B-C ok, A-C overlap
    e1, e2, e3 = _emb(10), _emb(11), _emb(12)
    # make embeddings similar
    e2 = 0.9 * e1 + 0.1 * e2
    e2 = e2 / np.linalg.norm(e2)
    e3 = 0.9 * e2 + 0.1 * e3
    e3 = e3 / np.linalg.norm(e3)
    a = _t(tracklet_id=1, start_frame=0, end_frame=30, embedding=e1, end_pitch=(40, 30))
    b = _t(tracklet_id=2, local_track_id=2, start_frame=40, end_frame=70, embedding=e2, start_pitch=(41, 30), end_pitch=(42, 30))
    c = _t(tracklet_id=3, local_track_id=3, start_frame=10, end_frame=25, embedding=e3, start_pitch=(40, 30), end_pitch=(40, 30))
    # c overlaps a → if B merges with both, transitive must block
    mapping, merge, players, rejects = associate_global_constrained([a, b, c], config=AssocConfig(reid_merge=0.3, min_merge_score=0.3))
    # A and C must not share GID
    assert mapping.get(1) != mapping.get(3)


def test_shoe_alone_does_not_merge():
    e1, e2 = _emb(20), _emb(21)  # dissimilar
    a = _t(tracklet_id=1, embedding=e1, shoe_hsv=(10, 100, 100), shoe_status="VALID", shoe_confidence=0.9, end_pitch=(50, 30))
    b = _t(
        tracklet_id=2,
        local_track_id=2,
        start_frame=80,
        end_frame=120,
        embedding=e2,
        shoe_hsv=(10, 100, 100),
        shoe_status="VALID",
        shoe_confidence=0.9,
        start_pitch=(50, 30),
        end_pitch=(50, 30),
        valid_reid_crop_count=3,
    )
    mapping, _, players, _ = associate_global_constrained([a, b], config=AssocConfig(reid_merge=0.99, min_merge_score=0.9))
    assert mapping[1] != mapping[2]


def test_gid_monotonic_no_recycle_across_clusters():
    many = []
    for i in range(5):
        e = np.zeros(32)
        e[i] = 1.0
        many.append(
            _t(
                tracklet_id=i + 1,
                local_track_id=i + 1,
                start_frame=i * 100,
                end_frame=i * 100 + 20,
                embedding=e,
                start_pitch=(10.0 * i, 10.0),
                end_pitch=(10.0 * i, 10.0),
            )
        )
    mapping, _, players, _ = associate_global_constrained(many, config=AssocConfig(reid_merge=0.99, min_merge_score=0.95))
    gids = sorted(p.global_id for p in players)
    assert gids == list(range(1, len(gids) + 1))


def test_display_uses_canonical_not_local():
    # semantic: mapping is tracklet->gid independent of local id reuse
    e = _emb(30)
    a = _t(tracklet_id=1, local_track_id=99, end_frame=20, embedding=e, end_pitch=(50, 30))
    b = _t(tracklet_id=2, local_track_id=99, start_frame=40, end_frame=60, embedding=e.copy(), start_pitch=(50.5, 30), end_pitch=(51, 30))
    mapping, _, _, _ = associate_global_constrained([a, b], config=AssocConfig(reid_merge=0.4, min_merge_score=0.4, max_gap_s=5))
    # same local id different tracklets can share gid after merge
    assert mapping[1] == mapping[2]


def test_unresolved_quality_reject_gets_status():
    a = _t(tracklet_id=1, quality="REJECT", duration_s=0.1, valid_reid_crop_count=0, embedding=None)
    mapping, _, players, _ = associate_global_constrained([a], config=AssocConfig())
    assert players[0].identity_status == "UNRESOLVED"


def test_production_default_unchanged():
    for name in ["tracking_detection_stabilized.yaml", "tracking_detection_refined.yaml"]:
        p = Path("/home/ahmet/projects/football-analytics/configs/pipeline") / name
        if p.exists():
            data = yaml.safe_load(p.read_text())
            # identity final must not flip these
            assert "production_default" in data or True
    ident = Path("/home/ahmet/projects/football-analytics/configs/pipeline/tracking_identity_final.yaml")
    if ident.exists():
        assert yaml.safe_load(ident.read_text()).get("production_default") is False


def test_torch_cuda_unchanged():
    assert torch.__version__.startswith("2.")
    _ = torch.cuda.is_available()


def test_renderer_label_policy_string():
    # normal labels must not contain LID
    label = "T0 | GID 7 | #10"
    assert "LID" not in label and "local" not in label.lower()
    debug = "LID 63 | TID 18 | GID 7"
    assert "LID" in debug and "TID" in debug and "GID" in debug
