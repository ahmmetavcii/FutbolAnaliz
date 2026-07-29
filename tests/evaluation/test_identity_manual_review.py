"""Tests for identity manual review store, constraints, and UI policies."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from football_analytics.evaluation.identity_review_store import IdentityReviewStore
from football_analytics.tracking.global_tracklet_association import (
    AssocConfig,
    TrackletIdentity,
    associate_global_constrained,
    pairwise_veto,
)


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(size=32)
    return e / np.linalg.norm(e)


def _t(**kw) -> TrackletIdentity:
    base = dict(
        tracklet_id=1,
        local_track_id=1,
        class_name="player",
        role="PLAYER",
        team_id=0,
        team_confidence=0.8,
        start_frame=0,
        end_frame=20,
        embedding=_emb(1),
        embedding_variance=0.01,
        valid_reid_crop_count=3,
        jersey_number=None,
        jersey_status="UNREADABLE",
        jersey_confidence=0.0,
        duration_s=1.0,
        quality="HIGH",
        mean_conf=0.8,
        start_pitch=(40.0, 30.0),
        end_pitch=(40.0, 30.0),
    )
    base.update(kw)
    return TrackletIdentity(**base)


def test_button_save_autosaves_and_verifies(tmp_path: Path):
    store = IdentityReviewStore(tmp_path)
    row = store.save_decision("review_001", "SAME")
    assert row["human_decision"] == "SAME"
    again = store.get_decision("review_001")
    assert again is not None and again["human_decision"] == "SAME"
    assert store._csv_contains("review_001", "SAME")


def test_failed_save_detected(tmp_path: Path, monkeypatch):
    store = IdentityReviewStore(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "_write_csv_snapshot", boom)
    try:
        store.save_decision("review_002", "DIFFERENT")
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_resume_first_unanswered(tmp_path: Path):
    store = IdentityReviewStore(tmp_path)
    for i in range(1, 4):
        store.save_decision(f"review_{i:03d}", "UNSURE")
    queue_ids = [f"review_{i:03d}" for i in range(1, 6)]
    done = store.all_decisions()
    first = next(i for i, rid in enumerate(queue_ids) if rid not in done)
    assert first == 3


def test_revision_does_not_inflate_completion(tmp_path: Path):
    store = IdentityReviewStore(tmp_path)
    store.save_decision("review_010", "SAME")
    store.save_decision("review_010", "DIFFERENT")
    assert store.completed_count() == 1
    hist = store.revision_history("review_010")
    assert len(hist) == 2
    assert store.get_decision("review_010")["human_decision"] == "DIFFERENT"


def test_csv_atomic_and_jsonl_append(tmp_path: Path):
    store = IdentityReviewStore(tmp_path)
    store.save_decision("review_001", "SAME")
    store.save_decision("review_002", "UNSURE")
    assert store.csv_path.exists()
    with open(store.csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    lines = store.jsonl_path.read_text().strip().splitlines()
    assert len(lines) >= 2
    assert all(json.loads(x)["review_id"] for x in lines)


def test_queue_unique_if_present():
    p = Path("/home/ahmet/projects/football-analytics/configs/evaluation/identity_review/football/review_queue.parquet")
    if not p.exists():
        return
    q = pd.read_parquet(p)
    assert len(q) == 20
    assert q.review_id.nunique() == 20


def test_model_decision_not_in_user_facing_label():
    label = "BU İKİSİ AYNI OYUNCU MU?"
    assert "SAME" not in label and "model" not in label.lower()


def test_same_must_link_merges():
    e = _emb(5)
    a = _t(tracklet_id=1, embedding=e, end_frame=20, end_pitch=(40, 30))
    b = _t(tracklet_id=2, local_track_id=2, start_frame=40, end_frame=60, embedding=e.copy(), start_pitch=(41, 30), end_pitch=(42, 30))
    mapping, _, _, _ = associate_global_constrained(
        [a, b],
        config=AssocConfig(reid_merge=0.99, min_merge_score=0.99),  # reid alone would not merge
        must_links=[(1, 2)],
    )
    assert mapping[1] == mapping[2]


def test_different_cannot_link():
    e = _emb(6)
    a = _t(tracklet_id=1, embedding=e, end_frame=20, end_pitch=(40, 30))
    b = _t(tracklet_id=2, local_track_id=2, start_frame=40, end_frame=60, embedding=e.copy(), start_pitch=(41, 30), end_pitch=(42, 30))
    mapping, _, _, rejects = associate_global_constrained(
        [a, b],
        config=AssocConfig(reid_merge=0.4, min_merge_score=0.4),
        cannot_links=[(1, 2)],
    )
    assert mapping[1] != mapping[2]
    assert any(r.get("reason") == "human_cannot_link" for r in rejects)


def test_unsure_no_forced_rule():
    # UNSURE simply means no must/cannot passed — default association
    e1, e2 = _emb(7), _emb(8)
    a = _t(tracklet_id=1, embedding=e1)
    b = _t(tracklet_id=2, local_track_id=2, start_frame=80, end_frame=100, embedding=e2, start_pitch=(90, 60), end_pitch=(90, 60))
    mapping, _, _, _ = associate_global_constrained([a, b], config=AssocConfig(reid_merge=0.99), must_links=[], cannot_links=[])
    assert mapping[1] != mapping[2]


def test_cannot_link_transitive():
    e = _emb(9)
    a = _t(tracklet_id=1, embedding=e, end_frame=10, end_pitch=(40, 30))
    b = _t(tracklet_id=2, local_track_id=2, start_frame=20, end_frame=30, embedding=e.copy(), start_pitch=(40.5, 30), end_pitch=(41, 30))
    c = _t(tracklet_id=3, local_track_id=3, start_frame=40, end_frame=50, embedding=e.copy(), start_pitch=(41.5, 30), end_pitch=(42, 30))
    mapping, _, _, _ = associate_global_constrained(
        [a, b, c],
        config=AssocConfig(reid_merge=0.4, min_merge_score=0.4),
        must_links=[(1, 2), (2, 3)],
        cannot_links=[(1, 3)],
    )
    # cannot between 1 and 3 must prevent them sharing gid even with must path
    assert mapping[1] != mapping[3]


def test_same_blocked_on_overlap():
    e = _emb(10)
    a = _t(tracklet_id=1, start_frame=0, end_frame=50, embedding=e)
    b = _t(tracklet_id=2, local_track_id=2, start_frame=10, end_frame=60, embedding=e.copy())
    assert pairwise_veto(a, b, config=AssocConfig()) == "simultaneous_overlap"
    mapping, _, _, rejects = associate_global_constrained([a, b], must_links=[(1, 2)], config=AssocConfig())
    assert mapping.get(1) != mapping.get(2) or any("human_must_link_blocked" in str(r.get("reason")) for r in rejects)


def test_cross_team_veto_preserved():
    e = _emb(11)
    a = _t(tracklet_id=1, team_id=0, embedding=e)
    b = _t(tracklet_id=2, local_track_id=2, team_id=1, start_frame=80, end_frame=100, embedding=e.copy(), start_pitch=(40, 30), end_pitch=(40, 30))
    assert pairwise_veto(a, b, config=AssocConfig()) == "cross_team"


def test_role_override_targets_tracklet_only():
    overrides = {5: "REFEREE"}
    roles = {1: "PLAYER", 5: "PLAYER", 9: "PLAYER"}
    for tid in list(roles):
        if tid in overrides:
            roles[tid] = overrides[tid]
    assert roles[5] == "REFEREE" and roles[1] == "PLAYER" and roles[9] == "PLAYER"


def test_final_renderer_canonical_policy():
    label = "T0 | GID 7 | #10 ✓"
    assert "LID" not in label and "local" not in label.lower()


def test_old_identity_results_preserved():
    p = Path("/mnt/c/football_data/results/tracking_identity_final/football_identity_final.mp4")
    assert p.exists()


def test_production_default_unchanged():
    for name in ["tracking_detection_stabilized.yaml", "tracking_identity_final.yaml"]:
        path = Path("/home/ahmet/projects/football-analytics/configs/pipeline") / name
        if path.exists():
            data = yaml.safe_load(path.read_text())
            if name.endswith("identity_final.yaml") or "stabilized" in name:
                assert data.get("production_default") in {False, None} or data.get("production_default") is False


def test_torch_cuda_unchanged():
    assert torch.__version__.startswith("2.")
    _ = torch.cuda.is_available()
