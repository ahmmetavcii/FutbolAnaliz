"""Invariant tests for autonomous global identity association."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("/home/ahmet/workspace/autonomous_tracking_final/scripts")))
from lib_global_id import TrackletIdentity, associate_global  # noqa: E402


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(size=16)
    return e / np.linalg.norm(e)


def test_simultaneous_overlap_reject():
    e = _emb(1)
    a = TrackletIdentity(1, "player", 0, 0, 50, e, "10", "CONFIRMED", 0.9, None, "UNAVAILABLE", 0, (50, 30), (50, 30), (50, 30), 2.0, "HIGH", 0.8)
    b = TrackletIdentity(2, "player", 0, 10, 60, e.copy(), "10", "CONFIRMED", 0.9, None, "UNAVAILABLE", 0, (51, 30), (51, 30), (51, 30), 2.0, "HIGH", 0.8)
    _, audit, _ = associate_global([a, b])
    assert any(x.get("reason") == "simultaneous_overlap" for x in audit)


def test_cross_team_reject():
    e = _emb(2)
    a = TrackletIdentity(1, "player", 0, 0, 40, e, None, "UNREADABLE", 0, None, "UNAVAILABLE", 0, None, None, None, 1.5, "HIGH", 0.8)
    b = TrackletIdentity(2, "player", 1, 80, 120, e.copy(), None, "UNREADABLE", 0, None, "UNAVAILABLE", 0, None, None, None, 1.5, "HIGH", 0.8)
    _, audit, _ = associate_global([a, b])
    assert any(x.get("reason") == "cross_team" for x in audit)


def test_referee_player_reject():
    e = _emb(3)
    a = TrackletIdentity(1, "player", 0, 0, 40, e, None, "UNREADABLE", 0, None, "UNAVAILABLE", 0, None, None, None, 1.5, "HIGH", 0.8)
    b = TrackletIdentity(2, "referee", None, 80, 120, e.copy(), None, "UNREADABLE", 0, None, "UNAVAILABLE", 0, None, None, None, 1.5, "HIGH", 0.8)
    _, audit, _ = associate_global([a, b])
    assert any(x.get("reason") == "referee_player_mismatch" for x in audit)


def test_conflicting_jersey_reject():
    e = _emb(4)
    a = TrackletIdentity(1, "player", 0, 0, 40, e, "10", "CONFIRMED", 0.9, None, "UNAVAILABLE", 0, (50, 30), (50, 30), (50, 30), 1.5, "HIGH", 0.8)
    b = TrackletIdentity(2, "player", 0, 80, 120, e.copy(), "7", "CONFIRMED", 0.9, None, "UNAVAILABLE", 0, (50, 30), (50, 30), (50, 30), 1.5, "HIGH", 0.8)
    _, audit, _ = associate_global([a, b])
    assert any(x.get("reason") == "conflicting_jersey" for x in audit)


def test_no_hard_cap_on_team_size():
    many = []
    for i in range(15):
        e = np.zeros(32, dtype=np.float64)
        e[i] = 1.0
        many.append(
            TrackletIdentity(
                100 + i,
                "player",
                0,
                i * 200,
                i * 200 + 30,
                e,
                None,
                "UNREADABLE",
                0,
                None,
                "UNAVAILABLE",
                0,
                None,
                None,
                None,
                1.0,
                "MEDIUM",
                0.5,
            )
        )
    _, _, players = associate_global(many, fps=25, reid_merge=0.95, reid_strong=0.99)
    assert len(players) == 15


def test_impossible_motion_reject():
    e = _emb(5)
    a = TrackletIdentity(1, "player", 0, 0, 25, e, None, "UNREADABLE", 0, None, "UNAVAILABLE", 0, (5, 5), (5, 5), (5, 5), 1.0, "HIGH", 0.8)
    # 1 second later at other end of pitch → impossible
    b = TrackletIdentity(2, "player", 0, 50, 75, e.copy(), None, "UNREADABLE", 0, None, "UNAVAILABLE", 0, (100, 60), (100, 60), (100, 60), 1.0, "HIGH", 0.8)
    # attach end pitch on first via association path — create first then try second
    # Use associate: first creates player with _end_pitch
    m, audit, players = associate_global([a, b], fps=25.0, max_speed_mps=8.0)
    # either rejected impossible or separate IDs
    assert m.get(1) != m.get(2) or any(x.get("reason") == "impossible_motion" for x in audit)
