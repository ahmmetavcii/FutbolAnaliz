"""Tests for pitch person filter, role classifier, bbox smoother."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from football_analytics.detection.pitch_person_filter import (
    FilterDecision,
    PersonZone,
    PitchPersonFilter,
)
from football_analytics.tracking.bbox_smoother import BBoxSmoother
from football_analytics.tracking.track_role_classifier import TrackRoleClassifier

CALIB = Path("/home/ahmet/projects/football-analytics/configs/calibration/manual_football_frame100.json")


def _H():
    return np.array(json.loads(CALIB.read_text())["calibration"]["homography"], dtype=np.float64)


def test_stands_static_excluded():
    f = PitchPersonFilter(_H(), (1920, 1080))
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:] = (20, 20, 20)
    zr = f.classify(frame, 50, 10, 90, 70, frame_green=0.05)
    assert zr.zone in {PersonZone.STANDS, PersonZone.OFF_FIELD, PersonZone.UNKNOWN}
    if zr.zone == PersonZone.STANDS:
        assert zr.filter_decision == FilterDecision.EXCLUDE_FROM_PLAYER_TRACKING


def test_closeup_does_not_blind_reject():
    f = PitchPersonFilter(_H(), (1920, 1080))
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:] = (10, 10, 10)  # low green → closeup hint
    zr = f.classify(frame, 800, 400, 900, 700, frame_green=0.05)
    assert zr.zone == PersonZone.UNKNOWN
    assert zr.filter_decision == FilterDecision.HOLD_UNKNOWN


def test_referee_needs_track_level_not_single_frame():
    rc = TrackRoleClassifier()
    # short bench refs → not REFEREE
    for _ in range(3):
        rc.update(1, "referee", "BENCH_TECHNICAL_AREA", (100.0, 900.0), 0.9)
    row = rc.finalize()[0]
    assert row["role"] != "REFEREE"


def test_smoothing_resets_on_scene_cut():
    sm = BBoxSmoother()
    sm.smooth(1, (10, 10, 40, 70), frame_idx=0)
    sm.smooth(1, (12, 12, 42, 72), frame_idx=1)
    out = sm.smooth(1, (200, 200, 240, 270), frame_idx=2, scene_cut=True)
    assert abs(out[0] - 200) < 1e-6


def test_smoothing_does_not_fully_freeze_motion():
    sm = BBoxSmoother()
    sm.smooth(2, (0, 0, 40, 80), frame_idx=0)
    out = sm.smooth(2, (30, 0, 70, 80), frame_idx=1, confidence=1.0)
    # should move toward 30, not stay at 0
    assert out[0] > 5


def test_no_hard_cap_roles():
    rc = TrackRoleClassifier()
    for tid in range(15):
        for _ in range(10):
            rc.update(tid, "player", "ON_PITCH", (float(tid), 500.0), 0.8)
    rows = rc.finalize()
    assert len(rows) == 15
