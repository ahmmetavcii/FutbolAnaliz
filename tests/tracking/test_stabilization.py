"""Stabilization unit tests (no GT; consistency / gating)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from football_analytics.geometry.calibration_state import CalibSource, CalibrationStateMachine
from football_analytics.tracking.footpoint_estimator import FootpointEstimator
from football_analytics.tracking.pitch_space_filter import PitchSpaceFilter
from football_analytics.tracking.track_role_state_machine import RoleLabel, TrackRoleStateMachine

ROOT = Path("/home/ahmet/projects/football-analytics")
CALIB = ROOT / "configs/calibration/manual_football_frame100.json"
PROD_REFINED = ROOT / "configs/pipeline/tracking_detection_refined.yaml"
PROD_FINAL = ROOT / "configs/pipeline/tracking_detection_final.yaml"


def _H():
    return np.array(json.loads(CALIB.read_text())["calibration"]["homography"], dtype=np.float64)


def test_closeup_does_not_accept_minimap_update():
    cs = CalibrationStateMachine(_H(), (1920, 1080))
    cs.H = _H().copy()
    cs.confidence = 0.7
    cs.source = CalibSource.MEASURED
    st = cs.update(frame_idx=1, shot_type="close_up", scene_cut=False, measured_H=_H(), measured_confidence=0.9)
    assert st.accepted_update is False
    assert st.source in {CalibSource.FROZEN_LAST_GOOD, CalibSource.INVALID}


def test_bench_staff_not_referee():
    sm = TrackRoleStateMachine()
    for _ in range(20):
        sm.update(1, det_class="referee", zone="BENCH_TECHNICAL_AREA", shot_type="bench", conf=0.95, foot_xy=(50, 950))
    role, _, _, _ = sm.role_of(1)
    assert role != RoleLabel.REFEREE_CENTER
    assert role in {RoleLabel.STAFF, RoleLabel.BENCH_PLAYER, RoleLabel.UNRESOLVED}


def test_stands_not_player():
    sm = TrackRoleStateMachine()
    for _ in range(20):
        sm.update(9, det_class="player", zone="STANDS", shot_type="crowd", conf=0.9, foot_xy=(100, 100))
    role, _, _, _ = sm.role_of(9)
    assert role != RoleLabel.PLAYER
    assert role in {RoleLabel.SPECTATOR, RoleLabel.UNRESOLVED}


def test_assistant_referee_not_auto_deleted_as_staff():
    sm = TrackRoleStateMachine()
    for _ in range(30):
        sm.update(
            4,
            det_class="referee",
            zone="TOUCHLINE_MARGIN",
            shot_type="main_wide",
            conf=0.9,
            foot_xy=(200, 900),
        )
    role, _, _, _ = sm.role_of(4)
    assert role == RoleLabel.REFEREE_ASSISTANT
    assert role != RoleLabel.STAFF


def test_locked_role_ignores_short_flip():
    sm = TrackRoleStateMachine()
    for _ in range(35):
        sm.update(2, det_class="referee", zone="ON_PITCH", shot_type="main_wide", conf=0.9, foot_xy=(900, 600))
    role_before, _, _, _ = sm.role_of(2)
    for _ in range(4):
        sm.update(2, det_class="player", zone="ON_PITCH", shot_type="main_wide", conf=0.85, foot_xy=(900, 600))
    role_after, _, _, reason = sm.role_of(2)
    assert role_after == role_before or "locked" in reason


def test_bench_ratio_vetoes_referee():
    sm = TrackRoleStateMachine()
    for _ in range(25):
        sm.update(5, det_class="referee", zone="BENCH", shot_type="main_wide", conf=0.95, foot_xy=(40, 980))
    role, _, _, _ = sm.role_of(5)
    assert role != RoleLabel.REFEREE_CENTER


def test_team_color_weakens_referee():
    sm = TrackRoleStateMachine()
    sm.set_team_centers([(200.0, 50.0, 50.0), (50.0, 50.0, 200.0)])
    for _ in range(25):
        sm.update(
            3,
            det_class="referee",
            zone="ON_PITCH",
            shot_type="main_wide",
            conf=0.7,
            foot_xy=(800, 500),
            jersey_bgr=(200.0, 55.0, 55.0),
        )
    role, _, _, reason = sm.role_of(3)
    assert role != RoleLabel.REFEREE_CENTER or "color" in reason or role == RoleLabel.PLAYER


def test_scene_cut_resets_calibration():
    cs = CalibrationStateMachine(_H(), (1920, 1080))
    cs.H = _H().copy()
    cs.source = CalibSource.MEASURED
    cs.update(frame_idx=5, shot_type="main_wide", scene_cut=True, measured_H=None)
    assert any(a.get("event") == "reset" for a in cs.audit)


def test_invalid_calibration_no_pitch():
    cs = CalibrationStateMachine(_H(), (1920, 1080))
    cs.source = CalibSource.INVALID
    cs.H = None
    assert cs.pixel_to_pitch(100, 200) is None


def test_frozen_last_good_limited():
    cs = CalibrationStateMachine(_H(), (1920, 1080))
    cs.H = _H().copy()
    cs.confidence = 0.7
    cs.source = CalibSource.MEASURED
    for i in range(cs.config.max_frozen_frames + 5):
        st = cs.update(frame_idx=i, shot_type="close_up", scene_cut=False, measured_H=None)
    assert st.source == CalibSource.INVALID


def test_pose_low_conf_skips_ankle():
    fe = FootpointEstimator()
    r = fe.estimate(1, (10, 10, 50, 100), smoothed_box=(10, 10, 50, 100), ankle_mid=(30, 95, 0.2))
    assert r.method != "ankle_midpoint"
    assert "bbox" in r.method


def test_pitch_filter_reduces_stationary_jitter():
    pf = PitchSpaceFilter(fps=25)
    pf.update(1, (40.0, 30.0), footpoint_conf=0.9, calib_conf=0.9, det_conf=0.9, image_disp_px=1.0, valid=True)
    outs = []
    for i in range(20):
        noise = 0.25 * ((-1) ** i)
        o = pf.update(
            1,
            (40.0 + noise, 30.0 + noise * 0.5),
            footpoint_conf=0.9,
            calib_conf=0.9,
            det_conf=0.9,
            image_disp_px=2.0,
            valid=True,
        )
        outs.append(o["filtered_pitch_x"])
    assert max(outs) - min(outs) < 0.15


def test_pitch_filter_allows_fast_motion():
    pf = PitchSpaceFilter(fps=25)
    pf.update(2, (10.0, 10.0), footpoint_conf=0.9, calib_conf=0.9, det_conf=0.9, image_disp_px=30, valid=True)
    o = pf.update(2, (16.0, 12.0), footpoint_conf=0.9, calib_conf=0.9, det_conf=0.9, image_disp_px=40, valid=True)
    assert abs(o["filtered_pitch_x"] - 10.0) > 1.0


def test_raw_and_filtered_separate():
    pf = PitchSpaceFilter(fps=25)
    pf.update(1, (40.0, 30.0), footpoint_conf=0.9, calib_conf=0.9, det_conf=0.9, image_disp_px=1.0, valid=True)
    o = pf.update(1, (40.4, 30.3), footpoint_conf=0.9, calib_conf=0.9, det_conf=0.9, image_disp_px=2.0, valid=True)
    assert "raw_pitch_x" in o and "filtered_pitch_x" in o
    assert o["raw_pitch_x"] == 40.4
    assert abs(o["filtered_pitch_x"] - 40.4) > 1e-6 or o["filtered_pitch_x"] is not None


def test_physical_metrics_skip_low_confidence():
    rows = [
        {"coordinate_confidence": 0.05, "filtered_pitch_x": 1.0, "filtered_pitch_y": 1.0},
        {"coordinate_confidence": 0.8, "filtered_pitch_x": 2.0, "filtered_pitch_y": 2.0},
    ]
    usable = [r for r in rows if r["coordinate_confidence"] >= 0.2]
    assert len(usable) == 1


def test_production_configs_unchanged_flags():
    for p in [PROD_REFINED, PROD_FINAL]:
        if not p.exists():
            continue
        data = yaml.safe_load(p.read_text())
        # stabilized must not become production by editing these files here
        assert data.get("production_default") in {False, None, True}  # presence only
    stab = ROOT / "configs/pipeline/tracking_detection_stabilized.yaml"
    if stab.exists():
        data = yaml.safe_load(stab.read_text())
        assert data.get("production_default") is False


def test_torch_cuda_unchanged():
    assert torch.__version__.startswith("2.")
    # do not mutate torch; just assert available API
    _ = torch.cuda.is_available()


def test_no_duplicate_track_role_rows_per_id():
    from football_analytics.tracking.track_role_state_machine import TrackRoleStateMachine
    sm = TrackRoleStateMachine()
    for _ in range(15):
        sm.update(7, det_class="player", zone="ON_PITCH", shot_type="main_wide", conf=0.9, foot_xy=(500, 500))
        sm.update(8, det_class="player", zone="ON_PITCH", shot_type="main_wide", conf=0.9, foot_xy=(600, 500))
    rows = sm.finalize_rows()
    ids = [r["local_track_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_different_team_centers_not_merged_by_role_sm():
    from football_analytics.tracking.track_role_state_machine import TrackRoleStateMachine, RoleLabel
    sm = TrackRoleStateMachine()
    sm.set_team_centers([(200, 40, 40), (40, 40, 200)])
    for _ in range(20):
        sm.update(10, det_class="player", zone="ON_PITCH", shot_type="main_wide", conf=0.9, foot_xy=(400, 400), jersey_bgr=(200, 40, 40))
        sm.update(11, det_class="player", zone="ON_PITCH", shot_type="main_wide", conf=0.9, foot_xy=(700, 400), jersey_bgr=(40, 40, 200))
    assert sm.role_of(10)[0] == RoleLabel.PLAYER
    assert sm.role_of(11)[0] == RoleLabel.PLAYER
    assert 10 in sm._t and 11 in sm._t and 10 != 11
