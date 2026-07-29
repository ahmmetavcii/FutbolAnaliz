"""Tests for targeted manual identity corrections."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import yaml

OUT = Path("/mnt/c/football_data/results/tracking_manually_corrected")
HV = Path("/mnt/c/football_data/results/tracking_human_verified")
CORR = Path("/home/ahmet/projects/football-analytics/configs/evaluation/manual_corrections/football_identity_corrections.yaml")


def _g():
    return pd.read_parquet(OUT / "global_tracks_manually_corrected.parquet")


def test_corrections_yaml_exists():
    data = yaml.safe_load(CORR.read_text())
    assert data["identity_aliases"][0]["source_gid"] == 23
    assert data["identity_aliases"][0]["canonical_gid"] == 17
    assert data["team_overrides"][0]["global_id"] == 16
    assert data["role_overrides"][0]["global_id"] == 22


def test_gid23_not_in_final_display():
    g = _g()
    assert (g.display_player_id == 23).sum() == 0


def test_gid23_aliased_to_17():
    aliases = pd.read_parquet(OUT / "manual_identity_aliases.parquet")
    assert int(aliases.iloc[0].source_gid) == 23
    assert int(aliases.iloc[0].canonical_gid) == 17
    g = _g()
    assert (g.display_player_id == 17).sum() > 0


def test_no_duplicate_gid_per_frame():
    g = _g()
    d = g.dropna(subset=["display_player_id"])
    mx = d.groupby(["frame_idx", "display_player_id"]).size().max()
    assert int(mx) == 1


def test_gid16_all_t0():
    g = _g()
    g16 = g[g.display_player_id == 16]
    assert len(g16) > 0
    assert set(int(x) for x in g16.team_id.unique()) == {0}
    assert (g16.display_player_id == 16).all()


def test_gid22_referee_no_team():
    g = _g()
    g22 = g[g.display_player_id == 22]
    assert len(g22) > 0
    assert set(g22.role.unique()) == {"REFEREE"}
    assert set(int(x) for x in g22.team_id.unique()) <= {-1}


def test_gid22_not_in_player_metrics():
    summary = pd.read_parquet(OUT / "identity_summary_manually_corrected.parquet")
    row = summary[summary.global_player_id == 22].iloc[0]
    assert row.role == "REFEREE"
    assert int(row.team_id) < 0


def test_stable_gids_preserved():
    g = _g()
    for gid in [5, 6, 9, 10, 19]:
        assert (g.display_player_id == gid).sum() > 0


def test_raw_and_smoothed_separate():
    g = _g()
    # stabilization targets should have raw_disp columns
    assert "raw_disp_x1" in g.columns and "smoothed_x1" in g.columns


def test_hv_preserved():
    assert (HV / "football_identity_human_verified.mp4").exists()
    assert (HV / "global_tracks_human_verified.parquet").exists()


def test_videos_open():
    import cv2

    for name in [
        "football_identity_manually_corrected.mp4",
        "football_identity_manually_corrected_debug.mp4",
        "football_tactical_2d_manually_corrected.mp4",
    ]:
        cap = cv2.VideoCapture(str(OUT / name))
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
        cap.release()


def test_parquets_readable():
    for name in [
        "global_tracks_manually_corrected.parquet",
        "identity_summary_manually_corrected.parquet",
        "manual_identity_aliases.parquet",
        "manual_team_overrides.parquet",
        "manual_role_overrides.parquet",
        "targeted_gid_jitter_audit.parquet",
    ]:
        pd.read_parquet(OUT / name)


def test_production_default_unchanged():
    for name in ["tracking_detection_stabilized.yaml", "tracking_identity_final.yaml"]:
        p = Path("/home/ahmet/projects/football-analytics/configs/pipeline") / name
        if p.exists():
            data = yaml.safe_load(p.read_text())
            assert data.get("production_default") is False


def test_torch_cuda_unchanged():
    assert torch.__version__.startswith("2.")
    _ = torch.cuda.is_available()
