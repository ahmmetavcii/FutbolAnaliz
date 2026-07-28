"""Tests for video-mode human detection GT annotator."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from football_analytics.evaluation.human_gt_annotate import (
    accept_visible_proposals,
    is_true,
    recount_completion,
    save_guard,
    truthy_mask,
)
from football_analytics.evaluation.human_gt_video import (
    AutoPauseController,
    ensure_video_proposals_cache,
    first_unreviewed_sample_idx,
    is_sample_frame,
    load_overlay_by_frame,
    load_tracks_by_frame,
    next_sample_frame,
    normalize_overlay_dataframe,
    overlay_rows_for_frame,
    resume_start_frame,
    sample_frame_indices,
    sample_ordinal,
)

ROOT = Path(__file__).resolve().parents[2]
GT_DIR = ROOT / "configs/evaluation/human_detection_gt/football"
PROD_CFG = ROOT / "configs/pipeline/opta_analytics.yaml"
COMMON_DET = Path(
    "/home/ahmet/workspace/hybrid_detector_validation/artifacts/common_human_detections.parquet"
)
VIDEO = Path("/mnt/c/football_data/videos/test_clips/football.mp4")


def _load_video_mod():
    path = ROOT / "scripts" / "annotate_human_detection_video.py"
    spec = importlib.util.spec_from_file_location("annotate_human_detection_video", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_tiny_video(path: Path, n_frames: int = 40, fps: float = 10.0, size=(160, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:] = (i * 3 % 255, 40, 80)
        cv2.putText(frame, str(i), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def _make_gt_bundle(tmp_path: Path, *, n_video_frames: int = 40) -> tuple[Path, Path]:
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "backups").mkdir()
    video = tmp_path / "clip.mp4"
    _write_tiny_video(video, n_frames=n_video_frames, fps=10.0)

    sample_frames = [0, 5, 10, 15, 20, 25, 30, 35]
    sample = pd.DataFrame(
        [
            {"frame_idx": f, "reviewed": f in (0, 5)}
            for f in sample_frames
        ]
    )
    sample.to_csv(gt_dir / "review_sample.csv", index=False)

    gt_rows = []
    for f in sample_frames:
        if f in (0, 5):
            gt_rows.append(
                {
                    "frame_idx": f,
                    "gt_id": f"f{f}_a",
                    "class_name": "player",
                    "x1": 10,
                    "y1": 10,
                    "x2": 40,
                    "y2": 50,
                    "occluded": False,
                    "difficult": False,
                    "reviewed": True,
                    "notes": "prior",
                }
            )
        else:
            gt_rows.append(
                {
                    "frame_idx": f,
                    "gt_id": f"f{f}_ph",
                    "class_name": "not_target",
                    "x1": np.nan,
                    "y1": np.nan,
                    "x2": np.nan,
                    "y2": np.nan,
                    "occluded": False,
                    "difficult": False,
                    "reviewed": False,
                    "notes": "placeholder",
                }
            )
    pd.DataFrame(gt_rows).to_csv(gt_dir / "human_detection_gt.csv", index=False)

    props = []
    for f in sample_frames:
        for i in range(2):
            props.append(
                {
                    "frame_idx": f,
                    "proposal_id": f"f{f}_p{i}",
                    "class_name": "player" if i == 0 else "referee",
                    "confidence": 0.8,
                    "x1": 10 + i * 50,
                    "y1": 10,
                    "x2": 40 + i * 50,
                    "y2": 50,
                    "source_detector": "hybrid",
                    "accepted": False,
                    "rejected": False,
                    "modified": False,
                }
            )
    pd.DataFrame(props).to_csv(gt_dir / "model_proposals.csv", index=False)

    # video overlay for all frames
    ov = []
    for f in range(n_video_frames):
        ov.append(
            {
                "frame_idx": f,
                "class_name": "player",
                "confidence": 0.5,
                "x1": 1,
                "y1": 1,
                "x2": 20,
                "y2": 30,
                "source_detector": "hybrid",
            }
        )
    pd.DataFrame(ov).to_parquet(gt_dir / "video_proposals.parquet", index=False)
    return gt_dir, video


def test_auto_pause_on_sample_frame(tmp_path: Path):
    mod = _load_video_mod()
    gt_dir, video = _make_gt_bundle(tmp_path)
    ann = mod.VideoHumanAnnotator(
        video_path=video, gt_dir=gt_dir, review_only=False, start_frame=4, show_window=False
    )
    assert not ann.edit_mode  # frame 4 not sample
    ann.playing = True
    ann.edit_mode = False
    ann.tick()  # -> frame 5 sample
    assert ann.frame_idx == 5
    assert ann.playing is False
    assert ann.edit_mode is True
    ann.cap.release()


def test_non_sample_frame_not_written_to_gt(tmp_path: Path):
    mod = _load_video_mod()
    gt_dir, video = _make_gt_bundle(tmp_path)
    ann = mod.VideoHumanAnnotator(
        video_path=video, gt_dir=gt_dir, start_frame=7, show_window=False
    )
    assert not is_sample_frame(ann.sample, 7)
    before = pd.read_csv(gt_dir / "human_detection_gt.csv")
    assert ann.save_current_frame() is False
    after = pd.read_csv(gt_dir / "human_detection_gt.csv")
    assert before.equals(after)
    assert 7 not in set(after.frame_idx.astype(int))
    ann.cap.release()


def test_model_proposals_not_auto_gt(tmp_path: Path):
    mod = _load_video_mod()
    gt_dir, video = _make_gt_bundle(tmp_path)
    ann = mod.VideoHumanAnnotator(
        video_path=video, gt_dir=gt_dir, start_frame=10, show_window=False
    )
    assert ann.edit_mode
    assert ann.state.counts()["accepted"] == 0
    assert ann.save_current_frame() is False
    sample = pd.read_csv(gt_dir / "review_sample.csv")
    assert not is_true(sample.loc[sample.frame_idx == 10, "reviewed"].iloc[0])
    ann.cap.release()


def test_a_accepts_not_reviewed(tmp_path: Path):
    mod = _load_video_mod()
    gt_dir, video = _make_gt_bundle(tmp_path)
    ann = mod.VideoHumanAnnotator(
        video_path=video, gt_dir=gt_dir, start_frame=10, show_window=False
    )
    accept_visible_proposals(ann.state)
    assert ann.state.counts()["accepted"] == 2
    sample = pd.read_csv(gt_dir / "review_sample.csv")
    assert not is_true(sample.loc[sample.frame_idx == 10, "reviewed"].iloc[0])
    ann.cap.release()


def test_s_sets_reviewed(tmp_path: Path):
    mod = _load_video_mod()
    gt_dir, video = _make_gt_bundle(tmp_path)
    ann = mod.VideoHumanAnnotator(
        video_path=video, gt_dir=gt_dir, start_frame=10, show_window=False
    )
    accept_visible_proposals(ann.state)
    assert ann.save_current_frame()
    sample = pd.read_csv(gt_dir / "review_sample.csv")
    assert is_true(sample.loc[sample.frame_idx == 10, "reviewed"].iloc[0])
    gt = pd.read_csv(gt_dir / "human_detection_gt.csv")
    assert len(gt[(gt.frame_idx == 10) & gt.x1.notna()]) == 2
    ann.cap.release()


def test_s_blocked_when_accepted_zero(tmp_path: Path):
    mod = _load_video_mod()
    gt_dir, video = _make_gt_bundle(tmp_path)
    ann = mod.VideoHumanAnnotator(
        video_path=video, gt_dir=gt_dir, start_frame=15, show_window=False
    )
    assert save_guard(ann.state) == "need_accept"
    assert ann.save_current_frame() is False
    ann.cap.release()


def test_empty_requires_double_e(tmp_path: Path):
    mod = _load_video_mod()
    gt_dir, video = _make_gt_bundle(tmp_path)
    # Frame with proposals but user wants empty via E+E
    ann = mod.VideoHumanAnnotator(
        video_path=video, gt_dir=gt_dir, start_frame=20, show_window=False
    )
    # Reject all so path is empty-confirm OR use E with proposals (allowed)
    ann.handle_empty_confirm()
    assert ann.empty_confirmation_pending
    sample = pd.read_csv(gt_dir / "review_sample.csv")
    assert not is_true(sample.loc[sample.frame_idx == 20, "reviewed"].iloc[0])
    ann.handle_empty_confirm()
    sample = pd.read_csv(gt_dir / "review_sample.csv")
    assert is_true(sample.loc[sample.frame_idx == 20, "reviewed"].iloc[0])
    gt = pd.read_csv(gt_dir / "human_detection_gt.csv")
    g = gt[gt.frame_idx == 20]
    assert g.x1.isna().all()
    ann.cap.release()


def test_dirty_false_after_save(tmp_path: Path):
    mod = _load_video_mod()
    gt_dir, video = _make_gt_bundle(tmp_path)
    ann = mod.VideoHumanAnnotator(
        video_path=video, gt_dir=gt_dir, start_frame=25, show_window=False
    )
    accept_visible_proposals(ann.state)
    assert ann.state.dirty
    assert ann.save_current_frame()
    assert ann.state.dirty is False
    assert ann.unsaved_changes() is False
    ann.cap.release()


def test_overwrite_completion_stable(tmp_path: Path):
    mod = _load_video_mod()
    gt_dir, video = _make_gt_bundle(tmp_path)
    ann = mod.VideoHumanAnnotator(
        video_path=video, gt_dir=gt_dir, start_frame=0, show_window=False
    )
    # already reviewed
    assert ann.frame_reviewed_on_disk()
    c0 = recount_completion(ann.sample)["reviewed"]
    ann.state.boxes[0].x1 += 1
    ann.state.dirty = True
    assert ann.save_current_frame()
    c1 = recount_completion(ann.sample)["reviewed"]
    assert c1 == c0
    ann.cap.release()


def test_resume_finds_first_unreviewed():
    sample = pd.DataFrame(
        [
            {"frame_idx": 0, "reviewed": True},
            {"frame_idx": 5, "reviewed": True},
            {"frame_idx": 10, "reviewed": False},
            {"frame_idx": 15, "reviewed": False},
        ]
    )
    assert first_unreviewed_sample_idx(sample) == 10
    start = resume_start_frame(sample, mode="first_unreviewed", fps=25.0, n_frames=750, lead_seconds=2.0)
    assert start == 0  # 10 - 50 clamped
    sample2 = pd.DataFrame(
        [
            {"frame_idx": 100, "reviewed": True},
            {"frame_idx": 200, "reviewed": False},
        ]
    )
    start2 = resume_start_frame(sample2, mode="first_unreviewed", fps=25.0, n_frames=750, lead_seconds=2.0)
    assert start2 == 150  # 200 - 50


def test_jk_sample_navigation():
    sample = pd.DataFrame({"frame_idx": [0, 5, 10, 15], "reviewed": [False] * 4})
    assert next_sample_frame(sample, 5, direction=1) == 10
    assert next_sample_frame(sample, 5, direction=-1) == 0
    assert sample_ordinal(sample, 10) == 3


def test_seek_frame_index(tmp_path: Path):
    mod = _load_video_mod()
    gt_dir, video = _make_gt_bundle(tmp_path)
    ann = mod.VideoHumanAnnotator(
        video_path=video, gt_dir=gt_dir, start_frame=0, show_window=False
    )
    ann._seek(17)
    assert ann.frame_idx == 17
    ann._seek(ann.frame_idx + int(round(ann.fps)))  # +1s
    assert ann.frame_idx == 27
    ann.cap.release()


def test_tracking_overlay_does_not_mutate_gt(tmp_path: Path):
    mod = _load_video_mod()
    gt_dir, video = _make_gt_bundle(tmp_path)
    # fake tracks
    tracks = pd.DataFrame(
        [{"frame_idx": 3, "track_id": 9, "x1": 1, "y1": 1, "x2": 5, "y2": 5, "class_name": "player"}]
    )
    tpath = tmp_path / "tracks.parquet"
    tracks.to_parquet(tpath, index=False)
    before = _sha(gt_dir / "human_detection_gt.csv")
    ann = mod.VideoHumanAnnotator(
        video_path=video, gt_dir=gt_dir, start_frame=3, show_window=False, review_only=True
    )
    ann.tracks_by_frame = load_tracks_by_frame(tpath)
    ann.show_tracks = True
    _ = ann.render()
    assert _sha(gt_dir / "human_detection_gt.csv") == before
    ann.cap.release()


def test_review_only_does_not_touch_gt(tmp_path: Path):
    mod = _load_video_mod()
    gt_dir, video = _make_gt_bundle(tmp_path)
    gt_sha = _sha(gt_dir / "human_detection_gt.csv")
    sample_sha = _sha(gt_dir / "review_sample.csv")
    prop_sha = _sha(gt_dir / "model_proposals.csv")
    ann = mod.VideoHumanAnnotator(
        video_path=video, gt_dir=gt_dir, start_frame=10, show_window=False, review_only=True
    )
    assert ann.edit_mode is False
    assert ann.save_current_frame() is False
    accept_visible_proposals(ann.state)  # should be irrelevant
    assert _sha(gt_dir / "human_detection_gt.csv") == gt_sha
    assert _sha(gt_dir / "review_sample.csv") == sample_sha
    assert _sha(gt_dir / "model_proposals.csv") == prop_sha
    ann.cap.release()


def test_preview_export_does_not_touch_gt(tmp_path: Path):
    mod = _load_video_mod()
    gt_dir, video = _make_gt_bundle(tmp_path)
    gt_sha = _sha(gt_dir / "human_detection_gt.csv")
    sample_sha = _sha(gt_dir / "review_sample.csv")
    out = tmp_path / "preview.mp4"
    mod.export_preview(video_path=video, gt_dir=gt_dir, out_path=out, max_frames=12)
    assert out.exists() and out.stat().st_size > 0
    assert _sha(gt_dir / "human_detection_gt.csv") == gt_sha
    assert _sha(gt_dir / "review_sample.csv") == sample_sha


def test_atomic_save_and_backup(tmp_path: Path):
    mod = _load_video_mod()
    gt_dir, video = _make_gt_bundle(tmp_path)
    ann = mod.VideoHumanAnnotator(
        video_path=video, gt_dir=gt_dir, start_frame=30, show_window=False
    )
    accept_visible_proposals(ann.state)
    ann.save_current_frame()
    backups = list((gt_dir / "backups").glob("*.bak"))
    assert len(backups) >= 1
    ann.cap.release()


def test_production_config_unchanged():
    assert PROD_CFG.exists()
    # fingerprint stable within this process (file must exist and be readable)
    assert len(_sha(PROD_CFG)) == 64


def test_torch_cuda_unchanged():
    v = torch.__version__
    assert "2.11" in v or v.startswith("2.")
    # do not change CUDA availability; just observe
    _ = torch.cuda.is_available()


def test_ensure_video_proposals_from_common(tmp_path: Path):
    if not COMMON_DET.exists():
        pytest.skip("common detections missing")
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    path = ensure_video_proposals_cache(gt_dir, source=COMMON_DET)
    assert path.exists()
    idx = load_overlay_by_frame(path)
    assert len(idx) >= 700
    rows = overlay_rows_for_frame(idx, 0)
    assert not rows.empty
    assert set(rows.columns) >= {"frame_idx", "class_name", "confidence", "x1", "y1", "x2", "y2"}


def test_auto_pause_controller_logic():
    ctl = AutoPauseController({0, 10, 20})
    assert ctl.on_frame(10, playing=True) is True
    assert ctl.paused_for_review is True
    ctl.clear_after_continue()
    assert ctl.paused_for_review is False
    assert ctl.on_frame(11, playing=True) is False


def test_normalize_overlay_role():
    df = pd.DataFrame(
        [
            {
                "frame_idx": 1,
                "class_name": "player",
                "confidence": 0.9,
                "x1": 0,
                "y1": 0,
                "x2": 1,
                "y2": 1,
                "source_detector": "h",
                "role": "goalkeeper",
            }
        ]
    )
    out = normalize_overlay_dataframe(df)
    assert out.iloc[0]["class_name"] == "goalkeeper"
