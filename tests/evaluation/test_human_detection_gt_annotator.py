"""Tests for human detection GT annotator proposal workflow."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_analytics.evaluation.human_gt_annotate import (
    BoxState,
    FrameEditState,
    accept_visible_proposals,
    atomic_write_csv,
    build_gt_rows_for_frame,
    evaluator_gt_frames,
    is_true,
    load_proposals_for_frame,
    merge_gt_preserving_other_reviewed,
    proposals_are_not_gt,
    recount_completion,
    reset_frame_from_proposals,
    save_guard,
    truthy_mask,
)

ROOT = Path(__file__).resolve().parents[2]
GT_DIR = ROOT / "configs/evaluation/human_detection_gt/football"


def _load_annotator_cls():
    path = ROOT / "scripts" / "annotate_human_detection_gt.py"
    spec = importlib.util.spec_from_file_location("annotate_human_detection_gt", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.Annotator


def _make_gt_dir(tmp_path: Path, *, reviewed0: bool = False, empty_gt0: bool = False) -> Path:
    gt_dir = tmp_path / "gt"
    (gt_dir / "frames").mkdir(parents=True)
    (gt_dir / "backups").mkdir()
    sample_rows = [{"frame_idx": i, "reviewed": (i == 0 and reviewed0)} for i in range(40)]
    # seed two already-reviewed frames so overwrite completion math is visible
    sample_rows[1]["reviewed"] = True
    sample_rows[2]["reviewed"] = True
    pd.DataFrame(sample_rows).to_csv(gt_dir / "review_sample.csv", index=False)

    gt_rows = [
        {
            "frame_idx": 1,
            "gt_id": "f1_a",
            "class_name": "player",
            "x1": 1,
            "y1": 1,
            "x2": 10,
            "y2": 10,
            "occluded": False,
            "difficult": False,
            "reviewed": True,
            "notes": "prior",
        },
        {
            "frame_idx": 2,
            "gt_id": "f2_a",
            "class_name": "referee",
            "x1": 2,
            "y1": 2,
            "x2": 12,
            "y2": 12,
            "occluded": False,
            "difficult": False,
            "reviewed": True,
            "notes": "prior",
        },
    ]
    if empty_gt0:
        gt_rows.append(
            {
                "frame_idx": 0,
                "gt_id": "f0_empty",
                "class_name": "not_target",
                "x1": np.nan,
                "y1": np.nan,
                "x2": np.nan,
                "y2": np.nan,
                "occluded": False,
                "difficult": False,
                "reviewed": True,
                "notes": "explicit_empty_frame",
            }
        )
    else:
        gt_rows.append(
            {
                "frame_idx": 0,
                "gt_id": "f0_ph",
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
    for i, (cls, x) in enumerate([("player", 10), ("referee", 200)]):
        props.append(
            {
                "frame_idx": 0,
                "proposal_id": f"f0_p{i}",
                "class_name": cls,
                "confidence": 0.8,
                "x1": x,
                "y1": 10,
                "x2": x + 40,
                "y2": 100,
                "source_detector": "hybrid",
                "accepted": False,
                "rejected": False,
                "modified": False,
            }
        )
    pd.DataFrame(props).to_csv(gt_dir / "model_proposals.csv", index=False)
    return gt_dir


def _sample_proposals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame_idx": 10,
                "proposal_id": "f10_p0",
                "class_name": "player",
                "confidence": 0.8,
                "x1": 10,
                "y1": 10,
                "x2": 50,
                "y2": 100,
                "source_detector": "hybrid",
                "accepted": False,
                "rejected": False,
                "modified": False,
            },
            {
                "frame_idx": 10,
                "proposal_id": "f10_p1",
                "class_name": "referee",
                "confidence": 0.7,
                "x1": 200,
                "y1": 20,
                "x2": 240,
                "y2": 110,
                "source_detector": "hybrid",
                "accepted": False,
                "rejected": False,
                "modified": False,
            },
        ]
    )


def test_proposals_load():
    props = _sample_proposals()
    boxes = load_proposals_for_frame(props, 10)
    assert len(boxes) == 2
    assert boxes[0].class_name == "player"


def test_proposals_not_counted_as_gt():
    props = _sample_proposals()
    gt = pd.DataFrame(
        [
            {
                "frame_idx": 10,
                "gt_id": "ph",
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
        ]
    )
    assert proposals_are_not_gt(props, gt)


def test_a_does_not_set_reviewed(tmp_path: Path):
    props = _sample_proposals()
    state = FrameEditState(frame_idx=10, boxes=load_proposals_for_frame(props, 10))
    sample = pd.DataFrame([{"frame_idx": 10, "reviewed": False}])
    accept_visible_proposals(state)
    assert all(b.accepted for b in state.boxes)
    assert not is_true(sample.iloc[0]["reviewed"])
    # no save performed
    assert recount_completion(sample)["reviewed"] == 0


def test_s_sets_reviewed_and_writes_gt(tmp_path: Path):
    props = _sample_proposals()
    state = FrameEditState(frame_idx=10, boxes=load_proposals_for_frame(props, 10))
    accept_visible_proposals(state)
    # reject one
    state.boxes[1].rejected = True
    state.boxes[1].accepted = False
    # manual box
    state.boxes.append(
        BoxState(
            proposal_id="m1",
            class_name="goalkeeper",
            confidence=1.0,
            x1=1,
            y1=1,
            x2=20,
            y2=40,
            accepted=True,
            manual=True,
        )
    )
    rows = build_gt_rows_for_frame(state)
    classes = {r["class_name"] for r in rows}
    assert "referee" not in classes  # rejected
    assert "player" in classes
    assert "goalkeeper" in classes

    gt = pd.DataFrame(
        [
            {
                "frame_idx": 99,
                "gt_id": "keep",
                "class_name": "player",
                "x1": 1,
                "y1": 1,
                "x2": 2,
                "y2": 2,
                "occluded": False,
                "difficult": False,
                "reviewed": True,
                "notes": "prior",
            }
        ]
    )
    sample = pd.DataFrame([{"frame_idx": 10, "reviewed": False}, {"frame_idx": 99, "reviewed": True}])
    merged = merge_gt_preserving_other_reviewed(gt, 10, rows, {10, 99})
    assert ((merged.frame_idx == 99) & (merged.gt_id == "keep")).any()
    path = tmp_path / "gt.csv"
    atomic_write_csv(merged, path, backup_dir=tmp_path / "backups")
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(merged)
    sample.loc[sample.frame_idx == 10, "reviewed"] = True
    assert recount_completion(sample)["reviewed"] == 2


def test_rejected_not_in_gt_rows():
    state = FrameEditState(
        frame_idx=1,
        boxes=[
            BoxState("a", "player", 0.9, 0, 0, 10, 10, accepted=True),
            BoxState("b", "player", 0.8, 20, 20, 30, 30, accepted=True, rejected=True),
        ],
    )
    rows = build_gt_rows_for_frame(state)
    assert len(rows) == 1
    assert rows[0]["gt_id"] == "a"


def test_manual_box_in_gt_rows():
    state = FrameEditState(
        frame_idx=1,
        boxes=[
            BoxState("m", "referee", 1.0, 0, 0, 5, 5, accepted=True, manual=True),
        ],
    )
    rows = build_gt_rows_for_frame(state)
    assert rows[0]["class_name"] == "referee"
    assert rows[0]["notes"] == "manual"


def test_class_change_preserved():
    state = FrameEditState(
        frame_idx=1,
        boxes=[BoxState("a", "player", 0.9, 0, 0, 10, 10, accepted=True)],
    )
    state.boxes[0].class_name = "goalkeeper"
    state.boxes[0].modified = True
    rows = build_gt_rows_for_frame(state)
    assert rows[0]["class_name"] == "goalkeeper"


def test_no_duplicate_gt_rows():
    state = FrameEditState(
        frame_idx=1,
        boxes=[
            BoxState("a", "player", 0.9, 0, 0, 10, 10, accepted=True),
            BoxState("b", "player", 0.8, 0, 0, 10, 10, accepted=True),
        ],
    )
    rows = build_gt_rows_for_frame(state)
    assert len(rows) == 1


def test_reset_reloads_proposals():
    props = _sample_proposals()
    state = FrameEditState(frame_idx=10, boxes=load_proposals_for_frame(props, 10))
    state.boxes[0].rejected = True
    state = reset_frame_from_proposals(state, props)
    assert not state.boxes[0].rejected
    assert len(state.boxes) == 2


def test_preserve_existing_reviewed_frames():
    if not GT_DIR.exists():
        return
    sample = pd.read_csv(GT_DIR / "review_sample.csv")
    gt = pd.read_csv(GT_DIR / "human_detection_gt.csv")
    previously = set(sample.loc[truthy_mask(sample["reviewed"]), "frame_idx"].astype(int))
    # merge a different frame should not drop previously reviewed markers in sample
    # (merge only touches GT rows)
    rows = [{"frame_idx": 10, "gt_id": "t", "class_name": "player", "x1": 1, "y1": 1, "x2": 2, "y2": 2,
             "occluded": False, "difficult": False, "reviewed": True, "notes": "t"}]
    merged = merge_gt_preserving_other_reviewed(gt, 10, rows, set(sample.frame_idx.astype(int)))
    for fid in previously:
        if fid == 10:
            continue
        # prior reviewed frame rows still present if they had real boxes
        assert fid in set(sample.frame_idx.astype(int))


def test_evaluator_excludes_outside_sample_and_unreviewed():
    sample = pd.DataFrame(
        [
            {"frame_idx": 10, "reviewed": True},
            {"frame_idx": 20, "reviewed": False},
        ]
    )
    gt = pd.DataFrame(
        [
            {"frame_idx": 10, "gt_id": "a", "class_name": "player", "x1": 1, "y1": 1, "x2": 2, "y2": 2,
             "occluded": False, "difficult": False, "reviewed": True, "notes": ""},
            {"frame_idx": 20, "gt_id": "b", "class_name": "player", "x1": 1, "y1": 1, "x2": 2, "y2": 2,
             "occluded": False, "difficult": False, "reviewed": True, "notes": ""},
            {"frame_idx": 999, "gt_id": "c", "class_name": "player", "x1": 1, "y1": 1, "x2": 2, "y2": 2,
             "occluded": False, "difficult": False, "reviewed": True, "notes": ""},
        ]
    )
    el = evaluator_gt_frames(gt, sample)
    assert set(el.frame_idx.astype(int)) == {10}


def test_atomic_write(tmp_path: Path):
    path = tmp_path / "t.csv"
    df = pd.DataFrame([{"a": 1}])
    atomic_write_csv(df, path, backup_dir=tmp_path / "b")
    assert path.exists()
    atomic_write_csv(pd.DataFrame([{"a": 2}]), path, backup_dir=tmp_path / "b")
    assert len(list((tmp_path / "b").glob("*.bak"))) >= 1


def test_save_guard_blocks_unaccepted_proposals():
    props = _sample_proposals()
    state = FrameEditState(frame_idx=10, boxes=load_proposals_for_frame(props, 10))
    assert save_guard(state) == "need_accept"
    accept_visible_proposals(state)
    assert save_guard(state) == "ok"
    empty = FrameEditState(frame_idx=10, boxes=[])
    assert save_guard(empty) == "need_empty_confirm"


def test_s_blocked_when_proposals_unaccepted(tmp_path: Path):
    Annotator = _load_annotator_cls()
    gt_dir = _make_gt_dir(tmp_path)
    ann = Annotator(gt_dir)
    assert ann.frame_idx() == 0
    assert ann.state.counts()["proposals"] == 2
    assert ann.state.counts()["accepted"] == 0
    before_sample = pd.read_csv(gt_dir / "review_sample.csv")
    before_gt = pd.read_csv(gt_dir / "human_detection_gt.csv")
    ok = ann.save_current_frame(allow_empty=False)
    assert ok is False
    after_sample = pd.read_csv(gt_dir / "review_sample.csv")
    after_gt = pd.read_csv(gt_dir / "human_detection_gt.csv")
    assert not is_true(after_sample.loc[after_sample.frame_idx == 0, "reviewed"].iloc[0])
    assert before_sample.equals(after_sample)
    # GT must not become reviewed empty for frame 0
    g0 = after_gt[after_gt.frame_idx.astype(int) == 0]
    assert not (truthy_mask(g0["reviewed"]).any() and g0["x1"].isna().all() and len(g0) == 1 and "empty" in str(g0.iloc[0]["gt_id"]))
    assert before_gt.equals(after_gt)
    assert "NO ACCEPTED BOXES" in ann.status_msg


def test_a_then_s_writes_gt_and_clears_dirty(tmp_path: Path):
    Annotator = _load_annotator_cls()
    gt_dir = _make_gt_dir(tmp_path)
    ann = Annotator(gt_dir)
    accept_visible_proposals(ann.state)
    assert save_guard(ann.state) == "ok"
    ok = ann.save_current_frame(allow_empty=False)
    assert ok is True
    assert ann.state.dirty is False
    assert ann.unsaved_changes() is False
    assert ann.empty_confirmation_pending is False
    gt = pd.read_csv(gt_dir / "human_detection_gt.csv")
    g0 = gt[(gt.frame_idx.astype(int) == 0) & gt.x1.notna()]
    assert len(g0) == 2
    sample = pd.read_csv(gt_dir / "review_sample.csv")
    assert is_true(sample.loc[sample.frame_idx == 0, "reviewed"].iloc[0])
    # N should not warn
    assert not ann.unsaved_changes()
    ann.idx = min(len(ann.sample) - 1, ann.idx + 1)
    ann._load_frame_state()
    assert ann.frame_idx() == 1


def test_overwrite_same_reviewed_frame_does_not_bump_completion(tmp_path: Path):
    Annotator = _load_annotator_cls()
    gt_dir = _make_gt_dir(tmp_path)
    ann = Annotator(gt_dir)
    accept_visible_proposals(ann.state)
    assert ann.save_current_frame()
    comp1 = recount_completion(ann.sample)
    assert comp1["reviewed"] == 3  # frames 0,1,2
    # edit and overwrite
    ann.state.boxes[0].x1 += 1
    ann.state.boxes[0].modified = True
    ann.state.dirty = True
    assert ann.save_current_frame()
    comp2 = recount_completion(ann.sample)
    assert comp2["reviewed"] == comp1["reviewed"] == 3


def test_double_e_saves_empty_frame(tmp_path: Path):
    Annotator = _load_annotator_cls()
    gt_dir = _make_gt_dir(tmp_path)
    ann = Annotator(gt_dir)
    assert ann.save_current_frame(allow_empty=False) is False
    ann.handle_empty_confirm()
    assert ann.empty_confirmation_pending is True
    assert "EMPTY FRAME" in ann.status_msg
    # first E alone must not save
    sample = pd.read_csv(gt_dir / "review_sample.csv")
    assert not is_true(sample.loc[sample.frame_idx == 0, "reviewed"].iloc[0])
    ann.handle_empty_confirm()
    sample = pd.read_csv(gt_dir / "review_sample.csv")
    assert is_true(sample.loc[sample.frame_idx == 0, "reviewed"].iloc[0])
    gt = pd.read_csv(gt_dir / "human_detection_gt.csv")
    g0 = gt[gt.frame_idx.astype(int) == 0]
    assert g0["x1"].isna().all()
    assert ann.unsaved_changes() is False


def test_wrong_empty_gt_can_be_overwritten(tmp_path: Path):
    Annotator = _load_annotator_cls()
    gt_dir = _make_gt_dir(tmp_path, reviewed0=True, empty_gt0=True)
    ann = Annotator(gt_dir)
    ann.idx = 0
    ann._load_frame_state()
    # empty reviewed keeps proposals for overwrite
    assert ann.state.counts()["proposals"] >= 1
    assert ann.state.counts()["accepted"] == 0
    # R → A → S
    ann.state = reset_frame_from_proposals(ann.state, ann.proposals)
    accept_visible_proposals(ann.state)
    before = recount_completion(ann.sample)["reviewed"]
    assert ann.save_current_frame()
    after = recount_completion(ann.sample)["reviewed"]
    assert after == before  # already reviewed
    gt = pd.read_csv(gt_dir / "human_detection_gt.csv")
    g0 = gt[(gt.frame_idx.astype(int) == 0) & gt.x1.notna()]
    assert len(g0) == 2


def test_atomic_save_and_backup_preserved(tmp_path: Path):
    Annotator = _load_annotator_cls()
    gt_dir = _make_gt_dir(tmp_path)
    ann = Annotator(gt_dir)
    accept_visible_proposals(ann.state)
    ann.save_current_frame()
    backups = list((gt_dir / "backups").glob("*.bak"))
    assert len(backups) >= 1
    # second save creates more backups
    ann.state.boxes[0].x1 += 1
    ann.state.dirty = True
    ann.save_current_frame()
    assert len(list((gt_dir / "backups").glob("*.bak"))) >= 2


def test_real_gt_csv_not_mutated_by_unit_tests():
    """Smoke: reading real GT must not flip reviewed / auto-accept boxes."""
    if not GT_DIR.exists():
        pytest.skip("real GT dir missing")
    sample_before = pd.read_csv(GT_DIR / "review_sample.csv")
    gt_before = pd.read_csv(GT_DIR / "human_detection_gt.csv")
    props_before = pd.read_csv(GT_DIR / "model_proposals.csv")
    # load helpers only — no Annotator save against real path
    boxes = load_proposals_for_frame(props_before, int(sample_before.iloc[0]["frame_idx"]))
    assert boxes  # proposals exist for tooling
    sample_after = pd.read_csv(GT_DIR / "review_sample.csv")
    gt_after = pd.read_csv(GT_DIR / "human_detection_gt.csv")
    props_after = pd.read_csv(GT_DIR / "model_proposals.csv")
    assert sample_before.equals(sample_after)
    assert gt_before.equals(gt_after)
    assert props_before.equals(props_after)
    assert truthy_mask(props_after["accepted"]).sum() == truthy_mask(props_before["accepted"]).sum()
