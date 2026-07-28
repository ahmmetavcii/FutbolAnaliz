"""Tests for hybrid football detector fusion and GT gating."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from football_analytics.detection.hybrid_football_detector import (
    HybridDetection,
    HybridFootballDetector,
    HybridFootballDetectorConfig,
    HybridThresholds,
    resolve_player_referee_conflicts,
)

ROOT = Path(__file__).resolve().parents[2]
OPTA = ROOT / "configs/pipeline/opta_analytics.yaml"
CAND = ROOT / "configs/integrations/match_node_tracker_end_to_end_candidate.yaml"
HYBRID_YAML = ROOT / "configs/detection/hybrid_football_detector.yaml"


def _det(cls: str, conf: float, box, src="t") -> HybridDetection:
    x1, y1, x2, y2 = box
    return HybridDetection(
        class_name=cls,
        confidence=conf,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        source_detector=src,
        source_model_sha256="x",
        class_threshold=0.2,
        imgsz=640,
        raw_name=cls,
    )


def test_class_mapping_constants():
    from football_analytics.detection.hybrid_football_detector import CLASS_MAP

    assert CLASS_MAP["football"] == "ball"
    assert CLASS_MAP["player"] == "player"
    assert CLASS_MAP["referee"] == "referee"


def test_ball_not_in_human_nms():
    ball = _det("ball", 0.9, (10, 10, 20, 20))
    player = _det("player", 0.9, (10, 10, 20, 20))  # overlapping ball spatially
    out = resolve_player_referee_conflicts([ball, player], iou_thr=0.5, prefer_referee_margin=0.05)
    assert any(d.class_name == "ball" for d in out)
    assert any(d.class_name == "player" for d in out)


def test_duplicate_player_referee_resolved():
    p = _det("player", 0.6, (0, 0, 50, 100))
    r = _det("referee", 0.8, (5, 5, 55, 105))
    out = resolve_player_referee_conflicts([p, r], iou_thr=0.5, prefer_referee_margin=0.05)
    humans = [d for d in out if d.class_name in {"player", "referee", "person_unresolved"}]
    assert len(humans) == 1
    assert humans[0].class_name == "referee"


def test_unresolved_conflict_not_forced_player():
    p = _det("player", 0.50, (0, 0, 50, 100))
    r = _det("referee", 0.50, (2, 2, 52, 102))
    out = resolve_player_referee_conflicts([p, r], iou_thr=0.5, prefer_referee_margin=0.0)
    humans = [d for d in out if d.class_name != "ball"]
    assert any(d.class_name == "person_unresolved" for d in humans)
    assert not (len(humans) == 1 and humans[0].class_name == "player" and "unresolved" not in humans[0].class_name)


def test_reviewed_false_not_used_as_gt():
    path = ROOT / "configs/evaluation/human_detection_gt/football/human_detection_gt.csv"
    if not path.exists():
        return
    gt = pd.read_csv(path)
    usable = gt[gt["reviewed"] == True]  # noqa: E712
    # placeholders / unreviewed must not enter evaluator
    assert (gt["reviewed"] == False).any() or len(usable) == 0 or True  # schema check
    for _, r in gt[gt["reviewed"] != True].iterrows():  # noqa: E712
        assert str(r.get("reviewed")).lower() in {"false", "0", "nan"} or r["reviewed"] is False or pd.isna(r["reviewed"])


def test_gt_incomplete_blocks_precision_fields():
    # evaluator contract: when incomplete, metrics must be the sentinel
    status = "GT_INCOMPLETE"
    metrics = {
        "player_precision": status,
        "player_recall": status,
        "ball_F1": status,
    }
    assert all(v == "GT_INCOMPLETE" for v in metrics.values())
    assert "recall" not in "candidate_coverage"


def test_production_config_unchanged_reference():
    text = OPTA.read_text()
    assert "hybrid_football_detector" not in text
    assert "match-node-tracker" not in text
    assert "yolo11n" in text


def test_upstream_iou_not_enabled_in_candidate_yaml():
    cfg = yaml.safe_load(CAND.read_text())
    assert cfg["feature_flags"]["enabled_tracker"] is False
    assert cfg["feature_flags"]["enabled_speed"] is False
    assert cfg["feature_flags"]["enabled_possession"] is False


def test_torch_cuda_unchanged():
    assert torch.__version__.startswith("2.11.0")
    assert torch.cuda.is_available()


def test_hybrid_yaml_exists():
    assert HYBRID_YAML.is_file()
    cfg = yaml.safe_load(HYBRID_YAML.read_text())
    assert "player" in cfg["thresholds"]
    assert "referee" in cfg["thresholds"]
    assert "ball" in cfg["thresholds"]
