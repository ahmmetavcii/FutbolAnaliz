"""Guards for Match-node-tracker provisional integration."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/authorized/match-node-tracker"
CANDIDATE = ROOT / "configs/integrations/match_node_tracker_candidate.yaml"
OPTA = ROOT / "configs/pipeline/opta_analytics.yaml"
BEST = UPSTREAM / "train/weights/best.pt"
COMMIT = "2777aa3f1e9cc563eba07a675cebdf4bfd9306bf"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_upstream_git_commit():
    import subprocess

    sha = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    assert sha == COMMIT


def test_upstream_not_lfs_pointer():
    head = BEST.read_bytes()[:64]
    assert not head.startswith(b"version https://git-lfs.github.com/spec/v1")
    assert head[:2] == b"PK"


def test_model_sha256_recorded():
    cfg = yaml.safe_load(CANDIDATE.read_text())
    assert cfg["detector"]["sha256"] == _sha256(BEST)


def test_model_names_auto_read():
    from ultralytics import YOLO

    m = YOLO(str(BEST))
    names = {int(k): str(v).lower() for k, v in dict(m.names).items()}
    assert "football" in names.values() or "ball" in names.values()
    assert "player" in names.values()
    assert "referee" in names.values()
    assert "goalkeeper" not in names.values()


def test_candidate_flags_default_false():
    cfg = yaml.safe_load(CANDIDATE.read_text())
    for k, v in cfg["feature_flags"].items():
        assert v is False, k


def test_adapters_respect_disabled_flags():
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from football_analytics.integrations.match_node_tracker.detector_adapter import (
        MatchNodeDetectorAdapter,
        MatchNodeDetectorConfig,
    )
    from football_analytics.integrations.match_node_tracker.tracker_adapter import (
        MatchNodeTrackerAdapter,
        MatchNodeTrackerConfig,
    )
    from football_analytics.integrations.match_node_tracker.marker_renderer import (
        MatchNodeMarkerRenderer,
        MatchNodeMarkerConfig,
    )
    import numpy as np

    det = MatchNodeDetectorAdapter(MatchNodeDetectorConfig(enabled=False))
    assert det.detect_frame(np.zeros((64, 64, 3), dtype=np.uint8)) == []
    tr = MatchNodeTrackerAdapter(MatchNodeTrackerConfig(enabled=False))
    assert tr.update(np.zeros((0, 4)), []) == []
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    out = MatchNodeMarkerRenderer(MatchNodeMarkerConfig(enabled=False)).draw(
        img, np.zeros((0, 4)), []
    )
    assert out is img


def test_iou_tracker_not_production_default():
    opta = OPTA.read_text()
    assert "match_node" not in opta
    assert "id_tracker" not in opta
    cand = yaml.safe_load(CANDIDATE.read_text())
    assert cand["feature_flags"]["enabled_tracker"] is False
    assert cand["status"] == "PROVISIONAL_CANDIDATE"


def test_speed_and_possession_flags_false():
    cfg = yaml.safe_load(CANDIDATE.read_text())
    assert cfg["feature_flags"]["enabled_speed"] is False
    assert cfg["feature_flags"]["enabled_possession"] is False


def test_pytorch_cuda_unchanged():
    assert torch.__version__.startswith("2.11.0")
    assert torch.cuda.is_available()


def test_no_accuracy_metric_files_without_gt():
    bench = Path("/home/ahmet/workspace/match_node_tracker_integration/benchmarks")
    for p in bench.rglob("*.json"):
        text = p.read_text().lower()
        assert "precision" not in text or "gt_incomplete" in text or "gt_status" in p.read_text()
        assert "candidate coverage" not in text
        # forbid calling candidate counts recall
        if "recall" in text:
            assert "gt_incomplete" in text or "not" in text


def test_production_opta_model_path_unchanged():
    text = OPTA.read_text()
    assert "yolo11n" in text
    assert "match-node-tracker" not in text
