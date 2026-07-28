"""Guard: isolated trackers share identical common detection input."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ART = Path("/home/ahmet/workspace/hybrid_detector_validation/artifacts")
RUNS = Path("/home/ahmet/workspace/hybrid_detector_validation/runs")


def _sha_df(df: pd.DataFrame) -> str:
    payload = df.sort_values(list(df.columns)).to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def test_common_detection_sha_recorded():
    meta = json.loads((ART / "common_human_detections.sha256").read_text())
    assert "file_sha256" in meta and "content_sha256" in meta
    df = pd.read_parquet(ART / "common_human_detections.parquet")
    assert _sha_df(df) == meta["content_sha256"]


def test_trackers_use_same_detection_order_and_sha():
    meta = json.loads((ART / "common_human_detections.sha256").read_text())
    for name in ("bytetrack", "botsort"):
        d = pd.read_parquet(RUNS / f"track_{name}" / "detections.parquet")
        assert _sha_df(d) == meta["content_sha256"]
        man = json.loads((RUNS / f"track_{name}" / "run_manifest.json").read_text())
        assert man["common_detections_sha256"] == meta["file_sha256"]
        assert man.get("upstream_iou_used") is False
    a = pd.read_parquet(RUNS / "track_bytetrack" / "detections.parquet")
    b = pd.read_parquet(RUNS / "track_botsort" / "detections.parquet")
    assert list(a["detection_id"]) == list(b["detection_id"])
    assert list(a["frame_idx"]) == list(b["frame_idx"])
