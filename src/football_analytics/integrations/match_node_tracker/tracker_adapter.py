"""Benchmark-only IoU tracker adapter wrapping upstream IdTracker.

Upstream source: third_party/authorized/match-node-tracker/id_tracker.py
Upstream commit: 2777aa3f1e9cc563eba07a675cebdf4bfd9306bf
Changes vs upstream:
  - loaded via importlib from immutable third_party path
  - feature-flagged; NEVER used as production default
  - no team/11-player hard-cap
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UPSTREAM_COMMIT = "2777aa3f1e9cc563eba07a675cebdf4bfd9306bf"
UPSTREAM_ID_TRACKER = Path(
    "/home/ahmet/projects/football-analytics/third_party/authorized/match-node-tracker/"
    "id_tracker.py"
)


@dataclass
class MatchNodeTrackerConfig:
    enabled: bool = False
    iou_thresh: float = 0.3
    max_age: int = 20


class MatchNodeTrackerAdapter:
    """Thin wrapper around upstream IdTracker for offline experiments."""

    def __init__(self, config: MatchNodeTrackerConfig | None = None) -> None:
        self.config = config or MatchNodeTrackerConfig()
        self._tracker = None
        if self.config.enabled:
            self._ensure()

    def _ensure(self) -> None:
        if self._tracker is not None:
            return
        spec = importlib.util.spec_from_file_location(
            "match_node_id_tracker", UPSTREAM_ID_TRACKER
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {UPSTREAM_ID_TRACKER}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._tracker = mod.IdTracker(
            iou_thresh=self.config.iou_thresh, max_age=self.config.max_age
        )

    def update(self, boxes: np.ndarray, mask: list[bool]) -> list[int]:
        if not self.config.enabled:
            return [-1] * len(boxes)
        self._ensure()
        assert self._tracker is not None
        return list(self._tracker.update(boxes, mask))
