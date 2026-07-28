"""Adapter for Match-node-tracker YOLO detectors (candidate / feature-flagged).

Upstream source: third_party/authorized/match-node-tracker/
Upstream commit: 2777aa3f1e9cc563eba07a675cebdf4bfd9306bf
Changes vs upstream:
  - class-name mapping to project schema (football->ball, player, referee)
  - no goalkeeper class in upstream weights (explicitly reported)
  - returns structured dicts instead of drawing
  - does not mutate production pipeline defaults
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

UPSTREAM_COMMIT = "2777aa3f1e9cc563eba07a675cebdf4bfd9306bf"
DEFAULT_BEST = Path(
    "third_party/authorized/match-node-tracker/train/weights/best.pt"
)
DEFAULT_CUSTOM = Path(
    "third_party/authorized/match-node-tracker/custom_model.pt"
)

CLASS_MAP = {
    "football": "ball",
    "player": "player",
    "referee": "referee",
    "goalkeeper": "goalkeeper",
    "ball": "ball",
}


@dataclass(frozen=True)
class MatchNodeDetectorConfig:
    enabled: bool = False
    weights: str = str(DEFAULT_BEST)
    imgsz: int = 1280
    conf: float = 0.25
    device: str | int = 0
    half: bool = False


class MatchNodeDetectorAdapter:
    """Optional YOLO detector using authorized Match-node-tracker weights."""

    def __init__(self, config: MatchNodeDetectorConfig | None = None) -> None:
        self.config = config or MatchNodeDetectorConfig()
        self._model = None
        self.names: dict[int, str] = {}
        self.has_goalkeeper = False
        if self.config.enabled:
            self._load()

    def _load(self) -> None:
        from ultralytics import YOLO

        path = Path(self.config.weights)
        if not path.is_file():
            # resolve relative to project root heuristics
            alt = Path("/home/ahmet/projects/football-analytics") / path
            path = alt if alt.is_file() else path
        self._model = YOLO(str(path))
        self.names = {int(k): str(v) for k, v in dict(self._model.names).items()}
        self.has_goalkeeper = any("goalkeeper" in v.lower() for v in self.names.values())

    def detect_frame(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        """Run detection on one BGR frame. No-op when disabled."""
        if not self.config.enabled:
            return []
        if self._model is None:
            self._load()
        assert self._model is not None
        res = self._model.predict(
            source=frame_bgr,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            device=self.config.device,
            verbose=False,
        )[0]
        out: list[dict[str, Any]] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            raw = self.names.get(int(cls[i]), str(int(cls[i])))
            mapped = CLASS_MAP.get(raw.lower(), raw.lower())
            x1, y1, x2, y2 = map(float, xyxy[i])
            out.append(
                {
                    "object_type": mapped,
                    "raw_name": raw,
                    "class_id": int(cls[i]),
                    "bbox_xyxy": (x1, y1, x2, y2),
                    "detection_confidence": float(confs[i]),
                    "source_model": f"match_node_tracker:{Path(self.config.weights).name}",
                    "upstream_commit": UPSTREAM_COMMIT,
                }
            )
        return out
