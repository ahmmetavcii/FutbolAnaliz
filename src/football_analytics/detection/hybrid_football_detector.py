"""Hybrid football detector: upstream human YOLO + production football-ball.

Upstream human weights (Match-node-tracker best.pt) provide player/referee.
Ball comes from the existing FootballBallDetector. Goalkeeper is never invented
as a detector class; optional role-classifier evidence may label GK after the
fact with low confidence when evidence is weak.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from football_analytics.integrations.football_ball_detector import (
    FootballBallDetector,
    FootballBallDetectorConfig,
)

DEFAULT_HUMAN_WEIGHTS = Path(
    "third_party/authorized/match-node-tracker/train/weights/best.pt"
)
DEFAULT_HUMAN_SHA256 = (
    "2caf0b0cac1a09600c7144edf91568c1848003d8e55bd9e3b3906622d7d2205a"
)
UPSTREAM_COMMIT = "2777aa3f1e9cc563eba07a675cebdf4bfd9306bf"

CLASS_MAP = {
    "football": "ball",
    "player": "player",
    "referee": "referee",
    "goalkeeper": "goalkeeper",
}


@dataclass(frozen=True)
class HybridThresholds:
    player: float = 0.25
    referee: float = 0.35
    ball: float = 0.10


@dataclass
class HybridFootballDetectorConfig:
    human_weights: str = str(DEFAULT_HUMAN_WEIGHTS)
    human_sha256: str = DEFAULT_HUMAN_SHA256
    ball_model_path: str = str(FootballBallDetectorConfig.model_path)
    imgsz: int = 1280
    device: int | str = 0
    thresholds: HybridThresholds = field(default_factory=HybridThresholds)
    person_iou_conflict: float = 0.55
    prefer_referee_margin: float = 0.05
    enable_role_classifier: bool = True
    goalkeeper_min_confidence: float = 0.55
    project_root: str = "/home/ahmet/projects/football-analytics"


@dataclass
class HybridDetection:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    source_detector: str
    source_model_sha256: str
    class_threshold: float
    imgsz: int
    raw_name: str = ""
    goalkeeper_confidence: float | None = None
    role: str | None = None

    @property
    def foot_xy(self) -> tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), self.y2)

    def as_row(
        self,
        *,
        video_name: str,
        frame_idx: int,
        timestamp_seconds: float,
        detection_id: str,
    ) -> dict[str, Any]:
        fx, fy = self.foot_xy
        return {
            "video_name": video_name,
            "frame_idx": int(frame_idx),
            "timestamp_seconds": float(timestamp_seconds),
            "detection_id": detection_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "x1": float(self.x1),
            "y1": float(self.y1),
            "x2": float(self.x2),
            "y2": float(self.y2),
            "foot_x": float(fx),
            "foot_y": float(fy),
            "source_detector": self.source_detector,
            "source_model_sha256": self.source_model_sha256,
            "class_threshold": float(self.class_threshold),
            "imgsz": int(self.imgsz),
            "raw_name": self.raw_name,
            "role": self.role,
            "goalkeeper_confidence": self.goalkeeper_confidence,
        }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def resolve_player_referee_conflicts(
    dets: list[HybridDetection],
    *,
    iou_thr: float,
    prefer_referee_margin: float,
) -> list[HybridDetection]:
    """Class-aware NMS between player and referee only (ball excluded)."""
    humans = [d for d in dets if d.class_name in {"player", "referee", "person_unresolved"}]
    balls = [d for d in dets if d.class_name == "ball"]
    others = [d for d in dets if d.class_name not in {"player", "referee", "person_unresolved", "ball"}]
    humans = sorted(humans, key=lambda d: d.confidence, reverse=True)
    kept: list[HybridDetection] = []
    for det in humans:
        conflict = None
        for k in kept:
            if _iou((det.x1, det.y1, det.x2, det.y2), (k.x1, k.y1, k.x2, k.y2)) < iou_thr:
                continue
            conflict = k
            break
        if conflict is None:
            kept.append(det)
            continue
        # conflict resolution
        same = det.class_name == conflict.class_name
        if same:
            continue  # lower conf duplicate dropped
        # different classes: prefer higher conf, with small referee margin
        det_score = det.confidence + (
            prefer_referee_margin if det.class_name == "referee" else 0.0
        )
        conf_score = conflict.confidence + (
            prefer_referee_margin if conflict.class_name == "referee" else 0.0
        )
        if abs(det_score - conf_score) < 0.03:
            # low-confidence conflict → unresolved (do NOT force player)
            conflict.class_name = "person_unresolved"
            conflict.raw_name = f"conflict:{conflict.raw_name}+{det.raw_name}"
            continue
        if det_score > conf_score:
            kept.remove(conflict)
            kept.append(det)
    return kept + balls + others


class HybridFootballDetector:
    """Combine Match-node human detections with production ball detections."""

    def __init__(self, config: HybridFootballDetectorConfig | None = None) -> None:
        self.config = config or HybridFootballDetectorConfig()
        self._human = None
        self._ball: FootballBallDetector | None = None
        self.human_sha256 = ""
        self.ball_sha256 = ""
        self.names: dict[int, str] = {}
        self.has_goalkeeper_class = False

    def _resolve(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_file():
            return p
        alt = Path(self.config.project_root) / p
        if alt.is_file():
            return alt
        return p

    def load(self) -> None:
        from ultralytics import YOLO

        human_path = self._resolve(self.config.human_weights)
        if not human_path.is_file():
            raise FileNotFoundError(human_path)
        self.human_sha256 = _sha256_file(human_path)
        if self.config.human_sha256 and self.human_sha256 != self.config.human_sha256:
            raise ValueError(
                f"human weights SHA256 mismatch: {self.human_sha256} != {self.config.human_sha256}"
            )
        self._human = YOLO(str(human_path))
        self.names = {int(k): str(v) for k, v in dict(self._human.names).items()}
        self.has_goalkeeper_class = any("goalkeeper" in v.lower() for v in self.names.values())

        ball_cfg = FootballBallDetectorConfig(
            model_path=str(self._resolve(self.config.ball_model_path)),
            device=self.config.device,
            conf=0.01,  # filter by per-class threshold after inference
            imgsz=int(self.config.imgsz),
            enable_tiles=True,
            enable_roi=True,
        )
        self._ball = FootballBallDetector(ball_cfg)
        ball_path = self._resolve(self.config.ball_model_path)
        self.ball_sha256 = _sha256_file(ball_path) if ball_path.is_file() else ""

    def detect_frame(self, frame_bgr: np.ndarray, frame_idx: int = 0) -> list[HybridDetection]:
        if self._human is None or self._ball is None:
            self.load()
        assert self._human is not None and self._ball is not None

        thr = self.config.thresholds
        # Human path: run at min(player, referee) then filter per-class
        min_human = min(thr.player, thr.referee)
        res = self._human.predict(
            source=frame_bgr,
            imgsz=self.config.imgsz,
            conf=min_human,
            device=self.config.device,
            verbose=False,
        )[0]
        out: list[HybridDetection] = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                raw = self.names.get(int(cls[i]), str(int(cls[i])))
                mapped = CLASS_MAP.get(raw.lower(), raw.lower())
                if mapped == "ball":
                    # ignore upstream ball — ball comes from production detector only
                    continue
                if mapped == "goalkeeper":
                    # never invent / accept GK class from upstream (absent anyway)
                    continue
                if mapped == "player" and float(confs[i]) < thr.player:
                    continue
                if mapped == "referee" and float(confs[i]) < thr.referee:
                    continue
                if mapped not in {"player", "referee"}:
                    continue
                x1, y1, x2, y2 = map(float, xyxy[i])
                out.append(
                    HybridDetection(
                        class_name=mapped,
                        confidence=float(confs[i]),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        source_detector="match_node_best",
                        source_model_sha256=self.human_sha256,
                        class_threshold=thr.player if mapped == "player" else thr.referee,
                        imgsz=int(self.config.imgsz),
                        raw_name=raw,
                        role="player" if mapped == "player" else "referee",
                    )
                )

        # Ball path — never enters human NMS; filter by ball threshold
        balls = self._ball.detect_frame(frame_bgr, frame_idx)
        for b in balls:
            if float(b.confidence) < thr.ball:
                continue
            out.append(
                HybridDetection(
                    class_name="ball",
                    confidence=float(b.confidence),
                    x1=float(b.x - b.width / 2),
                    y1=float(b.y - b.height / 2),
                    x2=float(b.x + b.width / 2),
                    y2=float(b.y + b.height / 2),
                    source_detector=f"football_ball:{b.source}",
                    source_model_sha256=self.ball_sha256,
                    class_threshold=float(thr.ball),
                    imgsz=int(self.config.imgsz),
                    raw_name="ball",
                    role=None,
                )
            )

        fused = resolve_player_referee_conflicts(
            out,
            iou_thr=self.config.person_iou_conflict,
            prefer_referee_margin=self.config.prefer_referee_margin,
        )
        if self.config.enable_role_classifier:
            fused = self._maybe_label_goalkeeper(fused, frame_bgr)
        return fused

    def _maybe_label_goalkeeper(
        self, dets: list[HybridDetection], frame_bgr: np.ndarray
    ) -> list[HybridDetection]:
        """Optional GK role on player boxes; never invents a detector class."""
        try:
            from football_analytics.roles.goalkeeper_classifier import GoalkeeperClassifier
            from football_analytics.roles.goalkeeper_classifier import GoalkeeperFeatures
        except Exception:
            return dets
        # Lightweight: deepest/largest player as weak prior only when classifier API allows.
        # Without rich features, leave as player with low GK confidence.
        players = [d for d in dets if d.class_name == "player"]
        if not players:
            return dets
        # Mark all players with low GK confidence by default (no forced flip).
        for d in players:
            d.role = "player"
            d.goalkeeper_confidence = 0.0
        # If a single very deep (high foot_y) and large box exists, raise score slightly
        # but only promote when above configured threshold.
        ranked = sorted(players, key=lambda d: (d.y2, (d.x2 - d.x1) * (d.y2 - d.y1)), reverse=True)
        candidate = ranked[0]
        # heuristic weak prior — still requires threshold
        weak = 0.35
        candidate.goalkeeper_confidence = weak
        if weak >= self.config.goalkeeper_min_confidence:
            candidate.role = "goalkeeper"
            # class_name stays player (detector class); role carries GK inference
        else:
            candidate.role = "player"
            candidate.goalkeeper_confidence = weak  # low
        return dets
