"""Football-specific ball detector (SoccerNet YOLO) with tiled/ROI inference.

COCO sports-ball remains fallback only when the football model is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


DEFAULT_FOOTBALL_BALL_MODEL = Path("/home/ahmet/models/football-ball/yolo-sn-ball-opt.pt")
COCO_FALLBACK_MODEL = Path("/home/ahmet/models/yolo11n.pt")


@dataclass(frozen=True)
class FootballBallDetectorConfig:
    model_path: str = str(DEFAULT_FOOTBALL_BALL_MODEL)
    fallback_coco_path: str = str(COCO_FALLBACK_MODEL)
    device: str | int = 0
    conf: float = 0.15
    imgsz: int = 1280
    tile_grid: int = 2
    tile_overlap: float = 0.15
    enable_tiles: bool = True
    enable_roi: bool = True
    roi_half_size: int = 256
    roi_imgsz: int = 640
    roi_conf: float = 0.10
    max_ball_size_px: float = 80.0
    min_ball_size_px: float = 2.0
    coco_ball_class_id: int = 32


@dataclass
class BallDet:
    frame_id: int
    x: float
    y: float
    confidence: float
    width: float
    height: float
    source: str  # football_yolo | tile | roi | coco_fallback


class FootballBallDetector:
    def __init__(self, config: FootballBallDetectorConfig | None = None) -> None:
        self.config = config or FootballBallDetectorConfig()
        self.model = None
        self.backend = "none"
        self._load()

    def _load(self) -> None:
        from ultralytics import YOLO

        path = Path(self.config.model_path)
        if path.is_file():
            self.model = YOLO(str(path))
            self.backend = "football_yolo"
            self.ball_class_ids = {0}  # single-class ball models
            return
        fallback = Path(self.config.fallback_coco_path)
        if fallback.is_file():
            self.model = YOLO(str(fallback))
            self.backend = "coco_fallback"
            self.ball_class_ids = {int(self.config.coco_ball_class_id)}
            return
        raise FileNotFoundError(
            f"No ball model at {path} and no COCO fallback at {fallback}"
        )

    def _boxes_to_dets(
        self,
        result: Any,
        frame_id: int,
        *,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        source: str,
        class_filter: set[int] | None = None,
    ) -> list[BallDet]:
        if result.boxes is None or len(result.boxes) == 0:
            return []
        class_filter = class_filter if class_filter is not None else self.ball_class_ids
        out: list[BallDet] = []
        for box in result.boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else 0
            if cls_id not in class_filter:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            w = x2 - x1
            h = y2 - y1
            size = (w * h) ** 0.5
            if size < self.config.min_ball_size_px or size > self.config.max_ball_size_px:
                continue
            out.append(
                BallDet(
                    frame_id=frame_id,
                    x=offset_x + (x1 + x2) / 2.0,
                    y=offset_y + (y1 + y2) / 2.0,
                    confidence=conf,
                    width=w,
                    height=h,
                    source=source,
                )
            )
        return out

    def detect_frame(
        self,
        bgr: np.ndarray,
        frame_id: int,
        *,
        prior_xy: tuple[float, float] | None = None,
    ) -> list[BallDet]:
        cfg = self.config
        h, w = bgr.shape[:2]
        dets: list[BallDet] = []

        # Full-frame high-res
        results = self.model.predict(
            source=bgr,
            conf=cfg.conf,
            imgsz=cfg.imgsz,
            classes=sorted(self.ball_class_ids) if self.backend == "coco_fallback" else None,
            verbose=False,
        )
        dets.extend(
            self._boxes_to_dets(
                results[0],
                frame_id,
                source="football_yolo" if self.backend == "football_yolo" else "coco_fallback",
            )
        )

        # Tiled inference for small balls
        if cfg.enable_tiles and cfg.tile_grid >= 2 and not dets:
            gh = gw = cfg.tile_grid
            overlap_x = int(w * cfg.tile_overlap / gw)
            overlap_y = int(h * cfg.tile_overlap / gh)
            tile_w = int(np.ceil(w / gw)) + overlap_x
            tile_h = int(np.ceil(h / gh)) + overlap_y
            for iy in range(gh):
                for ix in range(gw):
                    x0 = max(0, ix * (w // gw) - overlap_x // 2)
                    y0 = max(0, iy * (h // gh) - overlap_y // 2)
                    x1 = min(w, x0 + tile_w)
                    y1 = min(h, y0 + tile_h)
                    crop = bgr[y0:y1, x0:x1]
                    if crop.size == 0:
                        continue
                    tr = self.model.predict(
                        source=crop,
                        conf=cfg.conf,
                        imgsz=max(640, cfg.imgsz // cfg.tile_grid),
                        classes=sorted(self.ball_class_ids)
                        if self.backend == "coco_fallback"
                        else None,
                        verbose=False,
                    )
                    dets.extend(
                        self._boxes_to_dets(
                            tr[0],
                            frame_id,
                            offset_x=float(x0),
                            offset_y=float(y0),
                            source="tile",
                        )
                    )

        # ROI around prior
        if cfg.enable_roi and prior_xy is not None and not dets:
            px, py = prior_xy
            x0 = int(max(0, px - cfg.roi_half_size))
            y0 = int(max(0, py - cfg.roi_half_size))
            x1 = int(min(w, px + cfg.roi_half_size))
            y1 = int(min(h, py + cfg.roi_half_size))
            crop = bgr[y0:y1, x0:x1]
            if crop.size > 0:
                tr = self.model.predict(
                    source=crop,
                    conf=cfg.roi_conf,
                    imgsz=cfg.roi_imgsz,
                    classes=sorted(self.ball_class_ids)
                    if self.backend == "coco_fallback"
                    else None,
                    verbose=False,
                )
                dets.extend(
                    self._boxes_to_dets(
                        tr[0],
                        frame_id,
                        offset_x=float(x0),
                        offset_y=float(y0),
                        source="roi",
                    )
                )

        # Keep best confidence per frame (NMS-ish by distance)
        if not dets:
            return []
        dets.sort(key=lambda d: d.confidence, reverse=True)
        kept: list[BallDet] = []
        for det in dets:
            if any(
                ((det.x - k.x) ** 2 + (det.y - k.y) ** 2) ** 0.5 < 20.0 for k in kept
            ):
                continue
            kept.append(det)
            if len(kept) >= 3:
                break
        return kept

    def detect_video(
        self,
        video_path: Path,
        *,
        frame_count: int | None = None,
        progress_every: int = 100,
    ) -> pd.DataFrame:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        rows: list[dict[str, Any]] = []
        prior: tuple[float, float] | None = None
        fid = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_count is not None and fid >= frame_count:
                break
            dets = self.detect_frame(bgr, fid, prior_xy=prior)
            if dets:
                best = max(dets, key=lambda d: d.confidence)
                prior = (best.x, best.y)
                for det in dets:
                    rows.append(
                        {
                            "frame_id": det.frame_id,
                            "ball_x_pixel": det.x,
                            "ball_y_pixel": det.y,
                            "detection_confidence": det.confidence,
                            "bbox_w": det.width,
                            "bbox_h": det.height,
                            "detector_source": det.source,
                            "detector_backend": self.backend,
                        }
                    )
            else:
                # keep prior for ROI continuity but do not invent a detection
                pass
            fid += 1
            if progress_every and fid % progress_every == 0:
                print(f"[football_ball] frame {fid}", flush=True)
        cap.release()
        return pd.DataFrame(rows)
