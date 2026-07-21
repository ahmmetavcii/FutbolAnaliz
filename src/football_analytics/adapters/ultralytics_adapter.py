"""Ultralytics YOLO detection and tracking adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from ultralytics import YOLO

from football_analytics.contracts.schemas import SCHEMA_VERSION


COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    32: "sports ball",
}


class UltralyticsAdapter:
    def __init__(
        self,
        model_path: Path,
        device: int | str = 0,
        imgsz: int = 640,
        batch: int = 1,
        half: bool = True,
        conf: float = 0.25,
        classes: list[int] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.imgsz = imgsz
        self.batch = batch
        self.half = half
        self.conf = conf
        self.classes = classes
        self.model = YOLO(str(self.model_path))
        self.source_model = self.model_path.name

    def _common_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "device": self.device,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "half": self.half,
            "conf": self.conf,
            "verbose": False,
            "stream": True,
        }
        if self.classes is not None:
            kwargs["classes"] = self.classes
        return kwargs

    def detect(
        self,
        video_path: Path,
        project: Path,
        name: str,
        save: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        frames = 0
        results = self.model.predict(
            source=str(video_path),
            project=str(project),
            name=name,
            save=save,
            exist_ok=True,
            **self._common_kwargs(),
        )
        for frame_id, result in enumerate(results):
            frames += 1
            fps = float(result.speed.get("inference") or 0.0)
            # Prefer source FPS from video path later; timestamp from frame index.
            timestamp_ms = float(frame_id)  # replaced by caller with fps-aware values
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confs = boxes.conf.detach().cpu().tolist()
            clss = boxes.cls.detach().cpu().tolist()
            for det_idx, (box, conf, cls_id) in enumerate(zip(xyxy, confs, clss)):
                class_id = int(cls_id)
                rows.append(
                    {
                        "frame_id": int(frame_id),
                        "timestamp_ms": timestamp_ms,
                        "detection_id": f"f{frame_id}_d{det_idx}",
                        "object_type": COCO_NAMES.get(class_id, f"class_{class_id}"),
                        "class_id": class_id,
                        "bbox_x1": float(box[0]),
                        "bbox_y1": float(box[1]),
                        "bbox_x2": float(box[2]),
                        "bbox_y2": float(box[3]),
                        "detection_confidence": float(conf),
                        "source_model": self.source_model,
                        "schema_version": SCHEMA_VERSION,
                    }
                )
                _ = fps
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics = {
            "frames": frames,
            "detections": len(rows),
            "runtime_seconds": elapsed,
            "processing_fps": (frames / elapsed) if elapsed > 0 else 0.0,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "annotated_dir": str(project / name),
        }
        return pd.DataFrame(rows), metrics

    def track(
        self,
        video_path: Path,
        tracker: str,
        project: Path,
        name: str,
        save: bool = True,
        persist: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        tracker_cfg = tracker if tracker.endswith(".yaml") else f"{tracker}.yaml"
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        frames = 0
        unique_ids: set[int] = set()
        results = self.model.track(
            source=str(video_path),
            tracker=tracker_cfg,
            project=str(project),
            name=name,
            save=save,
            exist_ok=True,
            persist=persist,
            **self._common_kwargs(),
        )
        for frame_id, result in enumerate(results):
            frames += 1
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confs = boxes.conf.detach().cpu().tolist()
            clss = boxes.cls.detach().cpu().tolist()
            ids = (
                boxes.id.detach().cpu().tolist()
                if boxes.id is not None
                else [None] * len(xyxy)
            )
            for det_idx, (box, conf, cls_id, track_id) in enumerate(
                zip(xyxy, confs, clss, ids)
            ):
                if track_id is None:
                    continue
                class_id = int(cls_id)
                tid = int(track_id)
                unique_ids.add(tid)
                x1, y1, x2, y2 = map(float, box)
                rows.append(
                    {
                        "frame_id": int(frame_id),
                        "timestamp_ms": float(frame_id),
                        "track_id": tid,
                        "detection_id": f"f{frame_id}_t{tid}_d{det_idx}",
                        "object_type": COCO_NAMES.get(class_id, f"class_{class_id}"),
                        "class_id": class_id,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "foot_x_pixel": (x1 + x2) / 2.0,
                        "foot_y_pixel": y2,
                        "tracking_confidence": float(conf),
                        "source_tracker": tracker_cfg.replace(".yaml", ""),
                        "source_model": self.source_model,
                        "schema_version": SCHEMA_VERSION,
                    }
                )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics = {
            "frames": frames,
            "track_rows": len(rows),
            "unique_track_ids": len(unique_ids),
            "runtime_seconds": elapsed,
            "processing_fps": (frames / elapsed) if elapsed > 0 else 0.0,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "annotated_dir": str(project / name),
            "tracker": tracker_cfg.replace(".yaml", ""),
        }
        return pd.DataFrame(rows), metrics
