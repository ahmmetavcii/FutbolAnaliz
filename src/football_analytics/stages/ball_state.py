"""Rewrite ball_state to prefer football-specific YOLO ball detections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from football_analytics.analytics.ball_trajectory import (
    BallObservation,
    BallState,
    BallTrackerConfig,
    BallTrajectoryEstimator,
    Bounds,
)
from football_analytics.contracts.schemas import BALL_STATE_SCHEMA, validate_mvp2_columns
from football_analytics.integrations.football_ball_detector import (
    DEFAULT_FOOTBALL_BALL_MODEL,
    FootballBallDetector,
    FootballBallDetectorConfig,
)
from football_analytics.stages.base import Stage
from football_analytics.stages.mvp2_common import (
    canonical_common,
    load_video_manifest,
    read_required_parquet,
    video_frame_count,
    video_fps,
)
from football_analytics.utils.io import write_json, write_rows_with_schema


class BallStateStage(Stage):
    name = "ball_state"

    def validate_inputs(self) -> None:
        read_required_parquet(self.run_dir / "detections.parquet")
        read_required_parquet(self.run_dir / "shot_segments.parquet")
        read_required_parquet(self.run_dir / "calibration.parquet")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        detections = read_required_parquet(self.run_dir / "detections.parquet")
        shots = read_required_parquet(self.run_dir / "shot_segments.parquet").set_index(
            "frame_id"
        )
        calibration = read_required_parquet(
            self.run_dir / "calibration.parquet"
        ).set_index("frame_id")
        cfg = self.config["ball_trajectory"]
        fb_cfg = self.config.get("football_ball") or {}
        manifest = load_video_manifest(self.run_dir)
        width = float(manifest["working_summary"]["width"])
        height = float(manifest["working_summary"]["height"])
        fps = video_fps(self.run_dir)
        frames = video_frame_count(self.run_dir)
        video_path = Path(manifest.get("working_path") or "")

        # --- Football-specific detector (primary) ---
        football_rows: list[dict[str, Any]] = []
        football_by_frame: dict[int, list[BallObservation]] = {}
        detector_backend = "none"
        use_football = bool(fb_cfg.get("enabled", True))
        model_path = Path(fb_cfg.get("model_path") or DEFAULT_FOOTBALL_BALL_MODEL)
        if use_football and video_path.is_file():
            det_cfg = FootballBallDetectorConfig(
                model_path=str(model_path),
                device=fb_cfg.get("device", self.config.get("model", {}).get("device", 0)),
                conf=float(fb_cfg.get("conf", 0.15)),
                imgsz=int(fb_cfg.get("imgsz", 1280)),
                tile_grid=int(fb_cfg.get("tile_grid", 2)),
                enable_tiles=bool(fb_cfg.get("enable_tiles", True)),
                enable_roi=bool(fb_cfg.get("enable_roi", True)),
                roi_half_size=int(fb_cfg.get("roi_half_size", 256)),
                roi_conf=float(fb_cfg.get("roi_conf", 0.10)),
            )
            try:
                detector = FootballBallDetector(det_cfg)
                detector_backend = detector.backend
                # Stream frames once
                cap = cv2.VideoCapture(str(video_path))
                prior = None
                for frame_id in range(frames):
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    dets = detector.detect_frame(bgr, frame_id, prior_xy=prior)
                    if dets:
                        best = max(dets, key=lambda d: d.confidence)
                        prior = (best.x, best.y)
                        obs_list = []
                        for det in dets:
                            football_rows.append(
                                {
                                    "frame_id": det.frame_id,
                                    "ball_x_pixel": det.x,
                                    "ball_y_pixel": det.y,
                                    "detection_confidence": det.confidence,
                                    "bbox_w": det.width,
                                    "bbox_h": det.height,
                                    "detector_source": det.source,
                                    "detector_backend": detector.backend,
                                }
                            )
                            obs_list.append(
                                BallObservation(
                                    x=det.x,
                                    y=det.y,
                                    confidence=det.confidence,
                                    width=det.width,
                                    height=det.height,
                                )
                            )
                        football_by_frame[frame_id] = obs_list
                cap.release()
            except Exception as exc:  # noqa: BLE001
                write_json(
                    self.stage_dir / "football_ball_error.json",
                    {"error": str(exc), "fallback": "coco"},
                )
                detector_backend = f"error:{exc}"

        if football_rows:
            pd.DataFrame(football_rows).to_parquet(
                self.run_dir / "football_ball_detections.parquet", index=False
            )
        else:
            pd.DataFrame(
                columns=[
                    "frame_id",
                    "ball_x_pixel",
                    "ball_y_pixel",
                    "detection_confidence",
                    "bbox_w",
                    "bbox_h",
                    "detector_source",
                    "detector_backend",
                ]
            ).to_parquet(self.run_dir / "football_ball_detections.parquet", index=False)

        # --- COCO fallback detections from main detector ---
        ball_classes = set(int(value) for value in cfg["ball_class_ids"])
        ball_detections = detections.loc[detections["class_id"].isin(ball_classes)]
        coco_grouped: dict[int, list[BallObservation]] = {}
        for key, value in ball_detections.groupby("frame_id"):
            coco_grouped[int(key)] = [
                BallObservation(
                    x=(float(item.bbox_x1) + float(item.bbox_x2)) / 2.0,
                    y=(float(item.bbox_y1) + float(item.bbox_y2)) / 2.0,
                    confidence=float(item.detection_confidence),
                    width=float(item.bbox_x2) - float(item.bbox_x1),
                    height=float(item.bbox_y2) - float(item.bbox_y1),
                )
                for item in value.itertuples()
            ]

        ball_min_conf = float(
            cfg.get("min_detection_confidence", fb_cfg.get("conf", 0.12))
        )
        filter_type = str(cfg.get("filter_type", "kalman"))
        estimator = BallTrajectoryEstimator(
            BallTrackerConfig(
                max_gap_ms=float(cfg["maximum_gap_seconds"]) * 1000.0,
                max_gap_frames=int(cfg["maximum_gap_frames"]),
                short_occlusion_frames=min(5, int(cfg["maximum_gap_frames"])),
                max_speed=float(cfg["maximum_pixel_speed_per_second"]),
                max_acceleration=float(cfg["maximum_pixel_acceleration"]),
                min_confidence=ball_min_conf,
                frame_bounds=Bounds(0.0, 0.0, width, height),
                filter_type=filter_type
                if filter_type in {"kalman", "constant_velocity"}
                else "kalman",
            )
        )
        rows: list[dict[str, Any]] = []
        known_states = {
            BallState.DETECTED,
            BallState.PREDICTED,
            BallState.OCCLUDED_SHORT,
            BallState.AIRBORNE,
        }
        raw_detected_frames = 0
        provenance_counts = {
            "detected": 0,
            "tracked": 0,
            "predicted": 0,
            "interpolated": 0,
            "missing": 0,
        }
        for frame_id in range(frames):
            # Prefer football dets; fall back to COCO
            candidates = list(football_by_frame.get(frame_id) or [])
            source_tag = "football_ball"
            if not candidates:
                candidates = list(coco_grouped.get(frame_id) or [])
                source_tag = "coco_fallback"
            detection_confidence = (
                max(item.confidence for item in candidates) if candidates else None
            )
            if candidates:
                raw_detected_frames += 1
            scene_cut = (
                bool(shots.loc[frame_id, "scene_cut"]) if frame_id in shots.index else False
            )
            estimate = estimator.step(
                frame_id,
                frame_id * 1000.0 / fps,
                candidates,
                scene_cut=scene_cut,
            )
            field_x, field_y = self._field_position(
                estimate.x, estimate.y, calibration, frame_id
            )
            valid = estimate.state in known_states and estimate.x is not None
            # Provenance (separate from visible)
            if estimate.state == BallState.DETECTED and candidates:
                provenance = "detected"
            elif estimate.state in {BallState.OCCLUDED_SHORT, BallState.AIRBORNE}:
                provenance = "tracked"
            elif estimate.state == BallState.PREDICTED:
                provenance = "predicted"
            elif valid:
                provenance = "interpolated"
            else:
                provenance = "missing"
            provenance_counts[provenance] = provenance_counts.get(provenance, 0) + 1
            rows.append(
                {
                    **canonical_common(
                        self.run_dir,
                        frame_id,
                        frame_id * 1000.0 / fps,
                        (
                            f"{source_tag}_detection"
                            if estimate.state in {BallState.DETECTED, BallState.AIRBORNE}
                            else (
                                "bounded_kalman"
                                if valid
                                else "no_reliable_ball_state"
                            )
                        ),
                        estimate.confidence,
                        valid,
                    ),
                    "ball_x_pixel": estimate.x,
                    "ball_y_pixel": estimate.y,
                    "ball_x_field": field_x,
                    "ball_y_field": field_y,
                    "visibility_state": estimate.state.value,
                    "detection_confidence": detection_confidence,
                    "trajectory_confidence": estimate.confidence,
                    "invalid_reason": None if valid else "ball_unknown_or_out_of_frame",
                }
            )
        output = self.run_dir / "ball_state.parquet"
        write_rows_with_schema(output, rows, BALL_STATE_SCHEMA)

        # Side-car provenance (schema-safe)
        prov_frame = []
        for r in rows:
            vs = str(r["visibility_state"])
            has_det = r["detection_confidence"] is not None
            if vs == "detected" and has_det:
                p = "detected"
            elif vs in {"occluded_short", "airborne"}:
                p = "tracked"
            elif vs == "predicted":
                p = "predicted"
            elif vs != "unknown" and r["ball_x_pixel"] is not None:
                p = "interpolated"
            else:
                p = "missing"
            prov_frame.append(
                {
                    "frame_id": int(r["frame_id"]),
                    "provenance": p,
                    "visible": bool(r["valid"]),
                    "ball_x_pixel": r["ball_x_pixel"],
                    "ball_y_pixel": r["ball_y_pixel"],
                }
            )
        prov_df = pd.DataFrame(prov_frame)
        prov_df.to_parquet(self.run_dir / "ball_provenance.parquet", index=False)
        counts = prov_df["provenance"].value_counts().to_dict()
        metrics = {
            "detector_backend": detector_backend,
            "football_raw_detections": int(len(football_rows)),
            "raw_detection_frames": int(raw_detected_frames),
            "raw_detection_coverage": round(raw_detected_frames / max(frames, 1), 4),
            "provenance_counts": {k: int(counts.get(k, 0)) for k in (
                "detected", "tracked", "predicted", "interpolated", "missing"
            )},
            "tracked_coverage": round(
                float(prov_df["provenance"].isin(["detected", "tracked", "predicted", "interpolated"]).mean()),
                4,
            ),
            "frames": int(frames),
        }
        write_json(self.stage_dir / "metrics.json", metrics)
        write_json(self.run_dir / "ball_detection_report.json", metrics)
        return {
            "ball_state": output,
            "football_ball_detections": self.run_dir / "football_ball_detections.parquet",
            "ball_provenance": self.run_dir / "ball_provenance.parquet",
            "metrics": self.stage_dir / "metrics.json",
        }

    @staticmethod
    def _field_position(
        x: float | None,
        y: float | None,
        calibration: pd.DataFrame,
        frame_id: int,
    ) -> tuple[float | None, float | None]:
        if x is None or y is None or frame_id not in calibration.index:
            return None, None
        row = calibration.loc[frame_id]
        if not bool(row["valid"]) or not row["homography_json"]:
            return None, None
        matrix = np.asarray(json.loads(row["homography_json"]), dtype=np.float64)
        point = cv2.perspectiveTransform(
            np.asarray([[[x, y]]], dtype=np.float64), matrix
        )[0, 0]
        return float(point[0]), float(point[1])

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        frame = pd.read_parquet(artifacts["ball_state"])
        validate_mvp2_columns("ball_state", list(frame.columns))
        if frame.empty:
            raise RuntimeError("ball state produced zero rows")
