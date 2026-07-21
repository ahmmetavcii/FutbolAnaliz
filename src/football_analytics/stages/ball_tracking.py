"""Ball tracking stage: enhance ball_state → ball_trajectory.parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from football_analytics.events.ball_trajectory import (
    BallTrajectoryConfig,
    load_ball_trajectory_from_ball_state,
    write_ball_trajectory,
)
from football_analytics.opta.ball_recovery import (
    BallRecoveryConfig,
    enhance_ball_state,
    write_ball_detector_adapter_spec,
)
from football_analytics.evaluation.wrong_object import (
    WrongObjectConfig,
    filter_ball_trajectory_candidates,
)
from football_analytics.stages.base import Stage
from football_analytics.stages.mvp2_common import load_video_manifest, read_required_parquet
from football_analytics.utils.io import write_json


class BallTrackingStage(Stage):
    name = "ball_tracking"

    def validate_inputs(self) -> None:
        read_required_parquet(self.run_dir / "ball_state.parquet")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        ball_state = read_required_parquet(self.run_dir / "ball_state.parquet")
        cfg_raw = self.config.get("ball_trajectory") or {}
        recovery_raw = self.config.get("ball_recovery") or {}

        # Before coverage (raw detections only)
        raw_detected = (
            ball_state["visibility_state"].astype(str).eq("detected").mean()
            if not ball_state.empty and "visibility_state" in ball_state.columns
            else 0.0
        )
        before_tracked = (
            ball_state["visibility_state"]
            .astype(str)
            .isin(["detected", "predicted", "occluded_short", "airborne"])
            .mean()
            if not ball_state.empty and "visibility_state" in ball_state.columns
            else 0.0
        )

        video_path = Path()
        try:
            manifest = load_video_manifest(self.run_dir)
            video_path = Path(manifest.get("working_path") or "")
        except FileNotFoundError:
            manifest = {}
        model_path = (
            recovery_raw.get("model_path")
            or cfg_raw.get("ball_model_path")
            or (self.config.get("model") or {}).get("path")
        )
        rec_cfg = BallRecoveryConfig(
            ball_class_id=int((cfg_raw.get("ball_class_ids") or [32])[0]),
            roi_conf=float(recovery_raw.get("roi_conf", 0.12)),
            roi_imgsz=int(recovery_raw.get("roi_imgsz", 320)),
            roi_half_size=int(recovery_raw.get("roi_half_size", 160)),
            max_interp_frames=int(
                recovery_raw.get("max_interp_frames", cfg_raw.get("maximum_gap_frames", 5))
            ),
            max_interp_seconds=float(
                recovery_raw.get("max_interp_seconds", cfg_raw.get("maximum_gap_seconds", 0.25))
            ),
            max_pixel_speed=float(
                recovery_raw.get(
                    "max_pixel_speed", cfg_raw.get("maximum_pixel_speed_per_second", 2500.0)
                )
            ),
            enable_roi_search=bool(recovery_raw.get("enable_roi_search", True)),
            enable_optical_flow=bool(recovery_raw.get("enable_optical_flow", True)),
            model_path=str(model_path) if model_path else None,
            device=(self.config.get("model") or {}).get("device", 0),
        )
        enhanced, coverage = enhance_ball_state(
            ball_state, video_path=video_path if video_path.is_file() else None, config=rec_cfg
        )
        # Wrong-object lock rejection on football detections (if present)
        fb_path = self.run_dir / "football_ball_detections.parquet"
        if fb_path.is_file():
            fb = pd.read_parquet(fb_path)
            filtered, wo = filter_ball_trajectory_candidates(
                fb, config=WrongObjectConfig(max_prediction_frames=int(recovery_raw.get("max_prediction_frames", 8)))
            )
            coverage["wrong_object_switches"] = wo.get("wrong_object_switches", 0)
            coverage["trajectory_jumps"] = wo.get("trajectory_jumps", 0)
            filtered.to_parquet(self.run_dir / "football_ball_detections_filtered.parquet", index=False)
            write_json(self.run_dir / "wrong_object_ball_report.json", wo)
        else:
            coverage.setdefault("wrong_object_switches", 0)
            coverage.setdefault("trajectory_jumps", 0)

        coverage["raw_detection_coverage_before"] = round(float(raw_detected), 4)
        coverage["tracked_coverage_before"] = round(float(before_tracked), 4)
        enhanced.to_parquet(self.run_dir / "ball_state_enhanced.parquet", index=False)
        write_ball_detector_adapter_spec(self.run_dir / "ball_detector_adapter.md")

        # Prefer football-detector raw recall when ball_detection_report exists.
        # visibility_state=="detected" can be lower because the Kalman/track path
        # may label some YOLO hits as occluded_short/airborne ("tracked").
        det_report_path = self.run_dir / "ball_detection_report.json"
        if det_report_path.is_file():
            import json

            det_report = json.loads(det_report_path.read_text(encoding="utf-8"))
            football_raw = det_report.get("raw_detection_coverage")
            if football_raw is not None:
                coverage["ball_state_detected_coverage"] = coverage.get(
                    "raw_detection_coverage"
                )
                coverage["raw_detection_coverage"] = round(float(football_raw), 4)
                coverage["football_raw_detection_frames"] = int(
                    det_report.get("raw_detection_frames") or 0
                )
                coverage["detector_backend"] = det_report.get("detector_backend")
            if det_report.get("provenance_counts"):
                coverage["provenance_counts"] = det_report["provenance_counts"]
            if det_report.get("tracked_coverage") is not None:
                coverage["provenance_tracked_coverage"] = det_report["tracked_coverage"]

        traj_cfg = BallTrajectoryConfig(
            max_interp_gap_frames=int(cfg_raw.get("maximum_gap_frames", 5)),
            source_camera=str(cfg_raw.get("source_camera", "camera_1")),
        )
        ball = load_ball_trajectory_from_ball_state(enhanced, config=traj_cfg)
        if not ball.empty:
            ball = ball.sort_values("frame_id").reset_index(drop=True)
            velocities: list[float | None] = [None]
            directions: list[float | None] = [None]
            interp_lens: list[int] = [0]
            run = 0
            for i in range(1, len(ball)):
                if bool(ball.loc[i, "interpolated"]):
                    run += 1
                else:
                    run = 0
                interp_lens.append(run)
                dt_s = (
                    float(ball.loc[i, "timestamp_ms"]) - float(ball.loc[i - 1, "timestamp_ms"])
                ) / 1000.0
                if (
                    dt_s <= 0
                    or not ball.loc[i, "visible"]
                    or not ball.loc[i - 1, "visible"]
                    or pd.isna(ball.loc[i, "pitch_x"])
                    or pd.isna(ball.loc[i - 1, "pitch_x"])
                ):
                    velocities.append(None)
                    directions.append(None)
                    continue
                dx = float(ball.loc[i, "pitch_x"]) - float(ball.loc[i - 1, "pitch_x"])
                dy = float(ball.loc[i, "pitch_y"]) - float(ball.loc[i - 1, "pitch_y"])
                import math

                velocities.append((dx * dx + dy * dy) ** 0.5 / dt_s)
                directions.append(math.atan2(dy, dx))
            ball["velocity_mps"] = velocities
            ball["direction"] = directions
            ball["interpolation_length"] = interp_lens
            ball["camera_id"] = traj_cfg.source_camera
            ball["frame_index"] = ball["frame_id"]
            ball["timestamp"] = ball["timestamp_ms"] / 1000.0
            if "ball_confidence" in ball.columns:
                ball["confidence"] = ball["ball_confidence"]

        out = self.run_dir / "ball_trajectory.parquet"
        write_ball_trajectory(out, ball)
        metrics = {
            **coverage,
            "frames": int(len(ball)),
            "visible_frames": int(ball["visible"].sum()) if not ball.empty else 0,
            "interpolated_frames": int(ball["interpolated"].sum()) if not ball.empty else 0,
        }
        write_json(self.run_dir / "ball_coverage_report.json", metrics)
        write_json(self.stage_dir / "metrics.json", metrics)
        (self.run_dir / "stage_manifests").mkdir(parents=True, exist_ok=True)
        write_json(
            self.run_dir / "stage_manifests" / "ball_tracking.json",
            {"stage": self.name, "status": "PASS", **metrics},
        )
        return {
            "ball_trajectory": out,
            "ball_coverage_report": self.run_dir / "ball_coverage_report.json",
            "metrics": self.stage_dir / "metrics.json",
        }

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        if not (self.run_dir / "ball_trajectory.parquet").is_file():
            raise FileNotFoundError("ball_trajectory.parquet")
