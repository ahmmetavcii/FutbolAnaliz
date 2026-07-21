"""Pipeline orchestration for MVP-1."""

from __future__ import annotations

import datetime as dt
import os
import resource
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch

from football_analytics.stages.detection import DetectionStage
from football_analytics.stages.ingest import IngestStage
from football_analytics.stages.tracking import TrackingStage
from football_analytics.utils.hashing import sha256_file
from football_analytics.utils.io import read_yaml, write_json


def make_run_id() -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{stamp}_{uuid.uuid4().hex[:6]}"


class PipelineRunner:
    def __init__(
        self,
        config_path: Path,
        input_video: Path,
        runs_root: Path,
        resume_run_dir: Path | None = None,
        rerun_from: str | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.input_video = Path(input_video)
        self.runs_root = Path(runs_root)
        self.config = read_yaml(self.config_path)
        self.rerun_from = rerun_from
        if rerun_from and resume_run_dir is None:
            raise ValueError("--rerun-from requires --resume-run-dir")
        if resume_run_dir is None:
            self.run_id = make_run_id()
            self.run_dir = self.runs_root / self.run_id
        else:
            self.run_dir = Path(resume_run_dir)
            self.run_id = self.run_dir.name

    def _apply_thread_limits(self) -> None:
        runtime = self.config.get("runtime", {})
        os.environ.setdefault("OMP_NUM_THREADS", str(runtime.get("omp_num_threads", 2)))
        os.environ.setdefault("MKL_NUM_THREADS", str(runtime.get("mkl_num_threads", 2)))

    def run(self) -> dict[str, Any]:
        self._apply_thread_limits()
        minimum_gb = float(
            self.config.get("runtime", {}).get("minimum_free_disk_gb", 0.0)
        )
        self.runs_root.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(self.runs_root).free / (1024**3)
        if free_gb < minimum_gb:
            raise RuntimeError(
                f"Insufficient free disk: {free_gb:.2f} GiB < {minimum_gb:.2f} GiB"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            self.run_dir / "run_config.json",
            {
                "run_id": self.run_id,
                "config_path": str(self.config_path),
                "input_video": str(self.input_video),
                "config": self.config,
            },
        )

        stage_results: dict[str, Any] = {}
        started = dt.datetime.now().astimezone()

        stages: list[Any] = [
            IngestStage(self.run_dir, self.config, self.input_video),
        ]
        pipeline_name = self.config.get("pipeline", {}).get("name")
        if pipeline_name in {"mvp2_spatial_analytics", "opta_analytics"}:
            from football_analytics.stages.analytics_render import AnalyticsRenderStage
            from football_analytics.stages.ball_state import BallStateStage
            from football_analytics.stages.calibration import CalibrationStage
            from football_analytics.stages.camera_motion import CameraMotionStage
            from football_analytics.stages.metrics import MetricsStage
            from football_analytics.stages.possession import PossessionStage
            from football_analytics.stages.shot_classification import (
                ShotClassificationStage,
            )
            from football_analytics.stages.reid import ReidStage
            from football_analytics.stages.team_identity import TeamIdentityStage
            from football_analytics.stages.track_quality import TrackQualityStage

            stages.append(ShotClassificationStage(self.run_dir, self.config))
            stages.extend(
                [
                    DetectionStage(self.run_dir, self.config),
                    TrackingStage(self.run_dir, self.config),
                    TrackQualityStage(self.run_dir, self.config),
                    ReidStage(self.run_dir, self.config),
                    TeamIdentityStage(self.run_dir, self.config),
                    CameraMotionStage(self.run_dir, self.config),
                    CalibrationStage(self.run_dir, self.config),
                    BallStateStage(self.run_dir, self.config),
                    PossessionStage(self.run_dir, self.config),
                    MetricsStage(self.run_dir, self.config),
                    AnalyticsRenderStage(self.run_dir, self.config),
                ]
            )
            if pipeline_name == "mvp2_spatial_analytics":
                from football_analytics.stages.event_detection import EventDetectionStage

                stages.append(EventDetectionStage(self.run_dir, self.config))
            else:
                from football_analytics.stages.action_inference import ActionInferenceStage
                from football_analytics.stages.ball_tracking import BallTrackingStage
                from football_analytics.stages.global_identity import GlobalIdentityStage
                from football_analytics.stages.opta_analytics import OptaAnalyticsStage
                from football_analytics.stages.touch_inference import TouchInferenceStage

                stages.extend(
                    [
                        GlobalIdentityStage(self.run_dir, self.config),
                        BallTrackingStage(self.run_dir, self.config),
                        TouchInferenceStage(self.run_dir, self.config),
                        ActionInferenceStage(self.run_dir, self.config),
                        OptaAnalyticsStage(self.run_dir, self.config),
                    ]
                )
        else:
            stages.extend(
                [
                    DetectionStage(self.run_dir, self.config),
                    TrackingStage(self.run_dir, self.config),
                ]
            )
        if self.rerun_from is not None:
            stage_names = [stage.name for stage in stages]
            if self.rerun_from not in stage_names:
                raise ValueError(
                    f"--rerun-from stage '{self.rerun_from}' not in pipeline: "
                    f"{stage_names}"
                )
            boundary = stage_names.index(self.rerun_from)
        else:
            boundary = None
        for index, stage in enumerate(stages):
            if boundary is None:
                mode = "auto"
            elif index < boundary:
                # Stages before the rerun boundary are trusted: skip only if
                # their artifacts still checksum-validate; never recompute.
                mode = "trust"
            else:
                mode = "force"
            stage_results[stage.name] = {
                key: str(value) for key, value in stage.execute(mode=mode).items()
            }

        finished = dt.datetime.now().astimezone()
        required = [
            self.run_dir / "video_manifest.json",
            self.run_dir / "detections.parquet",
            self.run_dir / "tracks.parquet",
            self.run_dir / "annotated_video.mp4",
            self.run_dir / "stages" / "ingest" / "stage_manifest.json",
            self.run_dir / "stages" / "detection" / "stage_manifest.json",
            self.run_dir / "stages" / "tracking" / "stage_manifest.json",
        ]
        pipeline_name = self.config.get("pipeline", {}).get("name")
        if pipeline_name in {"mvp2_spatial_analytics", "opta_analytics"}:
            required.extend(
                [
                    self.run_dir / "shot_segments.parquet",
                    self.run_dir / "track_quality.parquet",
                    self.run_dir / "reid_embeddings.parquet",
                    self.run_dir / "track_reid_prototypes.parquet",
                    self.run_dir / "track_identities.parquet",
                    self.run_dir / "camera_motion.parquet",
                    self.run_dir / "calibration.parquet",
                    self.run_dir / "game_state.parquet",
                    self.run_dir / "ball_state.parquet",
                    self.run_dir / "possession_timeline.parquet",
                    self.run_dir / "team_possession_summary.json",
                    self.run_dir / "player_metrics.parquet",
                    self.run_dir / "team_metrics.parquet",
                    self.run_dir / "analytics_annotated.mp4",
                    self.run_dir / "tactical_preview.mp4",
                    self.run_dir / "team_possession_chart.png",
                    self.run_dir / "player_speed_summary.csv",
                ]
            )
            stage_names = [
                "shot_classification",
                "track_quality",
                "reid",
                "team_identity",
                "camera_motion",
                "calibration",
                "ball_state",
                "possession",
                "metrics",
                "analytics_render",
            ]
            if pipeline_name == "mvp2_spatial_analytics":
                required.extend(
                    [
                        self.run_dir / "match_events.parquet",
                        self.run_dir / "events.parquet",
                        self.run_dir / "stage_manifests" / "event_detection.json",
                    ]
                )
                stage_names.append("event_detection")
            else:
                required.extend(
                    [
                        self.run_dir / "global_identity_map.parquet",
                        self.run_dir / "identity_quality.json",
                        self.run_dir / "ball_trajectory.parquet",
                        self.run_dir / "ball_coverage_report.json",
                        self.run_dir / "touch_events.parquet",
                        self.run_dir / "pass_events.parquet",
                        self.run_dir / "dribble_events.parquet",
                        self.run_dir / "duel_events.parquet",
                        self.run_dir / "defensive_actions.parquet",
                        self.run_dir / "player_opta_summary.csv",
                        self.run_dir / "team_opta_summary.csv",
                        self.run_dir / "opta_stats_publishable.json",
                        self.run_dir / "pitch_zones.json",
                        self.run_dir / "stage_manifests" / "global_identity.json",
                        self.run_dir / "stage_manifests" / "ball_tracking.json",
                        self.run_dir / "stage_manifests" / "touch_inference.json",
                        self.run_dir / "stage_manifests" / "action_inference.json",
                        self.run_dir / "stage_manifests" / "opta_analytics.json",
                    ]
                )
                stage_names.extend(
                    [
                        "global_identity",
                        "ball_tracking",
                        "touch_inference",
                        "action_inference",
                        "opta_analytics",
                    ]
                )
            required.extend(
                self.run_dir / "stages" / name / "stage_manifest.json"
                for name in stage_names
            )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"Pipeline incomplete, missing: {missing}")

        report = {
            "run_id": self.run_id,
            "status": "PASS",
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "total_seconds": (finished - started).total_seconds(),
            "input_video": str(self.input_video),
            "run_dir": str(self.run_dir),
            "model": self.config.get("model"),
            "trackers": self.config.get("tracking", {}).get("trackers"),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_ram_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "free_disk_gb_at_start": free_gb,
            "stages": stage_results,
            "artifact_checksums": {
                path.name: sha256_file(path)
                for path in required
                if path.is_file() and path.suffix in {".json", ".parquet", ".mp4"}
            },
        }
        # Enrich with stage metrics if present.
        det_metrics = self.run_dir / "stages" / "detection" / "metrics.json"
        track_cmp = self.run_dir / "stages" / "tracking" / "tracker_comparison.json"
        if det_metrics.exists():
            import json

            report["detection_metrics"] = json.loads(det_metrics.read_text())
        if track_cmp.exists():
            import json

            report["tracking_metrics"] = json.loads(track_cmp.read_text())

        write_json(self.run_dir / "run_report.json", report)
        return report
