"""Calibration and field-coordinate stage with a real provider chain.

Provider priority comes from config. Every provider attempt (including
failures and blocked providers) is recorded in ``provider_attempts.json``
inside the stage directory so the manifest reflects what actually ran.

Supported providers:
- ``sn_calibration``: recorded as blocked (baseline weights are not
  retrievable and the env lacks required dependencies); never faked.
- ``pnlcalib``: out-of-process keypoint inference in an isolated env.
  The worker only returns detected image/pitch point correspondences;
  homography fitting and all validity gates run in-process through
  ``calibration_from_mapping``. Produces per-frame calibrations at a
  configurable stride with a bounded hold window.
- ``metadata``: explicit calibration embedded in the video manifest.
- ``manual_json``: operator-verified manual calibration JSON.
- ``demo_four_point``: intentionally not implemented; recorded as such.
"""

from __future__ import annotations

import bisect
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from football_analytics.analytics.calibration import (
    CalibrationConfig,
    CalibrationResult,
    ManualJsonCalibrationProvider,
    MetadataCalibrationProvider,
    PitchOrientation,
    calibration_from_mapping,
)
from football_analytics.analytics.field_coordinates import FieldCoordinateTransformer
from football_analytics.contracts.schemas import (
    CALIBRATION_SCHEMA,
    GAME_STATE_SCHEMA,
    validate_mvp2_columns,
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


class CalibrationStage(Stage):
    name = "calibration"

    def validate_inputs(self) -> None:
        read_required_parquet(self.run_dir / "tracks.parquet")
        read_required_parquet(self.run_dir / "track_identities.parquet")
        read_required_parquet(self.run_dir / "shot_segments.parquet")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        cfg = self.config["calibration"]
        orientation = PitchOrientation(str(cfg["orientation"]))
        calibration_cfg = CalibrationConfig(
            pitch_length_m=float(cfg["pitch_length_m"]),
            pitch_width_m=float(cfg["pitch_width_m"]),
            maximum_reprojection_error=float(cfg["maximum_reprojection_error"]),
            minimum_visible_pitch_coverage=float(
                cfg["minimum_visible_pitch_coverage"]
            ),
            minimum_confidence=float(cfg["minimum_confidence"]),
            orientation=orientation,
        )
        frames = video_frame_count(self.run_dir)
        fps = video_fps(self.run_dir)

        attempts: list[dict[str, Any]] = []
        frame_results: dict[int, CalibrationResult] = {}
        hold_max_frames = 0
        fallback_reason = "no calibration provider succeeded"

        for provider_name in cfg.get("provider_priority", []):
            if frame_results:
                attempts.append(
                    {"provider": provider_name, "status": "not_attempted",
                     "detail": "earlier provider already succeeded"}
                )
                continue
            if provider_name == "sn_calibration":
                attempts.append(
                    {
                        "provider": "sn_calibration",
                        "status": "blocked",
                        "detail": (
                            "baseline weights not retrievable (Google Drive) and "
                            "sn-calibration env lacks required dependencies; "
                            "no placeholder result produced"
                        ),
                    }
                )
            elif provider_name == "pnlcalib":
                results, attempt, hold = self._run_pnlcalib(
                    cfg, calibration_cfg, frames
                )
                attempts.append(attempt)
                if results:
                    frame_results = results
                    hold_max_frames = hold
            elif provider_name == "metadata":
                payload = MetadataCalibrationProvider().load(
                    load_video_manifest(self.run_dir)
                )
                attempt, result = self._single_result(
                    "metadata", payload, calibration_cfg
                )
                attempts.append(attempt)
                if result is not None:
                    frame_results = {fid: result for fid in range(frames)}
                    hold_max_frames = 0
            elif provider_name == "manual_json":
                manual_path = cfg.get("manual_json")
                if not manual_path:
                    attempts.append(
                        {"provider": "manual_json", "status": "unavailable",
                         "detail": "calibration.manual_json not configured"}
                    )
                    continue
                payload = ManualJsonCalibrationProvider(Path(manual_path)).load()
                attempt, result = self._single_result(
                    "manual_json", payload, calibration_cfg,
                    detail_prefix=str(manual_path),
                )
                attempts.append(attempt)
                if result is not None:
                    frame_results = {fid: result for fid in range(frames)}
                    hold_max_frames = 0
            elif provider_name == "demo_four_point":
                attempts.append(
                    {
                        "provider": "demo_four_point",
                        "status": "not_implemented",
                        "detail": "invented correspondences are never generated",
                    }
                )
            else:
                attempts.append(
                    {"provider": provider_name, "status": "unknown_provider",
                     "detail": "no implementation for this provider name"}
                )

        if not frame_results:
            fallback_reason = "; ".join(
                f"{item['provider']}: {item['status']}" for item in attempts
            )
        invalid_result = CalibrationResult.invalid(
            fallback_reason, config=calibration_cfg
        )

        valid_sample_ids = sorted(
            fid for fid, res in frame_results.items() if res.valid
        )

        def effective(frame_id: int) -> CalibrationResult:
            direct = frame_results.get(frame_id)
            if direct is not None and direct.valid:
                return direct
            if valid_sample_ids:
                idx = bisect.bisect_left(valid_sample_ids, frame_id)
                best: int | None = None
                for candidate_idx in (idx - 1, idx):
                    if 0 <= candidate_idx < len(valid_sample_ids):
                        candidate = valid_sample_ids[candidate_idx]
                        if best is None or abs(candidate - frame_id) < abs(best - frame_id):
                            best = candidate
                if best is not None and abs(best - frame_id) <= hold_max_frames:
                    return frame_results[best]
                return CalibrationResult.invalid(
                    "no valid calibration within hold window",
                    provider=frame_results[valid_sample_ids[0]].provider,
                    config=calibration_cfg,
                )
            if direct is not None:
                return direct
            return invalid_result

        shots = read_required_parquet(self.run_dir / "shot_segments.parquet").set_index(
            "frame_id"
        )
        tracks = read_required_parquet(self.run_dir / "tracks.parquet")
        identities = read_required_parquet(
            self.run_dir / "track_identities.parquet"
        ).set_index(["frame_id", "track_id"])
        valid_shots = set(self.config["shot_classifier"]["valid_for_spatial"])
        segment_ids: dict[int, int] = {}
        segment_id = 0
        for frame_id in range(frames):
            if frame_id in shots.index and bool(shots.loc[frame_id, "scene_cut"]):
                segment_id += 1
            segment_ids[frame_id] = segment_id

        calibration_rows: list[dict[str, Any]] = []
        transformers: dict[int, FieldCoordinateTransformer] = {}
        for frame_id in range(frames):
            result = effective(frame_id)
            transformers[frame_id] = FieldCoordinateTransformer(
                result, normalize_orientation=True
            )
            shot_type = (
                str(shots.loc[frame_id, "shot_type"])
                if frame_id in shots.index
                else "unknown"
            )
            shot_valid = shot_type in valid_shots
            valid = bool(result.valid and shot_valid)
            reason = (
                result.invalid_reason
                if not result.valid
                else (None if shot_valid else f"shot_type_{shot_type}_not_spatial")
            )
            calibration_rows.append(
                {
                    **canonical_common(
                        self.run_dir,
                        frame_id,
                        frame_id * 1000.0 / fps,
                        result.provider or "calibration_provider_chain",
                        result.confidence if valid else 0.0,
                        valid,
                    ),
                    "segment_id": segment_ids[frame_id],
                    "provider": result.provider,
                    "homography_json": (
                        json.dumps(result.homography.tolist())
                        if valid and result.homography is not None
                        else None
                    ),
                    "orientation": (
                        result.orientation.value if result.orientation else None
                    ),
                    "pitch_length_m": result.pitch_length_m,
                    "pitch_width_m": result.pitch_width_m,
                    "reprojection_error": result.reprojection_error,
                    "visible_pitch_coverage": result.visible_pitch_coverage,
                    "invalid_reason": reason,
                }
            )

        game_rows: list[dict[str, Any]] = []
        for item in tracks.itertuples():
            frame_id = int(item.frame_id)
            result = effective(frame_id)
            shot_type = (
                str(shots.loc[frame_id, "shot_type"])
                if frame_id in shots.index
                else "unknown"
            )
            coordinate = (
                transformers[frame_id].transform_point(
                    (item.foot_x_pixel, item.foot_y_pixel)
                )
                if result.valid and shot_type in valid_shots
                else None
            )
            identity = (
                identities.loc[(frame_id, int(item.track_id))]
                if (frame_id, int(item.track_id)) in identities.index
                else None
            )
            valid = coordinate is not None
            game_rows.append(
                {
                    **canonical_common(
                        self.run_dir,
                        frame_id,
                        float(item.timestamp_ms),
                        "calibrated_foot_point",
                        result.confidence if valid else 0.0,
                        valid,
                    ),
                    "track_id": int(item.track_id),
                    "role": identity["role"] if identity is not None else None,
                    "team_id": identity["team_id"] if identity is not None else None,
                    "foot_x_pixel": float(item.foot_x_pixel),
                    "foot_y_pixel": float(item.foot_y_pixel),
                    "x_field": coordinate.x if coordinate else None,
                    "y_field": coordinate.y if coordinate else None,
                    "shot_type": shot_type,
                    "calibration_confidence": result.confidence if valid else 0.0,
                    "invalid_reason": (
                        None
                        if valid
                        else (
                            result.invalid_reason
                            or (
                                f"shot_type_{shot_type}_not_spatial"
                                if shot_type not in valid_shots
                                else "foot_point_outside_pitch"
                            )
                        )
                    ),
                }
            )

        calibration_path = self.run_dir / "calibration.parquet"
        game_state_path = self.run_dir / "game_state.parquet"
        attempts_path = self.stage_dir / "provider_attempts.json"
        write_rows_with_schema(calibration_path, calibration_rows, CALIBRATION_SCHEMA)
        write_rows_with_schema(game_state_path, game_rows, GAME_STATE_SCHEMA)
        write_json(attempts_path, {"provider_attempts": attempts})
        return {
            "calibration": calibration_path,
            "game_state": game_state_path,
            "provider_attempts": attempts_path,
        }

    def _single_result(
        self,
        provider: str,
        payload: Mapping[str, Any] | None,
        calibration_cfg: CalibrationConfig,
        detail_prefix: str = "",
    ) -> tuple[dict[str, Any], CalibrationResult | None]:
        if payload is None:
            return (
                {"provider": provider, "status": "unavailable",
                 "detail": (detail_prefix + " not found").strip()},
                None,
            )
        result = calibration_from_mapping(payload, provider, calibration_cfg)
        if result.valid:
            detail = (
                f"reprojection_error={result.reprojection_error:.3f}m "
                f"coverage={result.visible_pitch_coverage:.3f} "
                f"confidence={result.confidence:.3f}"
            )
            label = payload.get("label")
            if label:
                detail = f"label={label} {detail}"
            return (
                {"provider": provider, "status": "valid", "detail": detail},
                result,
            )
        return (
            {"provider": provider, "status": "invalid",
             "detail": result.invalid_reason or "unknown"},
            None,
        )

    def _run_pnlcalib(
        self,
        cfg: Mapping[str, Any],
        calibration_cfg: CalibrationConfig,
        frames: int,
    ) -> tuple[dict[int, CalibrationResult], dict[str, Any], int]:
        pnl = cfg.get("pnlcalib") or {}
        attempt: dict[str, Any] = {"provider": "pnlcalib"}
        if not pnl.get("enabled", False):
            attempt.update(status="disabled", detail="calibration.pnlcalib.enabled=false")
            return {}, attempt, 0
        python = Path(str(pnl.get("python", "")))
        root = Path(str(pnl.get("root", "")))
        weights_kp = Path(str(pnl.get("weights_kp", "")))
        weights_line = Path(str(pnl.get("weights_line", "")))
        missing = [
            str(path)
            for path in (python, root / "inference.py", weights_kp, weights_line)
            if not path.exists()
        ]
        if missing:
            attempt.update(
                status="blocked",
                detail="missing prerequisites: " + ", ".join(missing),
            )
            return {}, attempt, 0
        manifest = load_video_manifest(self.run_dir)
        video_path = Path(manifest["working_path"])
        if not video_path.is_file():
            attempt.update(status="blocked", detail=f"video missing: {video_path}")
            return {}, attempt, 0
        stride = max(1, int(pnl.get("frame_stride", 5)))
        hold = max(stride, int(pnl.get("hold_max_frames", 2 * stride)))
        sample_ids = sorted(set(range(0, frames, stride)) | ({frames - 1} if frames else set()))
        worker = Path(__file__).resolve().parents[3] / "scripts" / "pnlcalib_worker.py"
        if not worker.is_file():
            worker = Path.cwd() / "scripts" / "pnlcalib_worker.py"
        if not worker.is_file():
            attempt.update(status="blocked", detail="pnlcalib_worker.py not found")
            return {}, attempt, 0
        with tempfile.NamedTemporaryFile(
            suffix=".json", dir=self.stage_dir, delete=False
        ) as handle:
            output_path = Path(handle.name)
        command = [
            str(python),
            str(worker),
            "--pnlcalib-root", str(root),
            "--weights-kp", str(weights_kp),
            "--weights-line", str(weights_line),
            "--video", str(video_path),
            "--frame-ids", ",".join(str(v) for v in sample_ids),
            "--output", str(output_path),
            "--device", str(pnl.get("device", "cuda:0")),
            "--kp-threshold", str(pnl.get("kp_threshold", 0.3434)),
            "--line-threshold", str(pnl.get("line_threshold", 0.7867)),
            "--min-keypoints", str(pnl.get("min_keypoints", 6)),
        ]
        log_path = self.stage_dir / "pnlcalib_worker.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=float(pnl.get("timeout_seconds", 1800)),
                    check=False,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            attempt.update(status="failed", detail=f"worker error: {exc}")
            return {}, attempt, 0
        if completed.returncode != 0 or not output_path.is_file():
            attempt.update(
                status="failed",
                detail=f"worker exit={completed.returncode}, log={log_path}",
            )
            return {}, attempt, 0
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        output_path.replace(self.stage_dir / "pnlcalib_frames.json")
        results: dict[int, CalibrationResult] = {}
        valid_count = 0
        for frame in payload.get("frames", []):
            frame_id = int(frame["frame_id"])
            if frame.get("status") != "ok":
                results[frame_id] = CalibrationResult.invalid(
                    f"pnlcalib: {frame.get('status')}",
                    provider="pnlcalib",
                    config=calibration_cfg,
                )
                continue
            result = calibration_from_mapping(
                {
                    "image_points": frame["image_points"],
                    "pitch_points": frame["pitch_points"],
                },
                "pnlcalib",
                calibration_cfg,
            )
            results[frame_id] = result
            if result.valid:
                valid_count += 1
        if valid_count == 0:
            attempt.update(
                status="invalid",
                detail=f"0/{len(sample_ids)} sampled frames passed validity gates",
            )
            return {}, attempt, 0
        attempt.update(
            status="valid",
            detail=(
                f"{valid_count}/{len(sample_ids)} sampled frames valid, "
                f"stride={stride}, hold_max_frames={hold}, "
                f"worker_elapsed_s={payload.get('elapsed_seconds')}"
            ),
        )
        return results, attempt, hold

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        calibration = pd.read_parquet(artifacts["calibration"])
        game_state = pd.read_parquet(artifacts["game_state"])
        validate_mvp2_columns("calibration", list(calibration.columns))
        validate_mvp2_columns("game_state", list(game_state.columns))
        if calibration.empty:
            raise RuntimeError("calibration stage produced zero frame rows")
