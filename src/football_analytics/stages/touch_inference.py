"""Touch inference stage with ankle proxy + contact-sheet export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from football_analytics.events.touch_inference import TouchInferenceConfig, infer_touches
from football_analytics.opta.aggregate import enrich_touches_with_ball_pitch
from football_analytics.opta.touch_debug import enrich_tracks_with_ankle, export_touch_contact_sheets
from football_analytics.opta.touch_dedup import deduplicate_touches
from football_analytics.stages.base import Stage
from football_analytics.stages.mvp2_common import load_video_manifest, read_required_parquet
from football_analytics.utils.io import write_json


class TouchInferenceStage(Stage):
    name = "touch_inference"

    def validate_inputs(self) -> None:
        read_required_parquet(self.run_dir / "ball_trajectory.parquet")
        read_required_parquet(self.run_dir / "tracks.parquet")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        ball = read_required_parquet(self.run_dir / "ball_trajectory.parquet")
        tracks = enrich_tracks_with_ankle(read_required_parquet(self.run_dir / "tracks.parquet"))
        identities_path = self.run_dir / "track_identities.parquet"
        identities = pd.read_parquet(identities_path) if identities_path.is_file() else None
        global_path = self.run_dir / "global_identity_map.parquet"
        global_map = pd.read_parquet(global_path) if global_path.is_file() else None
        cfg_raw = self.config.get("touch_inference") or {}
        cfg = TouchInferenceConfig(
            **{
                k: v
                for k, v in cfg_raw.items()
                if k in TouchInferenceConfig.__dataclass_fields__
            }
        )
        touches = infer_touches(
            ball, tracks, identities=identities, global_map=global_map, config=cfg
        )
        touches = enrich_touches_with_ball_pitch(touches, ball)
        raw_touch_count = int(len(touches))
        dedup_window = float(cfg_raw.get("dedup_window_ms", 300.0))
        touches = deduplicate_touches(touches, window_ms=dedup_window)

        # Enrich evidence columns from ball velocity series
        if not touches.empty and not ball.empty and "velocity_mps" in ball.columns:
            ball_idx = ball.set_index("frame_id")
            vel_changes = []
            dir_changes = []
            for row in touches.itertuples(index=False):
                fid = int(row.frame_id)
                if fid in ball_idx.index and (fid - 1) in ball_idx.index:
                    v0 = ball_idx.loc[fid - 1].get("velocity_mps")
                    v1 = ball_idx.loc[fid].get("velocity_mps")
                    d0 = ball_idx.loc[fid - 1].get("direction")
                    d1 = ball_idx.loc[fid].get("direction")
                    try:
                        vel_changes.append(
                            abs(float(v1) - float(v0))
                            if pd.notna(v0) and pd.notna(v1)
                            else None
                        )
                    except (TypeError, ValueError):
                        vel_changes.append(None)
                    try:
                        dir_changes.append(
                            abs(float(d1) - float(d0))
                            if pd.notna(d0) and pd.notna(d1)
                            else None
                        )
                    except (TypeError, ValueError):
                        dir_changes.append(None)
                else:
                    vel_changes.append(None)
                    dir_changes.append(None)
            touches = touches.copy()
            touches["ball_velocity_change"] = vel_changes
            touches["ball_direction_change"] = dir_changes
            touches["pose_evidence"] = False  # ankle proxy, not true pose model
            touches["touch_type"] = touches.get("touch_type", "unknown")
            if "touch_type" not in touches.columns or touches["touch_type"].isna().all():
                touches["touch_type"] = "unknown"
            touches["status"] = touches["confidence"].apply(
                lambda c: "confirmed"
                if float(c) >= 0.65
                else ("candidate" if float(c) >= 0.40 else "unresolved")
            )
            touches["timestamp"] = touches["timestamp_ms"] / 1000.0
            if "distance_to_ball" not in touches.columns and "distance_m" in touches.columns:
                touches["distance_to_ball"] = touches["distance_m"]

        out = self.run_dir / "touch_events.parquet"
        touches.to_parquet(out, index=False)

        # Visual debug / contact sheets
        video_path = Path()
        try:
            manifest = load_video_manifest(self.run_dir)
            video_path = Path(manifest.get("working_path") or "")
        except FileNotFoundError:
            pass
        debug_dir = self.run_dir / "touch_debug"
        if debug_dir.is_dir():
            for old in debug_dir.glob("*"):
                try:
                    old.unlink()
                except OSError:
                    pass
        debug_meta = {"written": 0, "path": str(debug_dir)}
        if video_path.is_file() and not touches.empty:
            debug_meta = export_touch_contact_sheets(
                touches,
                video_path,
                debug_dir,
                tracks=tracks,
                ball=ball,
            )

        metrics = {
            "touch_count": int(len(touches)),
            "raw_touch_count_before_dedup": raw_touch_count,
            "dedup_window_ms": dedup_window,
            "confirmed_count": int((touches["status"] == "confirmed").sum())
            if not touches.empty and "status" in touches.columns
            else 0,
            "candidate_count": int((touches["status"] == "candidate").sum())
            if not touches.empty and "status" in touches.columns
            else 0,
            "contact_sheets": debug_meta,
        }
        write_json(self.stage_dir / "metrics.json", metrics)
        (self.run_dir / "stage_manifests").mkdir(parents=True, exist_ok=True)
        write_json(
            self.run_dir / "stage_manifests" / "touch_inference.json",
            {"stage": self.name, "status": "PASS", **metrics},
        )
        return {
            "touch_events": out,
            "touch_debug": debug_dir,
            "metrics": self.stage_dir / "metrics.json",
        }

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        if not (self.run_dir / "touch_events.parquet").is_file():
            raise FileNotFoundError("touch_events.parquet")
