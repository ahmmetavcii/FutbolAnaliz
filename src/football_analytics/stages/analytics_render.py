"""Final streaming analytics rendering stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from football_analytics.stages.base import Stage
from football_analytics.stages.mvp2_common import (
    read_required_parquet,
    video_frame_count,
    video_fps,
)
from football_analytics.opta.overlay_slots import (
    build_overlay_slot_assignments,
    frame_display_lookup,
)
from football_analytics.visualization.analytics_renderer import (
    render_analytics_video,
    render_tactical_preview,
)


class AnalyticsRenderStage(Stage):
    name = "analytics_render"

    def validate_inputs(self) -> None:
        for name in (
            "tracks",
            "track_identities",
            "ball_state",
            "possession_timeline",
            "player_metrics",
            "camera_motion",
            "calibration",
            "game_state",
        ):
            read_required_parquet(self.run_dir / f"{name}.parquet")
        if not (self.run_dir / "input" / "test_clip.mp4").is_file():
            raise FileNotFoundError("working video missing")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        read = lambda name: read_required_parquet(self.run_dir / f"{name}.parquet")
        tracks = read("tracks")
        identities = read("track_identities")
        ball = read("ball_state")
        possession = read("possession_timeline")
        metrics = read("player_metrics")
        motion = read("camera_motion")
        calibration = read("calibration")
        game_state = read("game_state")

        display_id_by_track: dict[int, int] = {}
        team_by_track: dict[int, str] = {}
        frame_display_ids: dict[tuple[int, int], int] = {}
        stable_path = self.run_dir / "stable_track_map.parquet"
        gmap_path = self.run_dir / "global_identity_map.parquet"

        # Overlay slots: frame-to-frame identity for video labels (fixes multi-ID).
        overlay_cfg = (self.config.get("render") or {})
        stable_for_seed = pd.read_parquet(stable_path) if stable_path.is_file() else None
        frame_map, slot_track_map = build_overlay_slot_assignments(
            tracks,
            identities,
            max_slots_per_team=int(overlay_cfg.get("max_slots_per_team", 12)),
            match_dist_px=float(overlay_cfg.get("slot_match_dist_px", 200.0)),
            hold_frames=int(overlay_cfg.get("slot_hold_frames", 240)),
            stable_map=stable_for_seed,
        )
        frame_map_path = self.run_dir / "overlay_slot_frames.parquet"
        slot_map_path = self.run_dir / "overlay_slot_map.parquet"
        frame_map.to_parquet(frame_map_path, index=False)
        slot_track_map.to_parquet(slot_map_path, index=False)
        frame_display_ids = frame_display_lookup(frame_map)
        if not slot_track_map.empty:
            display_id_by_track = {
                int(r.local_track_id): int(r.display_id)
                for r in slot_track_map.itertuples(index=False)
            }
            team_by_track = {
                int(r.local_track_id): str(r.team_id)
                for r in slot_track_map.itertuples(index=False)
                if r.team_id is not None and str(r.team_id) not in {"", "nan", "None"}
            }
        elif stable_path.is_file():
            stable = pd.read_parquet(stable_path)
            display_id_by_track = {
                int(r.local_track_id): int(r.display_id) for r in stable.itertuples(index=False)
            }
            team_by_track = {
                int(r.local_track_id): str(r.team_id)
                for r in stable.itertuples(index=False)
                if r.team_id is not None and str(r.team_id) not in {"", "nan", "None"}
            }
        elif gmap_path.is_file():
            gmap = pd.read_parquet(gmap_path)
            display_id_by_track = {
                int(r.local_track_id): int(r.global_id) for r in gmap.itertuples(index=False)
            }

        analytics_video = self.run_dir / "analytics_annotated.mp4"
        render_analytics_video(
            self.run_dir / "input" / "test_clip.mp4",
            analytics_video,
            tracks,
            identities,
            ball,
            possession,
            metrics,
            motion,
            calibration,
            self.config["render"]["layers"],
            display_id_by_track=display_id_by_track,
            team_by_track=team_by_track,
            frame_display_ids=frame_display_ids,
        )
        tactical = self.run_dir / "tactical_preview.mp4"
        render_tactical_preview(
            tactical,
            video_fps(self.run_dir),
            video_frame_count(self.run_dir),
            game_state,
            pitch_length_m=float(self.config["calibration"]["pitch_length_m"]),
            pitch_width_m=float(self.config["calibration"]["pitch_width_m"]),
            display_id_by_track=display_id_by_track,
        )

        chart = self.run_dir / "team_possession_chart.png"
        self._write_possession_chart(chart, possession)
        speed_csv = self.run_dir / "player_speed_summary.csv"
        self._write_speed_summary(speed_csv, metrics)
        return {
            "analytics_annotated": analytics_video,
            "tactical_preview": tactical,
            "team_possession_chart": chart,
            "player_speed_summary": speed_csv,
        }

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        for key, value in artifacts.items():
            path = Path(value)
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"render artifact missing/empty: {key}")
        for key in ("analytics_annotated", "tactical_preview"):
            capture = cv2.VideoCapture(str(artifacts[key]))
            ok, frame = capture.read()
            capture.release()
            if not ok or frame is None:
                raise RuntimeError(f"rendered video unreadable: {key}")
        pd.read_csv(artifacts["player_speed_summary"])

    @staticmethod
    def _write_possession_chart(path: Path, possession: pd.DataFrame) -> None:
        width, height = 900, 480
        image = np.full((height, width, 3), 248, dtype=np.uint8)
        states = (
            possession["possession_state"].fillna("unknown").value_counts(normalize=True)
            if not possession.empty
            else pd.Series({"unknown": 1.0})
        )
        labels = list(states.index)
        margin = 70
        bar_width = max(20, int((width - 2 * margin) / max(1, len(labels)) * 0.65))
        for index, label in enumerate(labels):
            value = float(states[label])
            x = margin + int((index + 0.5) * (width - 2 * margin) / len(labels))
            bar_h = int(value * (height - 150))
            cv2.rectangle(
                image,
                (x - bar_width // 2, height - 70 - bar_h),
                (x + bar_width // 2, height - 70),
                (180, 100, 45),
                -1,
            )
            cv2.putText(
                image,
                f"{100 * value:.1f}%",
                (x - 34, height - 80 - bar_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(label)[:18],
                (x - 60, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            image,
            "Possession state coverage",
            (margin, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write {path}")

    @staticmethod
    def _write_speed_summary(path: Path, metrics: pd.DataFrame) -> None:
        columns = [
            "track_id",
            "measurable_frames",
            "measurable_seconds",
            "total_distance_m",
            "average_moving_speed_kmh",
            "maximum_reliable_speed_kmh",
            "sprint_frames",
            "coverage",
        ]
        valid = metrics.loc[metrics["valid"].fillna(False)].copy()
        if valid.empty:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            return
        summary = (
            valid.groupby("track_id")
            .agg(
                measurable_frames=("frame_id", "count"),
                measurable_seconds=(
                    "timestamp_ms",
                    lambda s: float(max(0.0, (s.max() - s.min()) / 1000.0)),
                ),
                total_distance_m=("cumulative_distance_m", "max"),
                average_moving_speed_kmh=("smoothed_speed_kmh", "mean"),
                maximum_reliable_speed_kmh=("smoothed_speed_kmh", "max"),
                sprint_frames=("sprint_state", lambda s: int((s == "sprint").sum())),
                coverage=("coverage", "max"),
            )
            .reset_index()
        )
        summary[columns].to_csv(path, index=False)
