"""Opta-like player/team analytics aggregation stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from football_analytics.events.possession_chain import (
    PossessionChainConfig,
    build_possession_chain,
)
from football_analytics.opta.aggregate import (
    PLAYER_SUMMARY_COLUMNS,
    build_opta_player_summary,
    build_opta_team_summary,
    build_player_identity_table,
    export_heatmaps,
)
from football_analytics.opta.pitch_zones import PitchZones
from football_analytics.opta.speed_audit import audit_and_filter_speeds
from football_analytics.stages.base import Stage
from football_analytics.utils.io import write_json


class OptaAnalyticsStage(Stage):
    name = "opta_analytics"

    def validate_inputs(self) -> None:
        if not (self.run_dir / "tracks.parquet").is_file():
            raise FileNotFoundError("tracks.parquet")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        tracks = pd.read_parquet(self.run_dir / "tracks.parquet")
        identities = (
            pd.read_parquet(self.run_dir / "track_identities.parquet")
            if (self.run_dir / "track_identities.parquet").is_file()
            else pd.DataFrame()
        )
        global_map = (
            pd.read_parquet(self.run_dir / "global_identity_map.parquet")
            if (self.run_dir / "global_identity_map.parquet").is_file()
            else pd.DataFrame()
        )
        id_report = (
            pd.read_parquet(self.run_dir / "global_identity_report.parquet")
            if (self.run_dir / "global_identity_report.parquet").is_file()
            else pd.DataFrame()
        )
        identity_quality = {}
        iq_path = self.run_dir / "identity_quality.json"
        if iq_path.is_file():
            identity_quality = json.loads(iq_path.read_text(encoding="utf-8"))

        touches = (
            pd.read_parquet(self.run_dir / "touch_events.parquet")
            if (self.run_dir / "touch_events.parquet").is_file()
            else pd.DataFrame()
        )
        if "distance_to_ball" in touches.columns and "distance_m" not in touches.columns:
            touches = touches.rename(columns={"distance_to_ball": "distance_m"})
        passes = (
            pd.read_parquet(self.run_dir / "pass_events.parquet")
            if (self.run_dir / "pass_events.parquet").is_file()
            else pd.DataFrame()
        )
        dribbles = (
            pd.read_parquet(self.run_dir / "dribble_events.parquet")
            if (self.run_dir / "dribble_events.parquet").is_file()
            else pd.DataFrame()
        )
        duels = (
            pd.read_parquet(self.run_dir / "duel_events.parquet")
            if (self.run_dir / "duel_events.parquet").is_file()
            else pd.DataFrame()
        )
        defensive = (
            pd.read_parquet(self.run_dir / "defensive_actions.parquet")
            if (self.run_dir / "defensive_actions.parquet").is_file()
            else pd.DataFrame()
        )
        player_metrics = (
            pd.read_parquet(self.run_dir / "player_metrics.parquet")
            if (self.run_dir / "player_metrics.parquet").is_file()
            else pd.DataFrame()
        )
        speed_audits: list = []
        if not player_metrics.empty:
            player_metrics, speed_audits = audit_and_filter_speeds(player_metrics)
            if speed_audits:
                pd.DataFrame(speed_audits).to_csv(
                    self.run_dir / "speed_spike_audit.csv", index=False
                )
            player_metrics.to_parquet(
                self.run_dir / "player_metrics_filtered.parquet", index=False
            )

        from football_analytics.evaluation.calibration_audit import (
            write_calibration_audit_bundle,
        )
        from football_analytics.evaluation.publishability import compute_publishability
        from football_analytics.evaluation.ball_gt import ball_gt_complete
        from football_analytics.evaluation.identity_gt import identity_gt_complete
        from football_analytics.evaluation.touch_review import touch_review_complete

        cal_eval: dict = {}
        frame_cal_cov = measured_cov = cont_s = max_reproj = pos_cov = None
        legacy_row_valid = None
        if (self.run_dir / "calibration.parquet").is_file():
            cal_eval = write_calibration_audit_bundle(self.run_dir, apply_propagation=True)
            frame_cal_cov = cal_eval.get("calibration_coverage_after")
            measured_cov = cal_eval.get("measured_coverage")
            cont_s = cal_eval.get("continuous_calibrated_seconds")
            cal_df = pd.read_parquet(self.run_dir / "calibration.parquet")
            if "reprojection_error" in cal_df.columns:
                max_reproj = float(
                    pd.to_numeric(cal_df["reprojection_error"], errors="coerce").max()
                )
        if (self.run_dir / "game_state.parquet").is_file():
            gs = pd.read_parquet(self.run_dir / "game_state.parquet")
            if "valid" in gs.columns:
                pos_cov = float(gs["valid"].mean())
        if not player_metrics.empty and "valid" in player_metrics.columns:
            legacy_row_valid = float(player_metrics["valid"].mean())

        ball_gt_path = (
            Path(__file__).resolve().parents[3]
            / "configs/evaluation/short_clip_gt_template/football_ball/ball_gt.csv"
        )
        id_gt_path = (
            Path(__file__).resolve().parents[3]
            / "configs/evaluation/short_clip_gt_template/football_identity/identity_gt.csv"
        )
        touch_review_csv = self.run_dir / "evaluation" / "touch_review" / "touch_review.csv"
        ball_ok = False
        if ball_gt_path.is_file():
            ball_ok = ball_gt_complete(pd.read_csv(ball_gt_path))[0]
        id_ok = False
        if id_gt_path.is_file():
            id_ok = identity_gt_complete(pd.read_csv(id_gt_path))[0]
        touch_ok = touch_review_complete(touch_review_csv)[0] if touch_review_csv.is_file() else False

        pub_flags = compute_publishability(
            ball_gt_complete=ball_ok,
            identity_gt_complete=id_ok,
            touch_review_complete=touch_ok,
            calibration_coverage=frame_cal_cov,
            measured_calibration_coverage=measured_cov,
            player_position_coverage=pos_cov,
            continuous_calibrated_seconds=cont_s,
            max_reprojection_error=max_reproj,
            speed_spike_candidates=len(speed_audits),
            identity_quality=identity_quality,
        )
        write_json(self.run_dir / "publishability_flags.json", pub_flags.to_dict())
        write_json(
            self.run_dir / "physical_metrics_quality.json",
            {
                "calibration_coverage": frame_cal_cov,
                "calibration_coverage_measured": measured_cov,
                "calibration_coverage_propagated": cal_eval.get("propagated_coverage"),
                "player_position_coverage": pos_cov,
                "player_metrics_row_valid_rate": legacy_row_valid,
                "continuous_calibrated_seconds": cont_s,
                "speed_spike_candidates": len(speed_audits),
                "publish_speed_distance": bool(pub_flags.physical_metrics_publishable),
                "note": (
                    "player_metrics_row_valid_rate is NOT frame calibration coverage."
                ),
            },
        )

        chain = build_possession_chain(touches, config=PossessionChainConfig())
        sequences = chain.rename(
            columns={
                "chain_id": "possession_id",
                "start_ms": "start_time",
                "end_ms": "end_time",
                "from_global_player_id": "global_player_id",
            }
        )
        if not sequences.empty:
            sequences["touch_count"] = 2
            sequences["start_zone"] = None
            sequences["end_zone"] = None
            sequences["termination_reason"] = "pass"
            sequences["confidence"] = sequences.get("confidence", 0.5)
        sequences.to_parquet(self.run_dir / "possession_sequences.parquet", index=False)
        if not (self.run_dir / "possession_timeline.parquet").is_file():
            pd.DataFrame().to_parquet(self.run_dir / "possession_timeline.parquet", index=False)

        zones_path = self.run_dir / "pitch_zones.json"
        if zones_path.is_file():
            raw = json.loads(zones_path.read_text(encoding="utf-8"))
            zones = PitchZones(
                pitch_length_m=float(raw.get("pitch_length_m", 105)),
                pitch_width_m=float(raw.get("pitch_width_m", 68)),
                team0_attack_sign=int(raw.get("team0_attack_sign", 1)),
            )
        else:
            zones = PitchZones()
            write_json(zones_path, zones.to_dict())

        # Prefer resolved identity report (validated globals only for stats)
        if not id_report.empty:
            identity_table = id_report.rename(
                columns={"global_player_id": "global_player_id"}
            ).copy()
            identity_table["unresolved"] = ~identity_table["validated"].astype(bool)
            identity_table = identity_table[identity_table["validated"].astype(bool)].copy()
            identity_flags = list(identity_quality.get("identity_flags") or [])
        else:
            identity_table = build_player_identity_table(
                tracks,
                identities if not identities.empty else None,
                global_map if not global_map.empty else None,
            )
            identity_flags = list(getattr(identity_table, "attrs", {}).get("identity_flags", []))

        # Never publish Opta tables without GT-backed identity_publishable.
        stats_publishable = bool(pub_flags.action_stats_publishable)
        validated_by_team = identity_quality.get("validated_by_team") or {}
        if any(int(v) > 11 for v in validated_by_team.values()):
            stats_publishable = False
            identity_flags.append("STATS_BLOCKED_VALIDATED_GT_11")
        if not identity_quality and not id_report.empty:
            for team, g in identity_table.groupby("team_id"):
                if team is not None and len(g) > 11:
                    stats_publishable = False
                    identity_flags.append(f"INVALID_PLAYER_IDENTITY_COUNT:{team}:{len(g)}")

        heat = export_heatmaps(player_metrics, self.run_dir / "heatmaps")

        if stats_publishable and not identity_table.empty:
            player_summary = build_opta_player_summary(
                identity_table,
                passes=passes,
                dribbles=dribbles,
                duels=duels,
                defensive=defensive,
                touches=touches,
                player_metrics=player_metrics,
                zones=zones,
            )
            team_summary = build_opta_team_summary(player_summary)
        else:
            player_summary = pd.DataFrame(columns=PLAYER_SUMMARY_COLUMNS)
            team_summary = pd.DataFrame(
                columns=[
                    "team_id",
                    "validated_player_count",
                    "metric_coverage",
                    "total_distance_m",
                    "sprint_count",
                    "pass_attempts",
                    "passes_completed",
                    "pass_completion_pct",
                    "long_pass_attempts",
                    "long_passes_completed",
                    "long_pass_completion_pct",
                    "zone_1_to_2_passes",
                    "zone_2_to_3_passes",
                    "dribble_attempts",
                    "dribbles_completed",
                    "dribble_success_pct",
                    "duels",
                    "duels_won",
                    "duel_win_pct",
                    "aerial_duels",
                    "aerial_duels_won",
                    "tackles_won",
                    "interceptions",
                    "clearances",
                    "turnovers",
                    "penalty_area_touches",
                ]
            )

        player_summary.to_csv(self.run_dir / "player_opta_summary.csv", index=False)
        team_summary.to_csv(self.run_dir / "team_opta_summary.csv", index=False)
        # Diagnostic unfiltered copy for engineers (panel must not show when gated)
        if not id_report.empty:
            id_report.to_csv(self.run_dir / "global_identity_report.csv", index=False)

        publish_meta = {
            "stats_publishable": bool(pub_flags.action_stats_publishable),
            "overall_publishable": bool(pub_flags.overall_publishable),
            "identity_publishable": bool(pub_flags.identity_publishable),
            "ball_detection_publishable": bool(pub_flags.ball_detection_publishable),
            "ball_tracking_publishable": bool(pub_flags.ball_tracking_publishable),
            "touch_publishable": bool(pub_flags.touch_publishable),
            "calibration_publishable": bool(pub_flags.calibration_publishable),
            "physical_metrics_publishable": bool(pub_flags.physical_metrics_publishable),
            "tactical_metrics_publishable": bool(pub_flags.tactical_metrics_publishable),
            "action_stats_publishable": bool(pub_flags.action_stats_publishable),
            "reason": pub_flags.reasons.get("overall")
            or pub_flags.reasons.get("identity")
            or "quality_gates",
            "publishability_reasons": pub_flags.reasons,
            "gt_incomplete": pub_flags.gt_incomplete,
            "identity_flags": identity_flags,
            "validated_by_team": validated_by_team,
            "raw_count_by_team": identity_quality.get("raw_count_by_team"),
            "product": "opta_like_automatic_video_analytics",
            "warning": (
                "Model coverage yüksek ancak doğruluk ground truth ile doğrulanmadı."
                if pub_flags.gt_incomplete.get("ball") or pub_flags.gt_incomplete.get("identity")
                else None
            ),
        }
        # Keep legacy key false when identity GT incomplete or identity not publishable
        if not pub_flags.identity_publishable:
            publish_meta["stats_publishable"] = False
            if publish_meta.get("reason") in {None, "ok"}:
                publish_meta["reason"] = pub_flags.reasons.get("identity", "identity_not_publishable")
        write_json(self.run_dir / "opta_stats_publishable.json", publish_meta)

        if not (self.run_dir / "player_summary.csv").is_file():
            player_summary.to_csv(self.run_dir / "player_summary.csv", index=False)
        if not (self.run_dir / "team_summary.csv").is_file():
            team_summary.to_csv(self.run_dir / "team_summary.csv", index=False)

        metrics = {
            **publish_meta,
            "validated_players": int(len(identity_table)),
            "heatmap_players": heat.get("players", 0),
            "pass_confirmed": int((passes["status"] == "confirmed").sum())
            if not passes.empty and "status" in passes.columns
            else 0,
            "pass_candidate": int((passes["status"] == "candidate").sum())
            if not passes.empty and "status" in passes.columns
            else 0,
        }
        write_json(self.stage_dir / "metrics.json", metrics)
        (self.run_dir / "stage_manifests").mkdir(parents=True, exist_ok=True)
        write_json(
            self.run_dir / "stage_manifests" / "opta_analytics.json",
            {"stage": self.name, "status": "PASS", **metrics},
        )
        return {
            "player_opta_summary": self.run_dir / "player_opta_summary.csv",
            "team_opta_summary": self.run_dir / "team_opta_summary.csv",
            "opta_stats_publishable": self.run_dir / "opta_stats_publishable.json",
            "heatmaps": self.run_dir / "heatmaps",
            "metrics": self.stage_dir / "metrics.json",
        }

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        for name in ("player_opta_summary.csv", "team_opta_summary.csv", "opta_stats_publishable.json"):
            if not (self.run_dir / name).is_file():
                raise FileNotFoundError(name)
