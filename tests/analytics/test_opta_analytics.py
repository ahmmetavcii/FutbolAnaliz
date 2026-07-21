"""Opta-like analytics unit tests (evidence-based; no invented confirmed totals)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_analytics.analytics.player_metrics import (  # noqa: E402
    PlayerMetricsConfig,
    PlayerSample,
    compute_player_metrics,
)
from football_analytics.opta.actions import (  # noqa: E402
    ActionInferenceConfig,
    infer_clearances,
    infer_dribbles,
    infer_duels_from_proximity,
    infer_passes,
    infer_turnovers_and_interceptions,
)
from football_analytics.opta.aggregate import (  # noqa: E402
    ActivityIndexConfig,
    build_opta_player_summary,
    build_opta_team_summary,
    build_player_identity_table,
    compute_activity_index,
    export_heatmaps,
)
from football_analytics.opta.pitch_zones import PitchZones  # noqa: E402
from football_analytics.stages.action_inference import ActionInferenceStage  # noqa: E402
from football_analytics.stages.ball_tracking import BallTrackingStage  # noqa: E402
from football_analytics.stages.opta_analytics import OptaAnalyticsStage  # noqa: E402
from football_analytics.stages.touch_inference import TouchInferenceStage  # noqa: E402


def _touch(
    *,
    tid: int,
    team: str,
    ts: float,
    x: float,
    y: float,
    conf: float = 0.8,
    controlled: bool = True,
    gid=None,
    frame: int = 0,
) -> dict:
    return {
        "touch_id": f"t-{tid}-{ts}",
        "frame_id": frame,
        "timestamp_ms": ts,
        "global_player_id": gid if gid is not None else tid,
        "track_id": tid,
        "team_id": team,
        "confidence": conf,
        "controlled_touch": controlled,
        "deflection": False,
        "pitch_x": x,
        "pitch_y": y,
        "distance_m": 0.5,
    }


class TestPasses:
    def test_pass_successful(self):
        touches = pd.DataFrame(
            [
                _touch(tid=1, team="team_0", ts=1000, x=20, y=34),
                _touch(tid=2, team="team_0", ts=1500, x=35, y=34),
            ]
        )
        passes = infer_passes(touches, config=ActionInferenceConfig(confirmed_confidence=0.65))
        assert len(passes) == 1
        assert bool(passes.iloc[0]["successful"]) is True
        assert passes.iloc[0]["status"] == "confirmed"

    def test_pass_intercepted(self):
        touches = pd.DataFrame(
            [
                _touch(tid=1, team="team_0", ts=1000, x=40, y=34),
                _touch(tid=9, team="team_1", ts=1400, x=50, y=34),
            ]
        )
        passes = infer_passes(touches)
        assert len(passes) == 1
        assert bool(passes.iloc[0]["successful"]) is False
        defs = infer_turnovers_and_interceptions(passes)
        assert (defs["action_type"] == "interception").any()

    def test_long_pass(self):
        touches = pd.DataFrame(
            [
                _touch(tid=1, team="team_0", ts=1000, x=10, y=34),
                _touch(tid=2, team="team_0", ts=1800, x=40, y=34),
            ]
        )
        passes = infer_passes(touches, config=ActionInferenceConfig(long_pass_m=25.0))
        assert bool(passes.iloc[0]["long_pass"]) is True

    def test_zone_transition_pass(self):
        zones = PitchZones(team0_attack_sign=1)
        touches = pd.DataFrame(
            [
                _touch(tid=1, team="team_0", ts=1000, x=20, y=34),  # zone_1
                _touch(tid=2, team="team_0", ts=1600, x=45, y=34),  # zone_2
            ]
        )
        passes = infer_passes(touches, zones=zones)
        assert passes.iloc[0]["start_zone"] == "zone_1"
        assert passes.iloc[0]["end_zone"] == "zone_2"


class TestDribbles:
    def test_dribble_successful(self):
        touches = pd.DataFrame(
            [
                _touch(tid=1, team="team_0", ts=1000, x=30, y=34, conf=0.8),
                _touch(tid=1, team="team_0", ts=2000, x=38, y=34, conf=0.8),
            ]
        )
        dribbles = infer_dribbles(touches, config=ActionInferenceConfig(dribble_min_m=3.0))
        assert len(dribbles) == 1
        assert bool(dribbles.iloc[0]["successful"]) is True

    def test_dribble_failed_low_confidence_candidate(self):
        touches = pd.DataFrame(
            [
                _touch(tid=1, team="team_0", ts=1000, x=30, y=34, conf=0.5),
                _touch(tid=1, team="team_0", ts=2000, x=38, y=34, conf=0.5),
            ]
        )
        dribbles = infer_dribbles(touches)
        assert len(dribbles) == 1
        assert dribbles.iloc[0]["status"] in {"candidate", "unresolved"}


class TestDefensive:
    def test_tackle_won_and_interception(self):
        passes = infer_passes(
            pd.DataFrame(
                [
                    _touch(tid=1, team="team_0", ts=1000, x=40, y=34),
                    _touch(tid=5, team="team_1", ts=1300, x=42, y=34),
                ]
            )
        )
        defs = infer_turnovers_and_interceptions(passes)
        assert "interception" in set(defs["action_type"])
        assert "dispossession" in set(defs["action_type"])

    def test_turnover(self):
        # Same-team failed? Use opponent without receiver gap — interception path.
        # Explicit turnover when we craft a failed pass frame without receiver_track.
        frame = pd.DataFrame(
            [
                {
                    "pass_id": "p1",
                    "passer_global_id": 1,
                    "receiver_global_id": None,
                    "passer_track_id": 1,
                    "receiver_track_id": None,
                    "team_id": "team_0",
                    "start_time_ms": 0,
                    "end_time_ms": 500,
                    "start_x": 10,
                    "start_y": 10,
                    "end_x": 20,
                    "end_y": 10,
                    "start_zone": "zone_1",
                    "end_zone": "zone_2",
                    "distance_m": 10,
                    "forward_progress_m": 10,
                    "successful": False,
                    "long_pass": False,
                    "progressive_pass": False,
                    "confidence": 0.7,
                    "status": "confirmed",
                }
            ]
        )
        defs = infer_turnovers_and_interceptions(frame)
        assert (defs["action_type"] == "turnover").any()

    def test_clearance(self):
        touches = pd.DataFrame(
            [
                _touch(tid=1, team="team_0", ts=1000, x=8, y=34),
                _touch(tid=1, team="team_0", ts=1400, x=30, y=34),
            ]
        )
        clr = infer_clearances(touches, zones=PitchZones())
        assert len(clr) >= 1
        assert clr.iloc[0]["action_type"] == "clearance"


class TestDuels:
    def test_duel_winner_unresolved(self):
        tracks = pd.DataFrame(
            [
                {"frame_id": 1, "track_id": 1, "timestamp_ms": 1000, "x_field": 50.0, "y_field": 34.0},
                {"frame_id": 1, "track_id": 2, "timestamp_ms": 1000, "x_field": 51.0, "y_field": 34.0},
            ]
        )
        touches = pd.DataFrame([_touch(tid=1, team="team_0", ts=1000, x=50.5, y=34, frame=1)])
        identities = pd.DataFrame(
            [
                {"track_id": 1, "team_id": "team_0", "frame_id": 1},
                {"track_id": 2, "team_id": "team_1", "frame_id": 1},
            ]
        )
        duels = infer_duels_from_proximity(tracks, touches, identities=identities)
        assert len(duels) >= 1
        assert duels.iloc[0]["winner_global_id"] is None or duels.iloc[0]["status"] == "unresolved"

    def test_aerial_duel_candidate(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        touches = pd.DataFrame([_touch(tid=1, team="team_0", ts=1000, x=50, y=34, frame=10, conf=0.7)])
        touches.to_parquet(run_dir / "touch_events.parquet", index=False)
        pd.DataFrame(
            [{"frame_id": 10, "visibility_state": "airborne", "timestamp_ms": 1000}]
        ).to_parquet(run_dir / "ball_state.parquet", index=False)
        pd.DataFrame().to_parquet(run_dir / "game_state.parquet", index=False)
        stage = ActionInferenceStage(run_dir, {"actions": {"confirmed_confidence": 0.65}})
        stage.validate_inputs()
        stage.prepare()
        stage.run()
        duels = pd.read_parquet(run_dir / "duel_events.parquet")
        aerial = duels[duels["duel_type"] == "aerial_duel"]
        assert len(aerial) >= 1
        assert aerial.iloc[0]["status"] == "candidate"


class TestPenaltyAndIdentity:
    def test_penalty_area_touch(self):
        zones = PitchZones(team0_attack_sign=1)
        # team_0 attacking +x → opponent penalty near x=105
        assert zones.in_opponent_penalty(100.0, 34.0, team_id=0)
        touches = pd.DataFrame(
            [_touch(tid=1, team="team_0", ts=1000, x=100.0, y=34.0, gid=1)]
        )
        identities = pd.DataFrame(
            [
                {
                    "global_player_id": 1,
                    "team_id": "team_0",
                    "role": "outfield",
                    "visible_seconds": 30.0,
                    "identity_quality": "medium",
                    "unresolved": False,
                }
            ]
        )
        summary = build_opta_player_summary(
            identities,
            passes=pd.DataFrame(),
            dribbles=pd.DataFrame(),
            duels=pd.DataFrame(),
            defensive=pd.DataFrame(),
            touches=touches,
            player_metrics=None,
            zones=zones,
        )
        assert int(summary.iloc[0]["penalty_area_touches"]) >= 1

    def test_validated_player_count_max_11(self):
        rows = []
        for i in range(12):
            rows.append(
                {
                    "frame_id": 0,
                    "track_id": i + 1,
                    "object_type": "person",
                    "timestamp_ms": float(i * 100),
                }
            )
        tracks = pd.DataFrame(rows)
        identities = pd.DataFrame(
            [{"track_id": i + 1, "team_id": "team_0", "frame_id": 0} for i in range(12)]
        )
        table = build_player_identity_table(tracks, identities, None)
        flags = getattr(table, "attrs", {}).get("identity_flags", [])
        assert any("INVALID_PLAYER_IDENTITY_COUNT" in f for f in flags)

    def test_referee_excluded(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        tracks = pd.DataFrame(
            [
                {"frame_id": 0, "track_id": 1, "object_type": "person", "timestamp_ms": 0.0},
                {"frame_id": 0, "track_id": 2, "object_type": "person", "timestamp_ms": 0.0},
            ]
        )
        identities = pd.DataFrame(
            [
                {"track_id": 1, "team_id": "team_0", "role": "outfield", "frame_id": 0},
                {"track_id": 2, "team_id": None, "role": "referee", "frame_id": 0},
            ]
        )
        tracks.to_parquet(run_dir / "tracks.parquet", index=False)
        identities.to_parquet(run_dir / "track_identities.parquet", index=False)
        for name in (
            "touch_events",
            "pass_events",
            "dribble_events",
            "duel_events",
            "defensive_actions",
            "player_metrics",
        ):
            pd.DataFrame().to_parquet(run_dir / f"{name}.parquet", index=False)
        stage = OptaAnalyticsStage(run_dir, {})
        stage.run()
        players = pd.read_csv(run_dir / "player_opta_summary.csv")
        # Without identity_quality publish gate artifacts, summary may be empty;
        # referee must never appear when rows exist.
        if not players.empty and "role" in players.columns:
            assert not (players["role"].astype(str).str.lower() == "referee").any()
        # Stage-level identity path: referee excluded from fragments
        from football_analytics.opta.identity_resolve import build_track_fragments

        frags = build_track_fragments(tracks, identities, None, None)
        assert all(f.track_id != 2 for f in frags)


class TestPhysical:
    def test_distance_smoothing_and_spike_rejection(self):
        samples = [
            PlayerSample(1, i * 40.0, float(i) * 0.2, 30.0, confidence=0.9)
            for i in range(40)
        ]
        # Inject single-frame teleport spike
        samples[20] = PlayerSample(1, 20 * 40.0, 80.0, 30.0, confidence=0.9)
        result = compute_player_metrics(
            samples,
            PlayerMetricsConfig(
                max_speed_kmh=38.0,
                max_acceleration_ms2=10.0,
                accepted_shot_types=frozenset({"wide", "tactical", "main_wide"}),
            ),
        )[1]
        assert result.max_speed_kmh < 100.0

    def test_sprint_segmentation(self):
        # ~28 km/h = 7.78 m/s for 2 seconds
        fps = 25.0
        speed = 7.8
        samples = []
        for i in range(int(2.5 * fps)):
            t = i / fps
            samples.append(PlayerSample(1, t * 1000.0, speed * t, 30.0, confidence=0.95))
        result = compute_player_metrics(
            samples,
            PlayerMetricsConfig(
                sprint_speed_kmh=25.0,
                sprint_min_duration_s=0.5,
                accepted_shot_types=frozenset({"wide", "tactical", "main_wide"}),
            ),
        )[1]
        assert len(result.sprints) >= 1


class TestHeatmapActivityQuality:
    def test_heatmap_generation(self, tmp_path: Path):
        rows = []
        for i in range(30):
            rows.append(
                {
                    "track_id": 7,
                    "timestamp_ms": float(i * 100),
                    "x_field": 40.0 + i * 0.1,
                    "y_field": 30.0,
                    "valid": True,
                }
            )
        metrics = pd.DataFrame(rows)
        out = export_heatmaps(metrics, tmp_path / "heatmaps")
        assert out["players"] >= 1
        assert (tmp_path / "heatmaps" / "player_7_position.png").is_file()

    def test_activity_index_minimum_visibility(self):
        assert compute_activity_index(
            visible_seconds=2.0,
            distance_m=100,
            touches=10,
            passes=5,
            duels=2,
            defensive=1,
            penalty_touches=0,
            config=ActivityIndexConfig(min_visible_seconds=8.0),
        ) is None
        score = compute_activity_index(
            visible_seconds=60.0,
            distance_m=120,
            touches=10,
            passes=5,
            duels=2,
            defensive=1,
            penalty_touches=1,
        )
        assert score is not None
        assert 0 <= score <= 100

    def test_candidate_events_excluded_from_confirmed_totals(self):
        passes = pd.DataFrame(
            [
                {
                    "passer_global_id": 1,
                    "passer_track_id": 1,
                    "successful": True,
                    "long_pass": False,
                    "start_zone": "zone_1",
                    "end_zone": "zone_2",
                    "status": "candidate",
                    "confidence": 0.5,
                },
                {
                    "passer_global_id": 1,
                    "passer_track_id": 1,
                    "successful": True,
                    "long_pass": False,
                    "start_zone": "zone_1",
                    "end_zone": "zone_2",
                    "status": "confirmed",
                    "confidence": 0.8,
                },
            ]
        )
        identities = pd.DataFrame(
            [
                {
                    "global_player_id": 1,
                    "team_id": "team_0",
                    "role": "outfield",
                    "visible_seconds": 30.0,
                    "identity_quality": "medium",
                    "unresolved": False,
                }
            ]
        )
        summary = build_opta_player_summary(
            identities,
            passes=passes,
            dribbles=pd.DataFrame(),
            duels=pd.DataFrame(),
            defensive=pd.DataFrame(),
            touches=pd.DataFrame(),
            player_metrics=None,
        )
        assert int(summary.iloc[0]["pass_attempts"]) == 1

    def test_quality_warning_label(self):
        from football_analytics.panel.opta_labels import quality_label

        assert quality_label("activity_insufficient_visibility") == "Mevcut değil"
        assert quality_label("low_metric_coverage", 0.3) == "Düşük güven"


class TestStagesSmoke:
    def test_ball_touch_action_opta_chain(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Minimal ball_state
        ball_rows = []
        for i in range(20):
            ball_rows.append(
                {
                    "frame_id": i,
                    "timestamp_ms": float(i * 40),
                    "ball_x_pixel": 100.0 + i,
                    "ball_y_pixel": 200.0,
                    "ball_x_field": 30.0 + i * 0.5,
                    "ball_y_field": 34.0,
                    "detection_confidence": 0.9,
                    "visibility_state": "detected",
                }
            )
        pd.DataFrame(ball_rows).to_parquet(run_dir / "ball_state.parquet", index=False)
        track_rows = []
        for i in range(20):
            for tid, team, x in ((1, "team_0", 30.0), (2, "team_0", 40.0)):
                track_rows.append(
                    {
                        "frame_id": i,
                        "timestamp_ms": float(i * 40),
                        "track_id": tid,
                        "object_type": "person",
                        "foot_x_pixel": 100.0 + i if tid == 1 else 100.0 + i + 20,
                        "foot_y_pixel": 200.0,
                        "bbox_x1": 90,
                        "bbox_x2": 110,
                        "bbox_y1": 150,
                        "bbox_y2": 200,
                        "x_field": x + i * 0.5,
                        "y_field": 34.0,
                    }
                )
        pd.DataFrame(track_rows).to_parquet(run_dir / "tracks.parquet", index=False)
        pd.DataFrame(
            [
                {"frame_id": i, "track_id": 1, "team_id": "team_0", "role": "outfield"}
                for i in range(20)
            ]
            + [
                {"frame_id": i, "track_id": 2, "team_id": "team_0", "role": "outfield"}
                for i in range(20)
            ]
        ).to_parquet(run_dir / "track_identities.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "frame_id": i,
                    "track_id": tid,
                    "team_id": "team_0",
                    "x_field": 30.0 + i * 0.5,
                    "y_field": 34.0,
                    "valid": True,
                }
                for i in range(20)
                for tid in (1, 2)
            ]
        ).to_parquet(run_dir / "game_state.parquet", index=False)
        # player metrics for heatmap
        pd.DataFrame(
            [
                {
                    "track_id": 1,
                    "timestamp_ms": float(i * 40),
                    "x_field": 30.0 + i * 0.5,
                    "y_field": 34.0,
                    "valid": True,
                    "cumulative_distance_m": float(i),
                    "smoothed_speed_kmh": 10.0,
                    "sprint_state": "none",
                }
                for i in range(20)
            ]
        ).to_parquet(run_dir / "player_metrics.parquet", index=False)

        cfg = {"ball_trajectory": {"maximum_gap_frames": 5}, "actions": {}, "pipeline": {"resume": False}}
        BallTrackingStage(run_dir, cfg).execute(mode="force")
        TouchInferenceStage(run_dir, cfg).execute(mode="force")
        ActionInferenceStage(run_dir, cfg).execute(mode="force")
        OptaAnalyticsStage(run_dir, cfg).execute(mode="force")
        assert (run_dir / "ball_trajectory.parquet").is_file()
        assert (run_dir / "touch_events.parquet").is_file()
        assert (run_dir / "player_opta_summary.csv").is_file()
        assert (run_dir / "pitch_zones.json").is_file()


class TestClipSmokeMarkers:
    """Short clips are smoke/infrastructure only — not real accuracy proof."""

    FOOTBALL = Path("/mnt/c/football_data/videos/test_clips/football.mp4")
    INIESTA = Path("/mnt/c/football_data/videos/test_clips/iniesta_sample.mp4")

    def test_football_mp4_smoke(self, tmp_path: Path):
        assert self.FOOTBALL.is_file()
        # Infrastructure smoke: run opta stages on synthetic run tagged with clip name
        run_dir = tmp_path / "football_smoke"
        run_dir.mkdir()
        pd.DataFrame(
            [
                {
                    "frame_id": 0,
                    "timestamp_ms": 0.0,
                    "ball_x_pixel": 10.0,
                    "ball_y_pixel": 10.0,
                    "ball_x_field": 50.0,
                    "ball_y_field": 34.0,
                    "detection_confidence": 0.5,
                    "visibility_state": "detected",
                }
            ]
        ).to_parquet(run_dir / "ball_state.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "frame_id": 0,
                    "timestamp_ms": 0.0,
                    "track_id": 1,
                    "object_type": "person",
                    "foot_x_pixel": 12.0,
                    "foot_y_pixel": 12.0,
                    "bbox_x1": 0,
                    "bbox_x2": 20,
                    "bbox_y1": 0,
                    "bbox_y2": 20,
                }
            ]
        ).to_parquet(run_dir / "tracks.parquet", index=False)
        cfg = {"pipeline": {"resume": False}, "ball_trajectory": {}, "actions": {}}
        BallTrackingStage(run_dir, cfg).execute(mode="force")
        TouchInferenceStage(run_dir, cfg).execute(mode="force")
        assert (run_dir / "stage_manifests" / "ball_tracking.json").is_file()
        report = {
            "clip": str(self.FOOTBALL),
            "note": "15-30s smoke only; not accuracy evidence",
            "status": "SMOKE_PASS",
        }
        (run_dir / "smoke_report.json").write_text(str(report), encoding="utf-8")

    def test_iniesta_sample_mp4_smoke(self, tmp_path: Path):
        assert self.INIESTA.is_file()
        run_dir = tmp_path / "iniesta_smoke"
        run_dir.mkdir()
        pd.DataFrame(
            [
                {
                    "frame_id": 0,
                    "timestamp_ms": 0.0,
                    "ball_x_pixel": np.nan,
                    "ball_y_pixel": np.nan,
                    "ball_x_field": np.nan,
                    "ball_y_field": np.nan,
                    "detection_confidence": 0.0,
                    "visibility_state": "missing",
                }
            ]
        ).to_parquet(run_dir / "ball_state.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "frame_id": 0,
                    "timestamp_ms": 0.0,
                    "track_id": 1,
                    "object_type": "person",
                    "foot_x_pixel": 1.0,
                    "foot_y_pixel": 1.0,
                    "bbox_x1": 0,
                    "bbox_x2": 2,
                    "bbox_y1": 0,
                    "bbox_y2": 2,
                }
            ]
        ).to_parquet(run_dir / "tracks.parquet", index=False)
        cfg = {"pipeline": {"resume": False}}
        BallTrackingStage(run_dir, cfg).execute(mode="force")
        TouchInferenceStage(run_dir, cfg).execute(mode="force")
        touches = pd.read_parquet(run_dir / "touch_events.parquet")
        # Ball invisible → no invented touches
        assert len(touches) == 0


def test_production_config_excludes_echoes_trackeval():
    text = Path(
        "/home/ahmet/projects/football-analytics/configs/pipeline/opta_analytics.yaml"
    ).read_text(encoding="utf-8")
    assert "opta_analytics" in text
    assert "sn-echoes" not in text.lower() or "excludes" in text.lower()
    assert "trackeval" not in text.lower() or "excludes" in text.lower()
    assert "ball_tracking" in text
    assert "touch_inference" in text
    assert "action_inference" in text
