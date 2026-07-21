"""Tests for identity resolve, ball recovery, touch debug, GT eval, publish gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_analytics.opta.ball_recovery import BallRecoveryConfig, enhance_ball_state  # noqa: E402
from football_analytics.opta.ground_truth import (  # noqa: E402
    evaluate_against_ground_truth,
    write_ground_truth_template,
)
from football_analytics.opta.identity_resolve import (  # noqa: E402
    IdentityResolveConfig,
    TrackFragment,
    build_track_fragments,
    resolve_global_identities,
)
from football_analytics.opta.touch_debug import ankle_point_from_bbox  # noqa: E402
from football_analytics.stages.global_identity import GlobalIdentityStage  # noqa: E402
from football_analytics.stages.opta_analytics import OptaAnalyticsStage  # noqa: E402


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=16)
    return v / np.linalg.norm(v)


class TestIdentityResolve:
    def test_simultaneous_tracks_not_merged(self):
        frags = [
            TrackFragment(
                track_id=1,
                team_id="team_0",
                team_confidence=0.9,
                role="outfield",
                first_ms=0,
                last_ms=2000,
                frame_count=20,
                visible_seconds=2.0,
                embedding=_emb(1),
                start_xy=(10, 10),
                end_xy=(12, 10),
                mean_xy=(11, 10),
            ),
            TrackFragment(
                track_id=2,
                team_id="team_0",
                team_confidence=0.9,
                role="outfield",
                first_ms=500,
                last_ms=2500,
                frame_count=20,
                visible_seconds=2.0,
                embedding=_emb(1),
                start_xy=(20, 20),
                end_xy=(22, 20),
                mean_xy=(21, 20),
            ),
        ]
        gmap, report, metrics, _dec = resolve_global_identities(frags)
        assert gmap["global_id"].nunique() == 2
        assert metrics["false_merge_guards_simultaneous"] >= 1

    def test_different_team_not_merged(self):
        emb = _emb(7)
        frags = [
            TrackFragment(
                1, "team_0", 0.9, "outfield", 0, 1000, 15, 1.0, emb, (10, 10), (11, 10), (10.5, 10)
            ),
            TrackFragment(
                2, "team_1", 0.9, "outfield", 2000, 3000, 15, 1.0, emb, (12, 10), (13, 10), (12.5, 10)
            ),
        ]
        gmap, _, metrics, _dec = resolve_global_identities(frags)
        assert gmap["global_id"].nunique() == 2
        assert metrics["false_merge_guards_team"] >= 1

    def test_short_gap_same_player_merged(self):
        emb = _emb(3)
        frags = [
            TrackFragment(
                1, "team_0", 0.9, "outfield", 0, 1000, 20, 1.0, emb, (10, 10), (15, 10), (12, 10)
            ),
            TrackFragment(
                9, "team_0", 0.9, "outfield", 1300, 2500, 20, 1.2, emb, (16, 10), (22, 10), (19, 10)
            ),
        ]
        gmap, report, _metrics, _dec = resolve_global_identities(
            frags, config=IdentityResolveConfig(reid_merge_threshold=0.5)
        )
        assert gmap["global_id"].nunique() == 1
        assert int(report.iloc[0]["track_fragment_count"]) == 2

    def test_on_field_surplus_demoted_without_false_merge(self):
        emb = [_emb(200 + i) for i in range(13)]
        frags = []
        for i in range(13):
            frags.append(
                TrackFragment(
                    track_id=i + 1,
                    team_id="team_0",
                    team_confidence=0.9,
                    role="outfield",
                    first_ms=0.0,
                    last_ms=2000.0 + i,  # all overlap
                    frame_count=30,
                    visible_seconds=2.0 + i * 0.01,
                    embedding=emb[i],
                    start_xy=(float(i) * 5.0, 20.0),
                    end_xy=(float(i) * 5.0 + 1.0, 20.0),
                    mean_xy=(float(i) * 5.0 + 0.5, 20.0),
                )
            )
        gmap, report, metrics, _ = resolve_global_identities(
            frags, config=IdentityResolveConfig(enforce_max_on_field=True)
        )
        assert gmap["global_id"].nunique() == 13  # no false merges
        assert metrics["validated_by_team"]["team_0"] <= 11
        assert metrics["stats_publishable"] is True
        assert metrics["reid_status"] == "SOLVED"
        assert int((report["identity_quality"] == "on_field_surplus").sum()) >= 2

    def test_validated_over_11_blocks_publish_without_hard_cap(self):
        frags = []
        for i in range(14):
            frags.append(
                TrackFragment(
                    track_id=i + 1,
                    team_id="team_0",
                    team_confidence=0.9,
                    role="outfield",
                    first_ms=float(i * 10_000),
                    last_ms=float(i * 10_000 + 2000),
                    frame_count=20,
                    visible_seconds=2.0 + i * 0.01,
                    embedding=_emb(100 + i),
                    # Far apart on pitch so position stitching cannot collapse them.
                    start_xy=(float(i) * 25.0, 10.0),
                    end_xy=(float(i) * 25.0 + 1.0, 10.0),
                    mean_xy=(float(i) * 25.0 + 0.5, 10.0),
                )
            )
        _gmap, _report, metrics, _dec = resolve_global_identities(
            frags,
            config=IdentityResolveConfig(
                allow_hard_cap_demotion=False,
                enforce_max_on_field=False,
            ),
        )
        # No demotion: validated can exceed 11; publish blocked
        assert metrics["validated_by_team"]["team_0"] > 11
        assert metrics["stats_publishable"] is False
        assert metrics["hard_cap_demotion_enabled"] is False
        assert any("INVALID_PLAYER_IDENTITY_COUNT" in f for f in metrics["identity_flags"])

    def test_referee_excluded_from_fragments(self):
        tracks = pd.DataFrame(
            [
                {
                    "frame_id": i,
                    "track_id": 1,
                    "object_type": "person",
                    "timestamp_ms": float(i * 40),
                }
                for i in range(20)
            ]
            + [
                {
                    "frame_id": i,
                    "track_id": 2,
                    "object_type": "person",
                    "timestamp_ms": float(i * 40),
                }
                for i in range(20)
            ]
        )
        identities = pd.DataFrame(
            [
                {
                    "track_id": 1,
                    "team_id": "team_0",
                    "team_confidence": 0.9,
                    "role": "outfield",
                    "frame_id": 0,
                }
            ]
            + [
                {
                    "track_id": 2,
                    "team_id": None,
                    "team_confidence": 0.0,
                    "role": "referee",
                    "frame_id": 0,
                }
            ]
        )
        frags = build_track_fragments(tracks, identities, None, None)
        assert all(f.track_id != 2 for f in frags)


class TestBallRecovery:
    def test_coverage_report_fields(self):
        rows = []
        for i in range(30):
            detected = i % 5 == 0
            rows.append(
                {
                    "frame_id": i,
                    "timestamp_ms": float(i * 40),
                    "ball_x_pixel": 100.0 + i if detected else np.nan,
                    "ball_y_pixel": 50.0 if detected else np.nan,
                    "ball_x_field": np.nan,
                    "ball_y_field": np.nan,
                    "visibility_state": "detected" if detected else "unknown",
                    "detection_confidence": 0.8 if detected else 0.0,
                    "valid": detected,
                }
            )
        _frame, cov = enhance_ball_state(
            pd.DataFrame(rows),
            video_path=None,
            config=BallRecoveryConfig(enable_roi_search=False, enable_optical_flow=False),
        )
        assert "raw_detection_coverage" in cov
        assert "tracked_coverage" in cov
        assert "interpolated_coverage" in cov
        assert cov["tracked_coverage"] >= cov["raw_detection_coverage"]
        assert "false_positive_candidates" in cov


class TestTouchAndGT:
    def test_ankle_point(self):
        x, y = ankle_point_from_bbox(0, 0, 100, 200)
        assert 40 < x < 60
        assert y > 150

    def test_gt_template_and_empty_eval(self, tmp_path: Path):
        paths = write_ground_truth_template(tmp_path / "gt")
        assert Path(paths["readme"]).is_file()
        result = evaluate_against_ground_truth(tmp_path / "run", tmp_path / "gt")
        assert result["unit_tests_are_not_accuracy"] is True
        assert result["touch"]["precision"] is None


class TestPublishGate:
    def test_opta_stats_hidden_when_not_publishable(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        pd.DataFrame(
            [
                {
                    "frame_id": 0,
                    "track_id": i,
                    "object_type": "person",
                    "timestamp_ms": 0.0,
                }
                for i in range(1, 5)
            ]
        ).to_parquet(run_dir / "tracks.parquet", index=False)
        (run_dir / "identity_quality.json").write_text(
            json.dumps(
                {
                    "stats_publishable": False,
                    "validated_by_team": {"team_0": 11},
                    "raw_count_by_team": {"team_0": 20},
                    "identity_flags": ["INVALID_PLAYER_IDENTITY_COUNT:team_0:20"],
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "global_player_id": i,
                    "local_track_ids": str(i),
                    "team_id": "team_0",
                    "role": "outfield",
                    "time_intervals_ms": "0-1000",
                    "simultaneous_conflicts": 0,
                    "reid_cosine_mean": None,
                    "team_colour_similarity_mean": 1.0,
                    "pitch_continuity_mean_m": None,
                    "visible_seconds": 2.0,
                    "track_fragment_count": 1,
                    "merge_reasons": "new",
                    "split_reasons": "",
                    "validated": True,
                    "identity_quality": "medium",
                }
                for i in range(1, 12)
            ]
        ).to_parquet(run_dir / "global_identity_report.parquet", index=False)
        for name in (
            "touch_events",
            "pass_events",
            "dribble_events",
            "duel_events",
            "defensive_actions",
            "player_metrics",
        ):
            pd.DataFrame().to_parquet(run_dir / f"{name}.parquet", index=False)
        stage = OptaAnalyticsStage(run_dir, {"pipeline": {"resume": False}})
        stage.execute(mode="force")
        meta = json.loads((run_dir / "opta_stats_publishable.json").read_text())
        assert meta["stats_publishable"] is False
        players = pd.read_csv(run_dir / "player_opta_summary.csv")
        assert players.empty

    def test_panel_source_hides_stats(self):
        source = Path(
            "/home/ahmet/projects/football-analytics/apps/full_match_panel.py"
        ).read_text(encoding="utf-8")
        assert "opta_stats_publishable.json" in source
        assert "İstatistikler yayınlanmadı" in source


def test_global_identity_stage_writes_map(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tracks = []
    for tid in (1, 2):
        for i in range(40):
            tracks.append(
                {
                    "frame_id": i + (0 if tid == 1 else 50),
                    "track_id": tid,
                    "object_type": "person",
                    "timestamp_ms": float((i + (0 if tid == 1 else 50)) * 40),
                    "bbox_x1": 10.0 + tid,
                    "bbox_y1": 10.0,
                    "bbox_x2": 40.0 + tid,
                    "bbox_y2": 80.0,
                }
            )
    pd.DataFrame(tracks).to_parquet(run_dir / "tracks.parquet", index=False)
    pd.DataFrame(
        [
            {
                "frame_id": 0,
                "track_id": 1,
                "team_id": "team_0",
                "team_confidence": 0.9,
                "role": "outfield",
            },
            {
                "frame_id": 50,
                "track_id": 2,
                "team_id": "team_0",
                "team_confidence": 0.9,
                "role": "outfield",
            },
        ]
    ).to_parquet(run_dir / "track_identities.parquet", index=False)
    emb = _emb(42).tolist()
    pd.DataFrame(
        [
            {"track_id": 1, "embedding": emb, "valid": True},
            {"track_id": 2, "embedding": emb, "valid": True},
        ]
    ).to_parquet(run_dir / "track_reid_prototypes.parquet", index=False)
    GlobalIdentityStage(
        run_dir, {"pipeline": {"resume": False}, "global_identity": {}}
    ).execute(mode="force")
    gmap = pd.read_parquet(run_dir / "global_identity_map.parquet")
    assert not gmap.empty
    quality = json.loads((run_dir / "identity_quality.json").read_text())
    assert "stats_publishable" in quality
    assert quality["validated_by_team"].get("team_0", 0) <= 11
