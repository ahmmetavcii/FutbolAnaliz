"""Tests for football_analytics.analytics.player_metrics and team_metrics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_analytics.analytics.player_metrics import (  # noqa: E402
    PlayerMetricsConfig,
    PlayerSample,
    compute_player_metrics,
)
from football_analytics.analytics.team_metrics import (  # noqa: E402
    TeamMetricsConfig,
    TeamPlayerSample,
    compute_team_metrics,
)


def _walk_samples(
    track_id: int,
    fps: float,
    duration_s: float,
    speed_ms: float,
    start_ts_ms: float = 0.0,
    confidence: float = 0.9,
) -> list[PlayerSample]:
    """Straight-line run at constant speed sampled at the given frame rate."""
    n = int(round(duration_s * fps)) + 1
    samples = []
    for i in range(n):
        t_s = i / fps
        samples.append(
            PlayerSample(
                track_id=track_id,
                timestamp_ms=start_ts_ms + t_s * 1000.0,
                x_field=speed_ms * t_s,
                y_field=30.0,
                confidence=confidence,
            )
        )
    return samples


class TestEmptyAndGates:
    def test_empty_input(self) -> None:
        assert compute_player_metrics([]) == {}

    def test_too_few_samples_invalid(self) -> None:
        samples = [PlayerSample(1, 0.0, 10.0, 10.0), PlayerSample(1, 40.0, 10.1, 10.0)]
        result = compute_player_metrics(samples)[1]
        assert not result.valid
        assert result.invalid_reason == "insufficient_usable_samples"
        assert result.total_distance_m == 0.0

    def test_calibration_gate_excludes_samples(self) -> None:
        samples = _walk_samples(1, fps=25.0, duration_s=4.0, speed_ms=5.0)
        gated = [
            PlayerSample(
                s.track_id, s.timestamp_ms, s.x_field, s.y_field, s.confidence,
                calibration_valid=False, shot_type=s.shot_type,
            )
            for s in samples
        ]
        result = compute_player_metrics(gated)[1]
        assert not result.valid
        assert result.used_sample_count == 0

    def test_shot_type_gate(self) -> None:
        samples = [
            PlayerSample(1, i * 40.0, float(i), 30.0, shot_type="closeup") for i in range(50)
        ]
        result = compute_player_metrics(samples)[1]
        assert not result.valid

    def test_quality_gate(self) -> None:
        samples = [
            PlayerSample(1, i * 40.0, float(i), 30.0, quality_ok=False) for i in range(50)
        ]
        result = compute_player_metrics(samples)[1]
        assert not result.valid

    def test_low_confidence_samples_excluded(self) -> None:
        config = PlayerMetricsConfig(min_confidence=0.5)
        samples = _walk_samples(1, fps=25.0, duration_s=4.0, speed_ms=5.0, confidence=0.1)
        result = compute_player_metrics(samples, config)[1]
        assert not result.valid

    def test_missing_coordinates_excluded(self) -> None:
        samples = [PlayerSample(1, i * 40.0, None, None) for i in range(50)]
        result = compute_player_metrics(samples)[1]
        assert not result.valid


class TestFrameRateAgnosticism:
    @pytest.mark.parametrize("fps", [24.0, 25.0, 30.0])
    def test_distance_and_speed_invariant_across_fps(self, fps: float) -> None:
        # 6 m/s for 10 s => 60 m and 21.6 km/h regardless of sampling rate.
        samples = _walk_samples(1, fps=fps, duration_s=10.0, speed_ms=6.0)
        result = compute_player_metrics(samples)[1]
        assert result.valid
        assert result.total_distance_m == pytest.approx(60.0, rel=0.02)
        assert result.max_speed_kmh == pytest.approx(21.6, rel=0.02)
        assert result.mean_speed_kmh == pytest.approx(21.6, rel=0.02)

    def test_mixed_frame_rate_timestamps(self) -> None:
        """Metrics come from timestamps, so an FPS change mid-track is fine."""
        first = _walk_samples(1, fps=25.0, duration_s=4.0, speed_ms=5.0)
        last_ts = first[-1].timestamp_ms
        last_x = first[-1].x_field
        assert last_x is not None
        second = [
            PlayerSample(1, last_ts + (i + 1) * (1000.0 / 30.0), last_x + 5.0 * (i + 1) / 30.0, 30.0)
            for i in range(120)
        ]
        result = compute_player_metrics(first + second)[1]
        assert result.valid
        assert result.total_distance_m == pytest.approx(40.0, rel=0.02)
        assert result.max_speed_kmh == pytest.approx(18.0, rel=0.02)


class TestPlausibilityRejection:
    def test_impossible_speed_spike_rejected(self) -> None:
        samples = _walk_samples(1, fps=25.0, duration_s=4.0, speed_ms=5.0)
        # Teleport 60 m in one 40 ms step: 5400 km/h.
        spike = PlayerSample(1, samples[50].timestamp_ms + 1.0, 999.0, 30.0)
        result = compute_player_metrics(samples + [spike])[1]
        clean = compute_player_metrics(samples)[1]
        assert result.valid
        assert result.max_speed_kmh <= PlayerMetricsConfig().max_speed_kmh
        assert result.total_distance_m == pytest.approx(clean.total_distance_m, rel=0.02)

    def test_impossible_acceleration_rejected(self) -> None:
        config = PlayerMetricsConfig(max_acceleration_ms2=6.0, smoothing_window_s=0.04)
        samples = _walk_samples(1, fps=25.0, duration_s=4.0, speed_ms=2.0)
        # Insert a burst: jumps from 2 m/s to ~10 m/s within one frame
        # (200 m/s^2), physically impossible for a human.
        burst_ts = samples[-1].timestamp_ms + 40.0
        last_x = samples[-1].x_field
        assert last_x is not None
        samples.append(PlayerSample(1, burst_ts, last_x + 0.4, 30.0))
        result = compute_player_metrics(samples, config)[1]
        assert result.valid
        assert result.max_speed_kmh < 10.0

    def test_long_gap_breaks_segment_without_teleport_distance(self) -> None:
        """Track lost then reappears far away: no distance credited across
        the gap, and metrics resume from the reappearance point."""
        config = PlayerMetricsConfig(max_segment_gap_ms=1000.0)
        before = _walk_samples(1, fps=25.0, duration_s=2.0, speed_ms=5.0)
        after = [
            PlayerSample(1, 10_000.0 + i * 40.0, 80.0 + 5.0 * i * 0.04, 30.0) for i in range(50)
        ]
        result = compute_player_metrics(before + after, config)[1]
        assert result.valid
        # ~10 m before the loss + ~9.8 m after; the 70 m jump is not counted.
        assert result.total_distance_m == pytest.approx(19.8, rel=0.05)
        assert result.max_speed_kmh < 20.0
        # Coverage reflects the unobserved hole in the span.
        assert result.coverage == pytest.approx(1.0)

    def test_no_id_stitching_between_tracks(self) -> None:
        """Two track ids for the same person stay separate summaries."""
        first = _walk_samples(1, fps=25.0, duration_s=2.0, speed_ms=5.0)
        second = _walk_samples(2, fps=25.0, duration_s=2.0, speed_ms=5.0, start_ts_ms=3000.0)
        results = compute_player_metrics(first + second)
        assert set(results) == {1, 2}
        assert results[1].total_distance_m == pytest.approx(10.0, rel=0.02)
        assert results[2].total_distance_m == pytest.approx(10.0, rel=0.02)


class TestSprintsCoverageConfidence:
    def test_sprint_detected(self) -> None:
        config = PlayerMetricsConfig(sprint_speed_kmh=25.0, sprint_min_duration_s=1.0)
        jog = _walk_samples(1, fps=25.0, duration_s=3.0, speed_ms=2.0)
        last = jog[-1]
        assert last.x_field is not None
        sprint = [
            PlayerSample(
                1,
                last.timestamp_ms + (i + 1) * 40.0,
                last.x_field + 8.0 * (i + 1) * 0.04,
                30.0,
            )
            for i in range(75)  # 3 s at 8 m/s = 28.8 km/h
        ]
        result = compute_player_metrics(jog + sprint, config)[1]
        assert result.valid
        assert len(result.sprints) == 1
        episode = result.sprints[0]
        assert episode.duration_s >= 1.0
        assert episode.peak_speed_kmh == pytest.approx(28.8, rel=0.05)

    def test_short_burst_not_a_sprint(self) -> None:
        config = PlayerMetricsConfig(
            sprint_speed_kmh=25.0, sprint_min_duration_s=2.0, smoothing_window_s=0.04
        )
        jog = _walk_samples(1, fps=25.0, duration_s=3.0, speed_ms=2.0)
        last = jog[-1]
        assert last.x_field is not None
        burst = [
            PlayerSample(1, last.timestamp_ms + (i + 1) * 40.0, last.x_field + 8.0 * (i + 1) * 0.04, 30.0)
            for i in range(10)  # only 0.4 s above threshold
        ]
        result = compute_player_metrics(jog + burst, config)[1]
        assert result.valid
        assert result.sprints == ()

    def test_coverage_and_confidence_reported(self) -> None:
        good = _walk_samples(1, fps=25.0, duration_s=4.0, speed_ms=3.0, confidence=0.8)
        # Append a gated-out tail: widens the raw span but not usable span.
        tail = [
            PlayerSample(1, 4000.0 + i * 40.0, None, None, confidence=0.9) for i in range(100)
        ]
        result = compute_player_metrics(good + tail)[1]
        assert result.valid
        assert result.mean_confidence == pytest.approx(0.8)
        assert 0.0 < result.coverage < 0.6


class TestTeamMetrics:
    def test_only_valid_coordinates_and_confident_identities_used(self) -> None:
        config = TeamMetricsConfig(min_team_confidence=0.7, min_players=2)
        players = [
            TeamPlayerSample(1, 0, 10.0, 10.0, team_confidence=0.9),
            TeamPlayerSample(2, 0, 20.0, 20.0, team_confidence=0.9),
            TeamPlayerSample(3, 0, None, None, team_confidence=0.9),  # no coords
            TeamPlayerSample(4, 0, 90.0, 60.0, team_confidence=0.3),  # unsure identity
            TeamPlayerSample(5, 0, math.nan, 10.0, team_confidence=0.9),  # non-finite
            TeamPlayerSample(6, 0, 50.0, 40.0, team_confidence=0.9, calibration_valid=False),
        ]
        result = compute_team_metrics(players, config)[0]
        assert result.valid
        assert result.player_count == 2
        assert result.centroid_x == pytest.approx(15.0)
        assert result.centroid_y == pytest.approx(15.0)
        assert result.mean_interplayer_distance_m == pytest.approx(math.hypot(10.0, 10.0))

    def test_insufficient_confident_players_invalid(self) -> None:
        config = TeamMetricsConfig(min_players=4)
        players = [
            TeamPlayerSample(1, 1, 10.0, 10.0, team_confidence=0.9),
            TeamPlayerSample(2, 1, 20.0, 20.0, team_confidence=0.9),
        ]
        result = compute_team_metrics(players, config)[1]
        assert not result.valid
        assert result.invalid_reason == "insufficient_confident_players"
        assert result.centroid_x is None

    def test_empty_input(self) -> None:
        assert compute_team_metrics([]) == {}

    def test_width_depth_shape(self) -> None:
        config = TeamMetricsConfig(min_players=4)
        players = [
            TeamPlayerSample(i, 0, x, y, team_confidence=1.0)
            for i, (x, y) in enumerate([(30.0, 10.0), (30.0, 58.0), (50.0, 20.0), (55.0, 40.0)])
        ]
        result = compute_team_metrics(players, config)[0]
        assert result.valid
        assert result.width_m == pytest.approx(48.0)
        assert result.depth_m == pytest.approx(25.0)
