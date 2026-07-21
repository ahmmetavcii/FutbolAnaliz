"""Tests for football_analytics.analytics.ball_trajectory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_analytics.analytics.ball_trajectory import (  # noqa: E402
    BallObservation,
    BallState,
    BallTrackerConfig,
    BallTrajectoryEstimator,
    Bounds,
)


def _obs(x: float, y: float, confidence: float = 0.9, size: float | None = 6.0) -> BallObservation:
    return BallObservation(x=x, y=y, confidence=confidence, width=size, height=size)


def _run_constant_motion(
    estimator: BallTrajectoryEstimator, fps: float, n_frames: int, speed_units_s: float = 100.0
) -> list:
    """Feed a ball moving at constant speed along +x, one detection per frame."""
    dt_ms = 1000.0 / fps
    results = []
    for i in range(n_frames):
        ts = i * dt_ms
        x = speed_units_s * ts / 1000.0
        results.append(estimator.step(i, ts, [_obs(x, 50.0)]))
    return results


class TestEmptyAndUnknown:
    def test_no_candidates_ever_yields_unknown_with_null_position(self) -> None:
        estimator = BallTrajectoryEstimator()
        for i in range(5):
            result = estimator.step(i, i * 40.0, [])
            assert result.state is BallState.UNKNOWN
            assert result.x is None and result.y is None
            assert result.confidence == 0.0

    def test_first_detection_is_detected_state(self) -> None:
        estimator = BallTrajectoryEstimator()
        result = estimator.step(0, 0.0, [_obs(10.0, 20.0)])
        assert result.state is BallState.DETECTED
        assert result.x == pytest.approx(10.0)
        assert result.y == pytest.approx(20.0)
        assert result.confidence == pytest.approx(0.9)


class TestGapPredictionAndLoss:
    def test_short_gap_is_occluded_then_predicted(self) -> None:
        config = BallTrackerConfig(max_gap_ms=500.0, max_gap_frames=12, short_occlusion_frames=2)
        estimator = BallTrajectoryEstimator(config)
        # Establish motion at 100 units/s along +x at 25 fps.
        for i in range(5):
            estimator.step(i, i * 40.0, [_obs(4.0 * i, 50.0)])
        occluded = estimator.step(5, 200.0, [])
        assert occluded.state is BallState.OCCLUDED_SHORT
        assert occluded.x == pytest.approx(20.0, abs=1.0)
        assert 0.0 < occluded.confidence < 0.9
        estimator.step(6, 240.0, [])
        predicted = estimator.step(7, 280.0, [])
        assert predicted.state is BallState.PREDICTED
        assert predicted.x == pytest.approx(28.0, abs=2.0)

    def test_long_loss_nulls_position_and_never_backfills(self) -> None:
        config = BallTrackerConfig(max_gap_ms=200.0, max_gap_frames=5)
        estimator = BallTrajectoryEstimator(config)
        estimator.step(0, 0.0, [_obs(10.0, 10.0)])
        estimator.step(1, 40.0, [_obs(14.0, 10.0)])
        outputs = [estimator.step(2 + k, 80.0 + k * 40.0, []) for k in range(10)]
        # Bounded prediction first, then null once the gap budget is exhausted.
        assert outputs[0].state in (BallState.OCCLUDED_SHORT, BallState.PREDICTED)
        tail = outputs[-1]
        assert tail.state is BallState.UNKNOWN
        assert tail.x is None and tail.y is None
        # Emitted outputs are immutable records: no bfill possible by design.
        assert all(o.x is None for o in outputs if o.state is BallState.UNKNOWN)

    def test_gap_bounded_by_frames_even_with_small_time_delta(self) -> None:
        config = BallTrackerConfig(max_gap_ms=10_000.0, max_gap_frames=3)
        estimator = BallTrajectoryEstimator(config)
        estimator.step(0, 0.0, [_obs(0.0, 0.0)])
        estimator.step(1, 10.0, [_obs(1.0, 0.0)])
        for k in range(2, 5):
            result = estimator.step(k, k * 10.0, [])
        result = estimator.step(6, 60.0, [])
        assert result.state is BallState.UNKNOWN

    def test_reappearance_after_long_loss_is_fresh_acquisition(self) -> None:
        config = BallTrackerConfig(max_gap_ms=100.0, max_gap_frames=2)
        estimator = BallTrajectoryEstimator(config)
        estimator.step(0, 0.0, [_obs(10.0, 10.0)])
        estimator.step(1, 40.0, [_obs(14.0, 10.0)])
        for k in range(2, 8):
            estimator.step(k, k * 40.0, [])
        # Reappears very far away: would fail any motion gate, but after a
        # long loss the estimator re-acquires fresh instead of forcing the
        # old identity/motion onto the new blob.
        result = estimator.step(8, 320.0, [_obs(900.0, 500.0)])
        assert result.state is BallState.DETECTED
        assert result.x == pytest.approx(900.0)
        assert result.velocity_x is None  # no stale velocity carried over


class TestGates:
    def test_low_confidence_rejected(self) -> None:
        estimator = BallTrajectoryEstimator(BallTrackerConfig(min_confidence=0.5))
        result = estimator.step(0, 0.0, [_obs(10.0, 10.0, confidence=0.2)])
        assert result.state is BallState.UNKNOWN

    def test_impossible_speed_rejected(self) -> None:
        config = BallTrackerConfig(max_speed=500.0)
        estimator = BallTrajectoryEstimator(config)
        estimator.step(0, 0.0, [_obs(0.0, 0.0)])
        # 5000 units in 40 ms => 125000 units/s, far beyond the gate.
        result = estimator.step(1, 40.0, [_obs(5000.0, 0.0)])
        assert result.state is not BallState.DETECTED
        assert result.x != pytest.approx(5000.0)

    def test_impossible_acceleration_rejected(self) -> None:
        config = BallTrackerConfig(max_speed=10_000.0, max_acceleration=1_000.0)
        estimator = BallTrajectoryEstimator(config)
        for i in range(4):
            estimator.step(i, i * 40.0, [_obs(2.0 * i, 0.0)])
        # Sudden reversal at huge speed: velocity delta implies absurd accel.
        result = estimator.step(4, 160.0, [_obs(-300.0, 0.0)])
        assert result.state is not BallState.DETECTED

    def test_size_gate(self) -> None:
        config = BallTrackerConfig(min_size=3.0, max_size=20.0)
        estimator = BallTrajectoryEstimator(config)
        assert estimator.step(0, 0.0, [_obs(5.0, 5.0, size=1.0)]).state is BallState.UNKNOWN
        assert estimator.step(1, 40.0, [_obs(5.0, 5.0, size=50.0)]).state is BallState.UNKNOWN
        assert estimator.step(2, 80.0, [_obs(5.0, 5.0, size=6.0)]).state is BallState.DETECTED

    def test_field_bounds_gate(self) -> None:
        bounds = Bounds(0.0, 0.0, 105.0, 68.0)
        estimator = BallTrajectoryEstimator(BallTrackerConfig(field_bounds=bounds))
        assert estimator.step(0, 0.0, [_obs(200.0, 30.0)]).state is BallState.UNKNOWN
        assert estimator.step(1, 40.0, [_obs(50.0, 30.0)]).state is BallState.DETECTED

    def test_nearest_admissible_candidate_wins(self) -> None:
        estimator = BallTrajectoryEstimator()
        estimator.step(0, 0.0, [_obs(10.0, 10.0)])
        result = estimator.step(1, 40.0, [_obs(60.0, 60.0, confidence=0.99), _obs(11.0, 10.0)])
        assert result.x == pytest.approx(11.0)


class TestSceneCutAndStates:
    def test_scene_cut_resets_motion_state(self) -> None:
        estimator = BallTrajectoryEstimator(BallTrackerConfig(max_speed=100.0))
        for i in range(4):
            estimator.step(i, i * 40.0, [_obs(2.0 * i, 10.0)])
        # After a cut, a far-away detection must be accepted fresh (the
        # motion gate must not compare against pre-cut positions).
        result = estimator.step(4, 160.0, [_obs(800.0, 400.0)], scene_cut=True)
        assert result.state is BallState.DETECTED
        assert result.x == pytest.approx(800.0)

    def test_airborne_from_size_jump(self) -> None:
        estimator = BallTrajectoryEstimator(BallTrackerConfig(airborne_size_ratio=1.5))
        for i in range(6):
            estimator.step(i, i * 40.0, [_obs(float(i), 10.0, size=6.0)])
        result = estimator.step(6, 240.0, [_obs(6.0, 10.0, size=12.0)])
        assert result.state is BallState.AIRBORNE

    def test_out_of_frame_prediction(self) -> None:
        config = BallTrackerConfig(
            frame_bounds=Bounds(0.0, 0.0, 100.0, 100.0),
            max_gap_ms=5000.0,
            max_gap_frames=100,
            short_occlusion_frames=0,
        )
        estimator = BallTrajectoryEstimator(config)
        estimator.step(0, 0.0, [_obs(90.0, 50.0)])
        estimator.step(1, 40.0, [_obs(98.0, 50.0)])  # 200 units/s toward edge
        result = estimator.step(2, 120.0, [])
        assert result.x is not None and result.x > 100.0
        assert result.state is BallState.OUT_OF_FRAME


class TestFrameRateAgnosticism:
    @pytest.mark.parametrize("fps", [24.0, 25.0, 30.0])
    def test_velocity_estimate_independent_of_fps(self, fps: float) -> None:
        estimator = BallTrajectoryEstimator()
        results = _run_constant_motion(estimator, fps=fps, n_frames=12, speed_units_s=100.0)
        final = results[-1]
        assert final.velocity_x == pytest.approx(100.0, rel=0.05)
        assert final.velocity_y == pytest.approx(0.0, abs=1.0)

    @pytest.mark.parametrize("fps", [24.0, 25.0, 30.0])
    def test_kalman_velocity_estimate_independent_of_fps(self, fps: float) -> None:
        estimator = BallTrajectoryEstimator(BallTrackerConfig(filter_type="kalman"))
        results = _run_constant_motion(estimator, fps=fps, n_frames=30, speed_units_s=100.0)
        final = results[-1]
        assert final.velocity_x == pytest.approx(100.0, rel=0.1)

    @pytest.mark.parametrize("fps", [24.0, 25.0, 30.0])
    def test_time_gap_budget_is_wall_clock_not_frames(self, fps: float) -> None:
        config = BallTrackerConfig(max_gap_ms=300.0, max_gap_frames=1000)
        estimator = BallTrajectoryEstimator(config)
        dt_ms = 1000.0 / fps
        estimator.step(0, 0.0, [_obs(0.0, 0.0)])
        estimator.step(1, dt_ms, [_obs(1.0, 0.0)])
        frame = 2
        ts = 2 * dt_ms
        last = None
        while ts - dt_ms <= dt_ms + 400.0:
            last = estimator.step(frame, ts, [])
            frame += 1
            ts += dt_ms
        assert last is not None
        assert last.state is BallState.UNKNOWN


def test_config_rejects_bad_filter_type() -> None:
    with pytest.raises(ValueError):
        BallTrackerConfig(filter_type="magic")
