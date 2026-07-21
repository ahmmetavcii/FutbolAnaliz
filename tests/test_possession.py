"""Tests for football_analytics.analytics.possession."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_analytics.analytics.possession import (  # noqa: E402
    BallSnapshot,
    PlayerSnapshot,
    PossessionConfig,
    PossessionState,
    PossessionTracker,
)


def _field_player(track_id: int, team: int, x: float, y: float) -> PlayerSnapshot:
    return PlayerSnapshot(track_id=track_id, team_id=team, x_field=x, y_field=y)


def _pixel_player(
    track_id: int, team: int, foot_x: float, foot_y: float, bbox_height: float
) -> PlayerSnapshot:
    return PlayerSnapshot(
        track_id=track_id, team_id=team, foot_x=foot_x, foot_y=foot_y, bbox_height=bbox_height
    )


class TestEmptyAndUnknown:
    def test_no_ball_ever_seen_is_unknown(self) -> None:
        tracker = PossessionTracker()
        result = tracker.step(0, 0.0, None, [_field_player(1, 0, 10.0, 10.0)])
        assert result.state is PossessionState.UNKNOWN
        assert result.owner_track_id is None

    def test_ball_but_no_players_is_loose(self) -> None:
        tracker = PossessionTracker()
        result = tracker.step(0, 0.0, BallSnapshot(x_field=50.0, y_field=34.0), [])
        assert result.state is PossessionState.LOOSE_BALL

    def test_ball_loss_times_out_to_unknown(self) -> None:
        config = PossessionConfig(unknown_timeout_ms=500.0, debounce_frames=1)
        tracker = PossessionTracker(config)
        players = [_field_player(7, 0, 50.0, 34.0)]
        result = tracker.step(0, 0.0, BallSnapshot(x_field=50.0, y_field=34.5), players)
        assert result.state is PossessionState.CONTROLLED_TEAM_0
        # Within the timeout the controlled state is retained.
        held = tracker.step(1, 300.0, None, players)
        assert held.state is PossessionState.CONTROLLED_TEAM_0
        # Beyond the timeout it degrades to unknown and the owner is cleared.
        lost = tracker.step(2, 700.0, None, players)
        assert lost.state is PossessionState.UNKNOWN
        assert lost.owner_track_id is None


class TestControlAndDebounce:
    def test_control_assigned_to_both_teams(self) -> None:
        tracker = PossessionTracker(PossessionConfig(debounce_frames=1))
        ball = BallSnapshot(x_field=50.0, y_field=34.0)
        r0 = tracker.step(0, 0.0, ball, [_field_player(1, 0, 50.5, 34.0)])
        assert r0.state is PossessionState.CONTROLLED_TEAM_0
        assert r0.owner_team_id == 0
        tracker.reset()
        r1 = tracker.step(0, 0.0, ball, [_field_player(2, 1, 50.5, 34.0)])
        assert r1.state is PossessionState.CONTROLLED_TEAM_1
        assert r1.owner_team_id == 1

    def test_debounce_delays_ownership_switch(self) -> None:
        config = PossessionConfig(debounce_frames=3, contest_ratio=1.0)
        tracker = PossessionTracker(config)
        holder = _field_player(1, 0, 50.0, 34.0)
        taker = _field_player(2, 1, 58.0, 34.0)
        near_holder = BallSnapshot(x_field=50.2, y_field=34.0)
        near_taker = BallSnapshot(x_field=57.8, y_field=34.0)
        assert (
            tracker.step(0, 0.0, near_holder, [holder, taker]).state
            is PossessionState.CONTROLLED_TEAM_0
        )
        # Ball moves next to the rival: two frames are not enough to flip.
        for frame in (1, 2):
            result = tracker.step(frame, frame * 40.0, near_taker, [holder, taker])
            assert result.state is PossessionState.CONTROLLED_TEAM_0
        # Third consecutive frame satisfies the debounce and control flips.
        flipped = tracker.step(3, 120.0, near_taker, [holder, taker])
        assert flipped.state is PossessionState.CONTROLLED_TEAM_1
        assert flipped.owner_track_id == 2

    def test_rival_proximity_yields_contested(self) -> None:
        tracker = PossessionTracker(PossessionConfig(contest_ratio=1.5))
        ball = BallSnapshot(x_field=50.0, y_field=34.0)
        players = [
            _field_player(1, 0, 50.0, 34.5),
            _field_player(2, 1, 50.0, 33.4),  # rival within 1.5x of closest
        ]
        result = tracker.step(0, 0.0, ball, players)
        assert result.state is PossessionState.CONTESTED
        assert result.owner_track_id is None

    def test_out_of_play(self) -> None:
        tracker = PossessionTracker()
        result = tracker.step(
            0, 0.0, BallSnapshot(x_field=-2.0, y_field=34.0, in_play=False), []
        )
        assert result.state is PossessionState.OUT_OF_PLAY

    def test_pass_in_flight_when_ball_fast_and_free(self) -> None:
        config = PossessionConfig(pass_speed_threshold=2.0)
        tracker = PossessionTracker(config)
        far_players = [_field_player(1, 0, 10.0, 10.0)]
        tracker.step(0, 0.0, BallSnapshot(x_field=40.0, y_field=34.0), far_players)
        # 10 m in 0.5 s => 20 m/s, no one within control range.
        result = tracker.step(1, 500.0, BallSnapshot(x_field=50.0, y_field=34.0), far_players)
        assert result.state is PossessionState.PASS_IN_FLIGHT


class TestDistanceNormalization:
    def test_pixel_distance_scales_with_bbox_height_not_fixed_pixels(self) -> None:
        """A 70 px gap means control for a large (near) player but not a
        small (far) player - the threshold must scale with apparent size."""
        config = PossessionConfig(control_radius_heights=0.9, debounce_frames=1)
        ball = BallSnapshot(x_pixel=500.0, y_pixel=500.0)

        tracker = PossessionTracker(config)
        near_player = [_pixel_player(1, 0, 570.0, 500.0, bbox_height=200.0)]  # 0.35 heights
        assert (
            tracker.step(0, 0.0, ball, near_player).state is PossessionState.CONTROLLED_TEAM_0
        )

        tracker = PossessionTracker(config)
        far_player = [_pixel_player(1, 0, 570.0, 500.0, bbox_height=40.0)]  # 1.75 heights
        assert tracker.step(0, 0.0, ball, far_player).state is PossessionState.LOOSE_BALL

    def test_field_coordinates_preferred_over_pixels(self) -> None:
        """When metric data exists it decides, even if pixel data disagrees."""
        config = PossessionConfig(control_radius_m=1.8, debounce_frames=1)
        tracker = PossessionTracker(config)
        ball = BallSnapshot(x_pixel=500.0, y_pixel=500.0, x_field=50.0, y_field=34.0)
        player = PlayerSnapshot(
            track_id=1,
            team_id=0,
            foot_x=501.0,  # pixel-close
            foot_y=500.0,
            bbox_height=100.0,
            x_field=80.0,  # metrically 30 m away
            y_field=34.0,
        )
        result = tracker.step(0, 0.0, ball, [player])
        assert result.state is PossessionState.LOOSE_BALL

    def test_player_without_bbox_height_excluded_in_pixel_space(self) -> None:
        tracker = PossessionTracker(PossessionConfig(debounce_frames=1))
        ball = BallSnapshot(x_pixel=500.0, y_pixel=500.0)
        player = PlayerSnapshot(track_id=1, team_id=0, foot_x=500.0, foot_y=501.0)
        result = tracker.step(0, 0.0, ball, [player])
        assert result.state is PossessionState.LOOSE_BALL


class TestReappearance:
    def test_ball_reappearance_restores_tracking(self) -> None:
        config = PossessionConfig(unknown_timeout_ms=200.0, debounce_frames=1)
        tracker = PossessionTracker(config)
        players = [_field_player(1, 0, 50.0, 34.0)]
        tracker.step(0, 0.0, BallSnapshot(x_field=50.0, y_field=34.5), players)
        for frame in range(1, 12):
            tracker.step(frame, frame * 40.0, None, players)
        assert tracker.step(11, 440.0, None, players).state is PossessionState.UNKNOWN
        back = tracker.step(12, 480.0, BallSnapshot(x_field=50.0, y_field=34.5), players)
        assert back.state is PossessionState.CONTROLLED_TEAM_0

    def test_unassigned_team_identity_is_unknown(self) -> None:
        tracker = PossessionTracker(PossessionConfig(debounce_frames=1))
        ball = BallSnapshot(x_field=50.0, y_field=34.0)
        referee = _field_player(99, -1, 50.2, 34.0)
        result = tracker.step(0, 0.0, ball, [referee])
        assert result.state is PossessionState.UNKNOWN
