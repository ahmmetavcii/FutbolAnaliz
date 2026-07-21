"""Possession state machine driven by ball-player proximity.

States: ``controlled_team_0``, ``controlled_team_1``, ``contested``,
``pass_in_flight``, ``loose_ball``, ``out_of_play``, ``unknown``.

Distance policy:

- When both ball and players have valid field coordinates (metres), control is
  decided on metric distance (``control_radius_m``).
- Otherwise the pixel foot distance is normalized by the candidate player's
  bounding-box height, so the threshold scales with apparent player size
  instead of using a fixed pixel radius (``control_radius_heights`` in
  "player heights"). Players without a usable bbox height are excluded from
  pixel-space contention rather than judged with an arbitrary constant.

Control changes are debounced (a challenger must win ``debounce_frames``
consecutive frames), a nearby rival within the contest ratio yields
``contested``, and losing ball information for longer than
``unknown_timeout_ms`` degrades the state to ``unknown``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class PossessionState(str, Enum):
    CONTROLLED_TEAM_0 = "controlled_team_0"
    CONTROLLED_TEAM_1 = "controlled_team_1"
    CONTESTED = "contested"
    PASS_IN_FLIGHT = "pass_in_flight"
    LOOSE_BALL = "loose_ball"
    OUT_OF_PLAY = "out_of_play"
    UNKNOWN = "unknown"


_CONTROL_STATES = {
    0: PossessionState.CONTROLLED_TEAM_0,
    1: PossessionState.CONTROLLED_TEAM_1,
}


@dataclass(frozen=True)
class PlayerSnapshot:
    """One player's position for a frame.

    ``foot_x`` / ``foot_y`` are pixel foot coordinates; ``x_field`` /
    ``y_field`` are metres when calibration is available. ``bbox_height`` is
    the pixel bbox height used to normalize pixel distances.
    """

    track_id: int
    team_id: int
    foot_x: float | None = None
    foot_y: float | None = None
    bbox_height: float | None = None
    x_field: float | None = None
    y_field: float | None = None

    @property
    def has_field_position(self) -> bool:
        return self.x_field is not None and self.y_field is not None

    @property
    def has_pixel_position(self) -> bool:
        return (
            self.foot_x is not None
            and self.foot_y is not None
            and self.bbox_height is not None
            and self.bbox_height > 0.0
        )


@dataclass(frozen=True)
class BallSnapshot:
    """Ball position for a frame; either coordinate space may be missing."""

    x_pixel: float | None = None
    y_pixel: float | None = None
    x_field: float | None = None
    y_field: float | None = None
    in_play: bool = True

    @property
    def has_field_position(self) -> bool:
        return self.x_field is not None and self.y_field is not None

    @property
    def has_pixel_position(self) -> bool:
        return self.x_pixel is not None and self.y_pixel is not None


@dataclass(frozen=True)
class PossessionConfig:
    control_radius_m: float = 1.8
    control_radius_heights: float = 0.9
    contest_ratio: float = 1.35
    debounce_frames: int = 3
    unknown_timeout_ms: float = 2000.0
    pass_speed_threshold: float = 2.0

    def __post_init__(self) -> None:
        if self.debounce_frames < 1:
            raise ValueError("debounce_frames must be >= 1")
        if self.contest_ratio < 1.0:
            raise ValueError("contest_ratio must be >= 1.0")


@dataclass(frozen=True)
class PossessionResult:
    frame_id: int
    timestamp_ms: float
    state: PossessionState
    owner_track_id: int | None
    owner_team_id: int | None
    transition_reason: str


@dataclass(frozen=True)
class _Candidate:
    player: PlayerSnapshot
    distance: float  # normalized: metres or player-heights, comparable within frame


class PossessionTracker:
    """Streaming possession classifier with debounced control transitions."""

    def __init__(self, config: PossessionConfig | None = None) -> None:
        self._config = config or PossessionConfig()
        self._owner_track_id: int | None = None
        self._owner_team_id: int | None = None
        self._challenger_track_id: int | None = None
        self._challenger_streak = 0
        self._last_ball_ts: float | None = None
        self._last_ball_field: tuple[float, float] | None = None
        self._last_ball_field_ts: float | None = None

    def reset(self) -> None:
        self.__init__(self._config)

    # ------------------------------------------------------------- distances

    def _candidates(
        self, ball: BallSnapshot, players: Sequence[PlayerSnapshot]
    ) -> tuple[list[_Candidate], float]:
        """Return per-player normalized distances and the control threshold.

        Prefers metric field coordinates; falls back to bbox-height-normalized
        pixel distances. The two spaces are never mixed within one frame.
        """
        cfg = self._config
        if ball.has_field_position:
            metric = [
                _Candidate(
                    player=p,
                    distance=math.hypot(
                        p.x_field - ball.x_field,  # type: ignore[operator]
                        p.y_field - ball.y_field,  # type: ignore[operator]
                    ),
                )
                for p in players
                if p.has_field_position
            ]
            if metric:
                return metric, cfg.control_radius_m
        if ball.has_pixel_position:
            pixel = [
                _Candidate(
                    player=p,
                    distance=math.hypot(
                        p.foot_x - ball.x_pixel,  # type: ignore[operator]
                        p.foot_y - ball.y_pixel,  # type: ignore[operator]
                    )
                    / p.bbox_height,  # type: ignore[operator]
                )
                for p in players
                if p.has_pixel_position
            ]
            return pixel, cfg.control_radius_heights
        return [], math.inf

    def _ball_speed_ms(self, ball: BallSnapshot, timestamp_ms: float) -> float | None:
        """Metric ball speed (m/s) if two recent field positions exist."""
        if not ball.has_field_position:
            return None
        current = (float(ball.x_field), float(ball.y_field))  # type: ignore[arg-type]
        speed: float | None = None
        if self._last_ball_field is not None and self._last_ball_field_ts is not None:
            dt_s = (timestamp_ms - self._last_ball_field_ts) / 1000.0
            if dt_s > 1e-3:
                speed = (
                    math.hypot(
                        current[0] - self._last_ball_field[0],
                        current[1] - self._last_ball_field[1],
                    )
                    / dt_s
                )
        self._last_ball_field = current
        self._last_ball_field_ts = timestamp_ms
        return speed

    # ----------------------------------------------------------------- state

    def _clear_owner(self) -> None:
        self._owner_track_id = None
        self._owner_team_id = None
        self._challenger_track_id = None
        self._challenger_streak = 0

    def _result(
        self,
        frame_id: int,
        timestamp_ms: float,
        state: PossessionState,
        reason: str,
        owner_track: int | None = None,
        owner_team: int | None = None,
    ) -> PossessionResult:
        return PossessionResult(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            state=state,
            owner_track_id=owner_track,
            owner_team_id=owner_team,
            transition_reason=reason,
        )

    def step(
        self,
        frame_id: int,
        timestamp_ms: float,
        ball: BallSnapshot | None,
        players: Sequence[PlayerSnapshot] = (),
    ) -> PossessionResult:
        """Classify possession for one frame."""
        cfg = self._config

        if ball is None or (not ball.has_field_position and not ball.has_pixel_position):
            if (
                self._last_ball_ts is None
                or timestamp_ms - self._last_ball_ts > cfg.unknown_timeout_ms
            ):
                self._clear_owner()
                return self._result(
                    frame_id, timestamp_ms, PossessionState.UNKNOWN, "ball_lost_timeout"
                )
            # Ball briefly missing: keep the previous controlled state if any.
            if self._owner_team_id is not None:
                return self._result(
                    frame_id,
                    timestamp_ms,
                    _CONTROL_STATES[self._owner_team_id],
                    "ball_briefly_missing",
                    self._owner_track_id,
                    self._owner_team_id,
                )
            return self._result(
                frame_id, timestamp_ms, PossessionState.LOOSE_BALL, "ball_briefly_missing"
            )

        self._last_ball_ts = timestamp_ms
        ball_speed = self._ball_speed_ms(ball, timestamp_ms)

        if not ball.in_play:
            self._clear_owner()
            return self._result(
                frame_id, timestamp_ms, PossessionState.OUT_OF_PLAY, "ball_out_of_play"
            )

        candidates, control_radius = self._candidates(ball, players)
        in_range = sorted(
            (c for c in candidates if c.distance <= control_radius),
            key=lambda c: c.distance,
        )

        if not in_range:
            self._clear_owner()
            if ball_speed is not None and ball_speed >= cfg.pass_speed_threshold:
                return self._result(
                    frame_id, timestamp_ms, PossessionState.PASS_IN_FLIGHT, "ball_moving_free"
                )
            return self._result(
                frame_id, timestamp_ms, PossessionState.LOOSE_BALL, "no_player_in_range"
            )

        closest = in_range[0]
        rivals = [
            c
            for c in in_range[1:]
            if c.player.team_id != closest.player.team_id
            and c.distance <= closest.distance * cfg.contest_ratio
        ]
        if rivals:
            self._clear_owner()
            return self._result(
                frame_id, timestamp_ms, PossessionState.CONTESTED, "rival_within_contest_ratio"
            )

        team = closest.player.team_id
        if team not in _CONTROL_STATES:
            self._clear_owner()
            return self._result(
                frame_id, timestamp_ms, PossessionState.UNKNOWN, "unassigned_team_identity"
            )

        if self._owner_team_id is None:
            self._owner_track_id = closest.player.track_id
            self._owner_team_id = team
            self._challenger_track_id = None
            self._challenger_streak = 0
        elif team != self._owner_team_id or closest.player.track_id != self._owner_track_id:
            if closest.player.track_id == self._challenger_track_id:
                self._challenger_streak += 1
            else:
                self._challenger_track_id = closest.player.track_id
                self._challenger_streak = 1
            if self._challenger_streak >= cfg.debounce_frames:
                self._owner_track_id = closest.player.track_id
                self._owner_team_id = team
                self._challenger_track_id = None
                self._challenger_streak = 0
        else:
            self._challenger_track_id = None
            self._challenger_streak = 0

        assert self._owner_team_id is not None
        return self._result(
            frame_id,
            timestamp_ms,
            _CONTROL_STATES[self._owner_team_id],
            "closest_player_control",
            self._owner_track_id,
            self._owner_team_id,
        )
