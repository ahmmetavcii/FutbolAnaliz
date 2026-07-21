from __future__ import annotations

import pytest

from football_analytics.multicamera import LocalObservation, PlayerRole


@pytest.fixture
def make_obs():
    """Factory for LocalObservation with sane defaults."""

    def factory(
        camera_id: str = "cam1",
        local_track_id: int = 1,
        frame_index: int = 0,
        reference_time_seconds: float = 0.0,
        bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 20.0),
        pitch_xy_m: tuple[float, float] | None = None,
        team_id: int | None = None,
        team_confidence: float = 0.0,
        jersey_number: int | None = None,
        jersey_confidence: float = 0.0,
        role: PlayerRole = PlayerRole.UNKNOWN,
        reid_embedding: tuple[float, ...] | None = None,
        detection_confidence: float = 1.0,
    ) -> LocalObservation:
        return LocalObservation(
            camera_id=camera_id,
            local_track_id=local_track_id,
            frame_index=frame_index,
            reference_time_seconds=reference_time_seconds,
            bbox_xyxy=bbox_xyxy,
            pitch_xy_m=pitch_xy_m,
            team_id=team_id,
            team_confidence=team_confidence,
            jersey_number=jersey_number,
            jersey_confidence=jersey_confidence,
            role=role,
            reid_embedding=reid_embedding,
            detection_confidence=detection_confidence,
        )

    return factory
