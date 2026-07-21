"""Camera configuration helpers for multicamera matches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    video_path: str
    role: str = "broadcast"
    offset_seconds: float = 0.0
    manual_offset_seconds: float | None = None
    fps: float = 25.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.manual_offset_seconds is None:
            object.__setattr__(self, "manual_offset_seconds", float(self.offset_seconds))

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "video_path": self.video_path,
            "role": self.role,
            "offset_seconds": self.offset_seconds,
            "manual_offset_seconds": self.manual_offset_seconds,
            "fps": self.fps,
            "enabled": self.enabled,
        }


@dataclass
class MultiCameraSetup:
    """Ordered camera set with an explicit reference camera for sync."""

    cameras: list[CameraConfig] = field(default_factory=list)
    reference_camera_id: str | None = None

    def __post_init__(self) -> None:
        if not self.cameras:
            return
        ids = [camera.camera_id for camera in self.cameras]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate camera_id in MultiCameraSetup")
        if self.reference_camera_id is None:
            self.reference_camera_id = self.cameras[0].camera_id
        elif self.reference_camera_id not in ids:
            raise KeyError(f"reference_camera_id unknown: {self.reference_camera_id}")

    def __contains__(self, camera_id: object) -> bool:
        return any(camera.camera_id == camera_id for camera in self.cameras)

    def __iter__(self) -> Iterator[CameraConfig]:
        return iter(self.cameras)

    def camera(self, camera_id: str) -> CameraConfig:
        for camera in self.cameras:
            if camera.camera_id == camera_id:
                return camera
        raise KeyError(f"unknown camera_id: {camera_id}")

    @classmethod
    def from_iterable(
        cls,
        cameras: Iterable[CameraConfig],
        *,
        reference_camera_id: str | None = None,
    ) -> MultiCameraSetup:
        return cls(cameras=list(cameras), reference_camera_id=reference_camera_id)
