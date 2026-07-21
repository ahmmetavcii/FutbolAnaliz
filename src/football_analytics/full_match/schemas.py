"""Pydantic v2 schemas for resumable full-match processing."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0.0"
ALLOWED_CAMERA_COUNTS = (1, 2, 4)

DEFAULT_CHUNK_SECONDS = 120.0
MIN_CHUNK_SECONDS = 30.0
MAX_CHUNK_SECONDS = 300.0

STAGE_ORDER = (
    "prepare",
    "sync",
    "calibration",
    "chunks",
    "detection",
    "tracking",
    "events",
    "consolidation",
    "export",
)
MODEL_STAGES = ("detection", "tracking", "events")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CameraRole(str, Enum):
    TACTICAL_FULL = "tactical_full"
    TACTICAL_LEFT = "tactical_left"
    TACTICAL_RIGHT = "tactical_right"
    BROADCAST = "broadcast"
    GOAL_LEFT = "goal_left"
    GOAL_RIGHT = "goal_right"
    REVERSE_ANGLE = "reverse_angle"
    CUSTOM = "custom"


class ChunkStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASS = "PASS"
    FAILED = "FAILED"
    RETRY = "RETRY"
    SKIPPED = "SKIPPED"
    INVALID_INPUT = "INVALID_INPUT"
    INVALIDATED = "INVALIDATED"


class StageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASS = "PASS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    INVALIDATED = "INVALIDATED"


class FrameCheck(BaseModel):
    """One decoded-sample check (first/middle/last frame)."""

    model_config = ConfigDict(extra="forbid")

    position: str
    frame_index: int = Field(ge=0)
    ok: bool
    mean_intensity: float | None = None


class VideoProbe(BaseModel):
    """ffprobe metadata combined with OpenCV decode spot checks."""

    model_config = ConfigDict(extra="forbid")

    path: str
    duration_seconds: float = Field(gt=0)
    avg_frame_rate: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    codec_name: str | None = None
    nb_frames: int | None = None
    size_bytes: int = Field(ge=0)
    bit_rate: int = Field(ge=0, default=0)
    frame_checks: list[FrameCheck] = Field(default_factory=list)

    @property
    def decodable(self) -> bool:
        return bool(self.frame_checks) and all(check.ok for check in self.frame_checks)


class CameraSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(min_length=1)
    role: CameraRole
    path: str
    sha256: str = Field(min_length=64, max_length=64)
    probe: VideoProbe | None = None


class MatchManifest(BaseModel):
    """Top-level manifest describing one match and its camera inputs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    match_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=utc_now_iso)
    chunk_seconds: float = Field(
        default=DEFAULT_CHUNK_SECONDS, ge=MIN_CHUNK_SECONDS, le=MAX_CHUNK_SECONDS
    )
    profile: str | None = None
    cameras: list[CameraSpec]

    @field_validator("cameras")
    @classmethod
    def _validate_cameras(cls, value: list[CameraSpec]) -> list[CameraSpec]:
        if len(value) not in ALLOWED_CAMERA_COUNTS:
            raise ValueError(
                f"exactly {ALLOWED_CAMERA_COUNTS} cameras are supported, got {len(value)}"
            )
        camera_ids = [camera.camera_id for camera in value]
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError(f"camera_id values must be unique: {camera_ids}")
        return value

    def camera(self, camera_id: str) -> CameraSpec:
        for spec in self.cameras:
            if spec.camera_id == camera_id:
                return spec
        raise KeyError(f"unknown camera_id: {camera_id}")


class Fingerprints(BaseModel):
    """Identity of the configuration, models, and inputs a result depends on."""

    model_config = ConfigDict(extra="forbid")

    config: str
    model: str
    inputs: dict[str, str]

    def combined(self, camera_id: str) -> str:
        input_fp = self.inputs.get(camera_id, "missing")
        payload = f"{self.config}|{self.model}|{camera_id}|{input_fp}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str
    chunk_index: int = Field(ge=0)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    frame_start: int = Field(ge=0)
    frame_end: int = Field(gt=0)
    status: ChunkStatus = ChunkStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    fingerprint: str | None = None
    error: str | None = None
    wall_seconds: float | None = None
    result_path: str | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


# Typed Parquet schema for chunk status tables (atomic writes use this schema).
CHUNK_STATUS_SCHEMA = pa.schema(
    [
        pa.field("camera_id", pa.string(), nullable=False),
        pa.field("chunk_index", pa.int32(), nullable=False),
        pa.field("start_seconds", pa.float64(), nullable=False),
        pa.field("end_seconds", pa.float64(), nullable=False),
        pa.field("frame_start", pa.int64(), nullable=False),
        pa.field("frame_end", pa.int64(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("attempts", pa.int32(), nullable=False),
        pa.field("fingerprint", pa.string()),
        pa.field("error", pa.string()),
        pa.field("wall_seconds", pa.float64()),
        pa.field("result_path", pa.string()),
        pa.field("updated_at", pa.string(), nullable=False),
    ]
)


def chunk_records_to_rows(records: list[ChunkRecord]) -> list[dict[str, Any]]:
    """Serialize chunk records to rows matching CHUNK_STATUS_SCHEMA."""
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "camera_id": record.camera_id,
                "chunk_index": int(record.chunk_index),
                "start_seconds": float(record.start_seconds),
                "end_seconds": float(record.end_seconds),
                "frame_start": int(record.frame_start),
                "frame_end": int(record.frame_end),
                "status": record.status.value,
                "attempts": int(record.attempts),
                "fingerprint": record.fingerprint,
                "error": record.error,
                "wall_seconds": record.wall_seconds,
                "result_path": record.result_path,
                "updated_at": record.updated_at,
            }
        )
    return rows


def chunk_records_to_table(records: list[ChunkRecord]) -> pa.Table:
    return pa.Table.from_pylist(chunk_records_to_rows(records), schema=CHUNK_STATUS_SCHEMA)


class ChunkManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    match_id: str
    chunk_seconds: float = Field(ge=MIN_CHUNK_SECONDS, le=MAX_CHUNK_SECONDS)
    records: list[ChunkRecord] = Field(default_factory=list)

    def counts_by_status(self) -> dict[str, int]:
        counts: dict[str, int] = {status.value: 0 for status in ChunkStatus}
        for record in self.records:
            counts[record.status.value] += 1
        return counts

    def to_arrow_table(self) -> pa.Table:
        return chunk_records_to_table(self.records)


class StageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: StageStatus = StageStatus.PENDING
    reason: str | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


class RunState(BaseModel):
    """Persistent state of one full-match run directory."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    match_id: str
    prepared_dir: str
    run_dir: str
    retry_limit: int = Field(default=3, ge=1)
    fail_fast: bool = True
    chunk_pipeline_adapter: str | None = None
    fingerprints: Fingerprints
    stages: list[StageRecord] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    def stage(self, name: str) -> StageRecord:
        for record in self.stages:
            if record.name == name:
                return record
        raise KeyError(f"unknown stage: {name}")

    def set_stage(self, name: str, status: StageStatus, reason: str | None = None) -> None:
        record = self.stage(name)
        record.status = status
        record.reason = reason
        record.updated_at = utc_now_iso()
        self.updated_at = utc_now_iso()


def default_stage_records() -> list[StageRecord]:
    return [StageRecord(name=name) for name in STAGE_ORDER]


def stages_from(stage: str) -> tuple[str, ...]:
    """Return the given stage and every downstream stage."""
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGE_ORDER}")
    return STAGE_ORDER[STAGE_ORDER.index(stage) :]
