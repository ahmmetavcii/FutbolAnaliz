"""Canonical PyArrow schemas for stable MVP-1 and nullable MVP-2 artifacts."""

from __future__ import annotations

import pyarrow as pa

SCHEMA_VERSION = "1.0.0"

DETECTIONS_SCHEMA = pa.schema(
    [
        ("frame_id", pa.int64()),
        ("timestamp_ms", pa.float64()),
        ("detection_id", pa.string()),
        ("object_type", pa.string()),
        ("class_id", pa.int64()),
        ("bbox_x1", pa.float64()),
        ("bbox_y1", pa.float64()),
        ("bbox_x2", pa.float64()),
        ("bbox_y2", pa.float64()),
        ("detection_confidence", pa.float64()),
        ("source_model", pa.string()),
        ("schema_version", pa.string()),
    ]
)

TRACKS_SCHEMA = pa.schema(
    [
        ("frame_id", pa.int64()),
        ("timestamp_ms", pa.float64()),
        ("track_id", pa.int64()),
        ("detection_id", pa.string()),
        ("object_type", pa.string()),
        ("class_id", pa.int64()),
        ("bbox_x1", pa.float64()),
        ("bbox_y1", pa.float64()),
        ("bbox_x2", pa.float64()),
        ("bbox_y2", pa.float64()),
        ("foot_x_pixel", pa.float64()),
        ("foot_y_pixel", pa.float64()),
        ("tracking_confidence", pa.float64()),
        ("source_tracker", pa.string()),
        ("source_model", pa.string()),
        ("schema_version", pa.string()),
    ]
)

REQUIRED_DETECTION_COLUMNS = [field.name for field in DETECTIONS_SCHEMA]
REQUIRED_TRACK_COLUMNS = [field.name for field in TRACKS_SCHEMA]

COMMON_FIELDS = [
    ("schema_version", pa.string()),
    ("run_id", pa.string()),
    ("match_id", pa.string()),
    ("frame_id", pa.int64()),
    ("timestamp_ms", pa.float64()),
    ("source_method", pa.string()),
    ("confidence", pa.float64()),
    ("valid", pa.bool_()),
]


def _schema(*fields: tuple[str, pa.DataType]) -> pa.Schema:
    return pa.schema([*COMMON_FIELDS, *fields])


SHOT_SEGMENTS_SCHEMA = _schema(
    ("shot_type", pa.string()),
    ("green_ratio", pa.float64()),
    ("mean_player_height_ratio", pa.float64()),
    ("player_count", pa.int64()),
    ("histogram_difference", pa.float64()),
    ("optical_flow_magnitude", pa.float64()),
    ("scene_cut", pa.bool_()),
    ("invalid_reason", pa.string()),
)

TRACK_IDENTITIES_SCHEMA = _schema(
    ("track_id", pa.int64()),
    ("role", pa.string()),
    ("role_confidence", pa.float64()),
    ("team_id", pa.string()),
    ("team_confidence", pa.float64()),
    ("color_quality", pa.float64()),
    ("temporal_consistency", pa.float64()),
)

CAMERA_MOTION_SCHEMA = _schema(
    ("dx_pixel", pa.float64()),
    ("dy_pixel", pa.float64()),
    ("rotation_deg", pa.float64()),
    ("scale", pa.float64()),
    ("inlier_count", pa.int64()),
    ("inlier_ratio", pa.float64()),
    ("reset_reason", pa.string()),
)

CALIBRATION_SCHEMA = _schema(
    ("segment_id", pa.int64()),
    ("provider", pa.string()),
    ("homography_json", pa.string()),
    ("orientation", pa.string()),
    ("pitch_length_m", pa.float64()),
    ("pitch_width_m", pa.float64()),
    ("reprojection_error", pa.float64()),
    ("visible_pitch_coverage", pa.float64()),
    ("invalid_reason", pa.string()),
)

GAME_STATE_SCHEMA = _schema(
    ("track_id", pa.int64()),
    ("role", pa.string()),
    ("team_id", pa.string()),
    ("foot_x_pixel", pa.float64()),
    ("foot_y_pixel", pa.float64()),
    ("x_field", pa.float64()),
    ("y_field", pa.float64()),
    ("shot_type", pa.string()),
    ("calibration_confidence", pa.float64()),
    ("invalid_reason", pa.string()),
)

BALL_STATE_SCHEMA = _schema(
    ("ball_x_pixel", pa.float64()),
    ("ball_y_pixel", pa.float64()),
    ("ball_x_field", pa.float64()),
    ("ball_y_field", pa.float64()),
    ("visibility_state", pa.string()),
    ("detection_confidence", pa.float64()),
    ("trajectory_confidence", pa.float64()),
    ("invalid_reason", pa.string()),
)

POSSESSION_TIMELINE_SCHEMA = _schema(
    ("owner_track_id", pa.int64()),
    ("owner_team_id", pa.string()),
    ("possession_state", pa.string()),
    ("transition_reason", pa.string()),
)

TRACK_QUALITY_SCHEMA = _schema(
    ("track_id", pa.int64()),
    ("track_length", pa.int64()),
    ("visible_frames", pa.int64()),
    ("coverage", pa.float64()),
    ("detection_confidence_mean", pa.float64()),
    ("detection_confidence_min", pa.float64()),
    ("bbox_jitter", pa.float64()),
    ("fragmentation", pa.int64()),
    ("scene_cut_count", pa.int64()),
    ("border_truncation", pa.float64()),
    ("team_consistency", pa.float64()),
    ("usable_for_metrics", pa.bool_()),
    ("invalid_reason", pa.string()),
)

REID_EMBEDDINGS_SCHEMA = _schema(
    ("track_id", pa.int64()),
    ("bbox_x1", pa.float64()),
    ("bbox_y1", pa.float64()),
    ("bbox_x2", pa.float64()),
    ("bbox_y2", pa.float64()),
    ("embedding", pa.list_(pa.float32())),
    ("embedding_dim", pa.int64()),
    ("model_name", pa.string()),
)

TRACK_REID_PROTOTYPES_SCHEMA = _schema(
    ("track_id", pa.int64()),
    ("n_samples", pa.int64()),
    ("embedding", pa.list_(pa.float32())),
    ("embedding_dim", pa.int64()),
    ("model_name", pa.string()),
)

PLAYER_METRICS_SCHEMA = _schema(
    ("track_id", pa.int64()),
    ("x_field", pa.float64()),
    ("y_field", pa.float64()),
    ("instantaneous_speed_kmh", pa.float64()),
    ("smoothed_speed_kmh", pa.float64()),
    ("cumulative_distance_m", pa.float64()),
    ("sprint_state", pa.string()),
    ("coverage", pa.float64()),
    ("invalid_reason", pa.string()),
)

TEAM_METRICS_SCHEMA = _schema(
    ("team_id", pa.string()),
    ("centroid_x", pa.float64()),
    ("centroid_y", pa.float64()),
    ("width_m", pa.float64()),
    ("depth_m", pa.float64()),
    ("mean_interplayer_distance_m", pa.float64()),
    ("compactness_m", pa.float64()),
    ("last_line_height_m", pa.float64()),
    ("player_count", pa.int64()),
    ("player_coverage", pa.float64()),
    ("regional_occupancy_json", pa.string()),
    ("invalid_reason", pa.string()),
)

MVP2_SCHEMAS: dict[str, pa.Schema] = {
    "shot_segments": SHOT_SEGMENTS_SCHEMA,
    "track_identities": TRACK_IDENTITIES_SCHEMA,
    "camera_motion": CAMERA_MOTION_SCHEMA,
    "calibration": CALIBRATION_SCHEMA,
    "game_state": GAME_STATE_SCHEMA,
    "ball_state": BALL_STATE_SCHEMA,
    "possession_timeline": POSSESSION_TIMELINE_SCHEMA,
    "track_quality": TRACK_QUALITY_SCHEMA,
    "reid_embeddings": REID_EMBEDDINGS_SCHEMA,
    "track_reid_prototypes": TRACK_REID_PROTOTYPES_SCHEMA,
    "player_metrics": PLAYER_METRICS_SCHEMA,
    "team_metrics": TEAM_METRICS_SCHEMA,
}


def validate_detections_frame(columns: list[str]) -> None:
    missing = [name for name in REQUIRED_DETECTION_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"detections.parquet missing columns: {missing}")


def validate_tracks_frame(columns: list[str]) -> None:
    missing = [name for name in REQUIRED_TRACK_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"tracks.parquet missing columns: {missing}")


def validate_mvp2_columns(name: str, columns: list[str]) -> None:
    schema = MVP2_SCHEMAS[name]
    missing = [field.name for field in schema if field.name not in columns]
    if missing:
        raise ValueError(f"{name}.parquet missing columns: {missing}")
