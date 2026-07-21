"""Multi-camera synchronization, calibration, identity, and fusion.

Public entry points resolved by ``scripts/sync_cameras.py`` and
``scripts/calibrate_cameras.py`` are :func:`synchronize_cameras` and
:func:`calibrate_cameras` (aliased as ``sync_cameras`` / ``calibrate_run``).
"""

from .audio_sync import (
    AudioSyncConfig,
    AudioSyncResult,
    cross_correlate_offset,
    estimate_audio_sync,
    windowed_offsets,
)
from .calibration_manager import (
    CalibrationGates,
    CalibrationManager,
    CameraCalibration,
    calibrate_cameras,
    validate_calibration,
)
from .camera_config import CameraConfig, MultiCameraSetup
from .camera_coverage import (
    CameraCoverage,
    compute_camera_coverage,
    coverage_report,
    find_coverage_gaps,
)
from .cross_camera_reid import (
    CueScores,
    IdentitySnapshot,
    MatchScore,
    ReidMatchConfig,
    score_candidate,
)
from .drift_correction import DriftModel, apply_drift_model, fit_drift_model
from .duplicate_suppression import (
    DuplicateConfig,
    DuplicatePair,
    find_impossible_shared_identities,
    find_intra_camera_duplicates,
    suppress_duplicates,
)
from .global_identity import (
    Assignment,
    AssignmentStatus,
    GlobalIdentity,
    GlobalIdentityRegistry,
    apply_manual_corrections,
)
from .identity_audit import AuditEvent, AuditEventKind, IdentityAuditLog
from .local_tracking import (
    LocalObservation,
    LocalTrack,
    PlayerRole,
    cosine_similarity,
    group_into_tracks,
)
from .observation_fusion import (
    FusedObservation,
    FusionConfig,
    fuse_observations,
    fuse_timeline,
)
from .synchronization import (
    OffsetEstimate,
    OffsetSource,
    TimelineSynchronizer,
    load_offsets_file,
    synchronize_cameras,
)
from .visual_sync import (
    MatchedEventPair,
    VisualSyncResult,
    offset_from_activity_signals,
    offset_from_matched_events,
)

# Compatibility aliases for callers that resolve the alternative entry-point
# names probed by the full-match CLIs.
sync_cameras = synchronize_cameras
calibrate_run = calibrate_cameras

__all__ = [
    # synchronization
    "AudioSyncConfig",
    "AudioSyncResult",
    "cross_correlate_offset",
    "estimate_audio_sync",
    "windowed_offsets",
    "MatchedEventPair",
    "VisualSyncResult",
    "offset_from_activity_signals",
    "offset_from_matched_events",
    "DriftModel",
    "apply_drift_model",
    "fit_drift_model",
    "OffsetEstimate",
    "OffsetSource",
    "TimelineSynchronizer",
    "load_offsets_file",
    "synchronize_cameras",
    "sync_cameras",
    # configuration & calibration
    "CameraConfig",
    "MultiCameraSetup",
    "CalibrationGates",
    "CalibrationManager",
    "CameraCalibration",
    "calibrate_cameras",
    "calibrate_run",
    "validate_calibration",
    # identity
    "CueScores",
    "IdentitySnapshot",
    "MatchScore",
    "ReidMatchConfig",
    "score_candidate",
    "Assignment",
    "AssignmentStatus",
    "GlobalIdentity",
    "GlobalIdentityRegistry",
    "apply_manual_corrections",
    "AuditEvent",
    "AuditEventKind",
    "IdentityAuditLog",
    # observations & fusion
    "LocalObservation",
    "LocalTrack",
    "PlayerRole",
    "cosine_similarity",
    "group_into_tracks",
    "FusedObservation",
    "FusionConfig",
    "fuse_observations",
    "fuse_timeline",
    "DuplicateConfig",
    "DuplicatePair",
    "find_impossible_shared_identities",
    "find_intra_camera_duplicates",
    "suppress_duplicates",
    "CameraCoverage",
    "compute_camera_coverage",
    "coverage_report",
    "find_coverage_gaps",
]
