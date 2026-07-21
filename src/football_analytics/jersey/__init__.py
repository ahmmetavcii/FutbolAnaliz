"""Clean-room temporal jersey recognition."""

from .dataset import JerseyTrackletDataset, build_records, deterministic_tracklet_split
from .infer import predict_records, run_inference
from .model import TemporalJerseyRecognizer, build_model
from .schemas import (
    KNOWN_LABELS,
    NUM_CLASSES,
    UNKNOWN_CLASS_INDEX,
    UNKNOWN_LABEL,
    JerseyPrediction,
    TrackletRecord,
)

__all__ = [
    "JerseyPrediction",
    "JerseyTrackletDataset",
    "KNOWN_LABELS",
    "NUM_CLASSES",
    "TemporalJerseyRecognizer",
    "TrackletRecord",
    "UNKNOWN_CLASS_INDEX",
    "UNKNOWN_LABEL",
    "build_model",
    "build_records",
    "deterministic_tracklet_split",
    "predict_records",
    "run_inference",
]
