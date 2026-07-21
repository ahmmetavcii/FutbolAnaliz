"""Evaluation package: TrackEval adapter + Opta accuracy evaluation helpers."""

from football_analytics.evaluation.publishability import (
    PublishabilityFlags,
    compute_publishability,
)
from football_analytics.evaluation.trackeval_adapter import (
    DEFAULT_TRACKEVAL_ROOT,
    EvaluationResult,
    canonical_gt_to_mot,
    canonical_tracks_to_mot,
    export_soccernet_gs_predictions,
    parse_soccernet_gs_predictions,
    run_trackeval,
)

__all__ = [
    "DEFAULT_TRACKEVAL_ROOT",
    "EvaluationResult",
    "PublishabilityFlags",
    "canonical_gt_to_mot",
    "canonical_tracks_to_mot",
    "compute_publishability",
    "export_soccernet_gs_predictions",
    "parse_soccernet_gs_predictions",
    "run_trackeval",
]
