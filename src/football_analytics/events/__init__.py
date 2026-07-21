"""Match event detection, evidence, review, clips, and summaries.

Also re-exports the run-level recompute entry point from
``football_analytics.full_match.recompute`` for the corrections CLI.
"""

from football_analytics.events.assist_detector import (
    AssistDetector,
    AssistDetectorConfig,
    PassObservation,
)
from football_analytics.events.event_clips import (
    ClipConfig,
    ClipWindow,
    build_clips,
    clip_for_event,
    export_clip_mp4,
    opencv_available,
)
from football_analytics.events.event_detector import EventDetector, EventDetectorConfig
from football_analytics.events.event_evidence import EvidenceBundle, EvidenceItem
from football_analytics.events.event_review import (
    Correction,
    CorrectionKind,
    ReviewLog,
    ReviewResult,
    apply_review,
)
from football_analytics.events.event_summary import (
    EventSummary,
    recompute_summary,
    summarize_events,
)
from football_analytics.events.goal_detector import GoalDetector, GoalDetectorConfig, GoalSignals
from football_analytics.events.schemas import (
    CONFIRMED_STATUSES,
    EventStatus,
    EventType,
    MatchEvent,
    SubstitutionInterval,
    is_confirmed,
    is_countable,
)
from football_analytics.events.shot_detector import ShotDetector, ShotDetectorConfig, ShotSignals
from football_analytics.events.substitution_detector import (
    PitchEntry,
    PitchExit,
    SubstitutionDetector,
    SubstitutionDetectorConfig,
    interval_of,
)
from football_analytics.full_match.recompute import recompute_after_manual_correction

recompute_events = recompute_after_manual_correction

__all__ = [
    "AssistDetector",
    "AssistDetectorConfig",
    "CONFIRMED_STATUSES",
    "ClipConfig",
    "ClipWindow",
    "Correction",
    "CorrectionKind",
    "EventDetector",
    "EventDetectorConfig",
    "EventStatus",
    "EventSummary",
    "EventType",
    "EvidenceBundle",
    "EvidenceItem",
    "GoalDetector",
    "GoalDetectorConfig",
    "GoalSignals",
    "MatchEvent",
    "PassObservation",
    "PitchEntry",
    "PitchExit",
    "ReviewLog",
    "ReviewResult",
    "ShotDetector",
    "ShotDetectorConfig",
    "ShotSignals",
    "SubstitutionDetector",
    "SubstitutionDetectorConfig",
    "SubstitutionInterval",
    "apply_review",
    "build_clips",
    "clip_for_event",
    "export_clip_mp4",
    "interval_of",
    "is_confirmed",
    "is_countable",
    "opencv_available",
    "recompute_after_manual_correction",
    "recompute_events",
    "recompute_summary",
    "summarize_events",
]
