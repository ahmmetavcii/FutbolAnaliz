"""Match-node-tracker integration package (feature-flagged adapters).

Upstream source is kept immutable under:
  third_party/authorized/match-node-tracker/
"""

from __future__ import annotations

__all__ = [
    "MatchNodeDetectorAdapter",
    "MatchNodeTeamColorAdapter",
    "MatchNodeCameraMotionAdapter",
    "MatchNodeMarkerRenderer",
    "MatchNodeTrackerAdapter",
]

from football_analytics.integrations.match_node_tracker.detector_adapter import (
    MatchNodeDetectorAdapter,
)
from football_analytics.integrations.match_node_tracker.team_color_adapter import (
    MatchNodeTeamColorAdapter,
)
from football_analytics.integrations.match_node_tracker.camera_motion_adapter import (
    MatchNodeCameraMotionAdapter,
)
from football_analytics.integrations.match_node_tracker.marker_renderer import (
    MatchNodeMarkerRenderer,
)
from football_analytics.integrations.match_node_tracker.tracker_adapter import (
    MatchNodeTrackerAdapter,
)
