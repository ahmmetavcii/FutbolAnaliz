#!/usr/bin/env python3
"""Export canonical football-analytics artifacts as SoccerNet GSR JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_analytics.integrations.sn_gamestate_compatible import (  # noqa: E402
    ExportInputs,
    write_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean-room SoccerNet Game State-compatible JSON exporter"
    )
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--track-identities", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--game-state", required=True, type=Path)
    parser.add_argument(
        "--jersey-predictions",
        type=Path,
        help="Optional CSV/JSON with track_id, optional frame_id, and jersey",
    )
    parser.add_argument("--video-id", help="Override game_state match_id")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = write_predictions(
        ExportInputs(
            detections=args.detections,
            tracks=args.tracks,
            track_identities=args.track_identities,
            calibration=args.calibration,
            game_state=args.game_state,
            jersey_predictions=args.jersey_predictions,
        ),
        args.output,
        video_id=args.video_id,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
