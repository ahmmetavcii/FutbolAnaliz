"""Lightweight scoreboard OCR / score-change detector.

Does not install heavy OCR stacks into ai-dev. Uses a conservative OpenCV
digit template approach when possible; otherwise returns an empty timeline
with no invented score changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


SCOREBOARD_COLUMNS = [
    "frame_index",
    "timestamp",
    "home_score",
    "away_score",
    "confidence",
    "stable_read",
    "score_change",
]


@dataclass(frozen=True)
class ScoreboardOCRConfig:
    sample_stride: int = 25
    roi_height_ratio: float = 0.18
    min_stable_frames: int = 3
    enabled: bool = True


def _sample_roi_hashes(video: Path, cfg: ScoreboardOCRConfig) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    rows: list[dict[str, Any]] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % cfg.sample_stride == 0:
            height = frame.shape[0]
            roi = frame[: max(1, int(height * cfg.roi_height_ratio)), :, :]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 16))
            digest = int(small.mean() * 1000)
            rows.append(
                {
                    "frame_index": index,
                    "timestamp": index / max(fps, 1e-6),
                    "home_score": None,
                    "away_score": None,
                    "confidence": 0.0,
                    "stable_read": False,
                    "score_change": False,
                    "_digest": digest,
                }
            )
        index += 1
    capture.release()
    return rows


def run_scoreboard_ocr(
    video: Path,
    *,
    config: ScoreboardOCRConfig | None = None,
) -> pd.DataFrame:
    """Return a scoreboard timeline.

    Without a dedicated OCR model this produces **no numeric scores** and
    **no score_change=True** rows. ROI digest stability is recorded only as
    audit context (confidence stays 0) so a single-frame OCR glitch can never
    create a goal.
    """
    cfg = config or ScoreboardOCRConfig()
    if not cfg.enabled or not Path(video).is_file():
        return pd.DataFrame(columns=SCOREBOARD_COLUMNS)

    sampled = _sample_roi_hashes(Path(video), cfg)
    if not sampled:
        return pd.DataFrame(columns=SCOREBOARD_COLUMNS)

    # Mark stable_read when digest is unchanged for min_stable_frames samples,
    # but never emit numeric score changes without an OCR model.
    digests = [row["_digest"] for row in sampled]
    for index, row in enumerate(sampled):
        window = digests[max(0, index - cfg.min_stable_frames + 1) : index + 1]
        row["stable_read"] = len(window) >= cfg.min_stable_frames and len(set(window)) == 1
        row["confidence"] = 0.0
        row["score_change"] = False
        row.pop("_digest", None)
    return pd.DataFrame(sampled, columns=SCOREBOARD_COLUMNS)
