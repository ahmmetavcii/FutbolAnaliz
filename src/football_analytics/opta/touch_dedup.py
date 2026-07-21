"""Touch event deduplication: merge near-duplicate contacts into one event."""

from __future__ import annotations

from typing import Any

import pandas as pd


def deduplicate_touches(
    touches: pd.DataFrame,
    *,
    window_ms: float = 300.0,
) -> pd.DataFrame:
    """Merge same-player contacts within ``window_ms`` into one event.

    Keeps peak-confidence frame metadata and expands start/end frames.
    """
    if touches is None or touches.empty:
        return touches

    ordered = touches.sort_values(["track_id", "timestamp_ms"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    i = 0
    eid = 0
    while i < len(ordered):
        seed = ordered.iloc[i].to_dict()
        group = [seed]
        j = i + 1
        while j < len(ordered):
            nxt = ordered.iloc[j]
            if int(nxt["track_id"]) != int(seed["track_id"]):
                break
            if float(nxt["timestamp_ms"]) - float(group[-1]["timestamp_ms"]) > window_ms:
                break
            group.append(nxt.to_dict())
            j += 1
        eid += 1
        peak = max(group, key=lambda r: float(r.get("confidence") or 0.0))
        merged = dict(peak)
        merged["touch_id"] = f"touch-{eid:05d}"
        merged["start_frame"] = int(min(int(r["frame_id"]) for r in group))
        merged["end_frame"] = int(max(int(r["frame_id"]) for r in group))
        merged["peak_confidence_frame"] = int(peak["frame_id"])
        merged["raw_touch_count"] = len(group)
        merged["deduplicated"] = len(group) > 1
        merged["timestamp_ms"] = float(peak["timestamp_ms"])
        merged["frame_id"] = int(peak["frame_id"])
        merged["confidence"] = float(peak.get("confidence") or 0.0)
        rows.append(merged)
        i = j
    return pd.DataFrame(rows)
