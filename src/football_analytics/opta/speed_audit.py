"""Physical metric quality: spike rejection + max-speed audit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def audit_and_filter_speeds(
    player_metrics: pd.DataFrame,
    *,
    max_plausible_kmh: float = 38.0,
    min_consecutive: int = 3,
    p95_for_report: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Filter single-frame speed spikes; report candidates near 36–40 km/h."""
    if player_metrics is None or player_metrics.empty:
        return player_metrics, []

    frame = player_metrics.copy()
    speed_col = None
    for cand in ("smoothed_speed_kmh", "speed_kmh", "instant_speed_kmh"):
        if cand in frame.columns:
            speed_col = cand
            break
    if speed_col is None:
        return frame, []

    audits: list[dict[str, Any]] = []
    # Mark spikes: above plausible OR isolated peaks
    speeds = pd.to_numeric(frame[speed_col], errors="coerce")
    spike = speeds > max_plausible_kmh
    # Isolated: high vs neighbors
    isolated = pd.Series(False, index=frame.index)
    for tid, g in frame.groupby("track_id"):
        idx = g.index.to_list()
        s = speeds.loc[idx].to_numpy()
        for i, ix in enumerate(idx):
            if not np.isfinite(s[i]):
                continue
            if s[i] < 30.0:
                continue
            left = s[i - 1] if i > 0 else s[i]
            right = s[i + 1] if i + 1 < len(s) else s[i]
            if s[i] - max(left, right) > 8.0:
                isolated.loc[ix] = True
            if s[i] >= 36.0:
                audits.append(
                    {
                        "track_id": int(tid),
                        "global_player_id": int(tid),
                        "frame_id": int(g.loc[ix, "frame_id"])
                        if "frame_id" in g.columns
                        else None,
                        "timestamp_ms": float(g.loc[ix, "timestamp_ms"])
                        if "timestamp_ms" in g.columns
                        else None,
                        "speed_kmh": float(s[i]),
                        "reason": "high_speed_candidate",
                        "calibration_confidence": float(g.loc[ix, "confidence"])
                        if "confidence" in g.columns
                        else None,
                    }
                )

    reject = spike | isolated
    frame["speed_spike_rejected"] = reject
    if reject.any() and speed_col in frame.columns:
        frame.loc[reject, speed_col] = np.nan

    # Per-track reported max: require consecutive non-nan frames; prefer p95
    reported = []
    for tid, g in frame.groupby("track_id"):
        s = pd.to_numeric(g[speed_col], errors="coerce").dropna()
        if s.empty:
            continue
        # consecutive run check for max
        vals = pd.to_numeric(g[speed_col], errors="coerce").to_numpy()
        best = None
        run = []
        for v in vals:
            if np.isfinite(v):
                run.append(float(v))
                if len(run) >= min_consecutive:
                    peak = max(run[-min_consecutive:])
                    best = peak if best is None else max(best, peak)
            else:
                run = []
        p95 = float(np.nanpercentile(s.to_numpy(), 95))
        reported.append(
            {
                "track_id": int(tid),
                "max_speed_raw_kmh": float(s.max()),
                "max_speed_consecutive_kmh": best,
                "p95_speed_kmh": p95,
                "reported_max_speed_kmh": p95 if p95_for_report else best,
            }
        )
    if reported:
        rep = pd.DataFrame(reported)
        frame = frame.merge(rep, on="track_id", how="left")
    return frame, audits
