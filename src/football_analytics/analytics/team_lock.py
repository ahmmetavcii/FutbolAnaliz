"""Lock team assignments so a track/player cannot flip sides mid-clip."""

from __future__ import annotations

import pandas as pd


def lock_track_teams(
    identities: pd.DataFrame,
    *,
    min_confidence: float = 0.35,
) -> pd.DataFrame:
    """Majority/confidence-weighted team lock per local ``track_id``."""
    if identities is None or identities.empty or "track_id" not in identities.columns:
        return identities
    out = identities.copy()
    for tid, group in out.groupby("track_id"):
        valid = group[group["team_id"].notna()].copy()
        if "team_confidence" in valid.columns:
            strong = valid[valid["team_confidence"].fillna(0) >= min_confidence]
            if not strong.empty:
                valid = strong
        if valid.empty:
            continue
        if "team_confidence" in valid.columns:
            scores = valid.groupby(valid["team_id"].astype(str))["team_confidence"].sum()
        else:
            scores = valid.groupby(valid["team_id"].astype(str)).size()
        locked = str(scores.idxmax())
        mean_conf = float(
            valid.loc[valid["team_id"].astype(str) == locked, "team_confidence"].mean()
        ) if "team_confidence" in valid.columns else 0.7
        mask = out["track_id"] == tid
        out.loc[mask, "team_id"] = locked
        if "team_confidence" in out.columns:
            out.loc[mask, "team_confidence"] = mean_conf
        if "valid" in out.columns:
            out.loc[mask, "valid"] = True
        if "temporal_consistency" in out.columns:
            out.loc[mask, "temporal_consistency"] = 1.0
        if "confidence" in out.columns:
            out.loc[mask, "confidence"] = mean_conf
    return out


def lock_display_teams(
    identities: pd.DataFrame,
    stable_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Force one team per ``display_id`` and propagate to all member tracks."""
    if identities is None or identities.empty or stable_map is None or stable_map.empty:
        return identities, stable_map
    ident = identities.copy()
    stable = stable_map.copy()
    local_to_display = {
        int(r.local_track_id): int(r.display_id) for r in stable.itertuples(index=False)
    }
    ident["_display_id"] = ident["track_id"].map(local_to_display)

    display_team: dict[int, str] = {}
    for did, group in ident.dropna(subset=["_display_id"]).groupby("_display_id"):
        valid = group[group["team_id"].notna()]
        if valid.empty:
            continue
        if "team_confidence" in valid.columns:
            scores = valid.groupby(valid["team_id"].astype(str))["team_confidence"].sum()
        else:
            scores = valid.groupby(valid["team_id"].astype(str)).size()
        display_team[int(did)] = str(scores.idxmax())

    for did, team in display_team.items():
        members = [lid for lid, d in local_to_display.items() if d == did]
        mask = ident["track_id"].isin(members)
        ident.loc[mask, "team_id"] = team
        if "valid" in ident.columns:
            ident.loc[mask, "valid"] = True
        stable.loc[stable["display_id"] == did, "team_id"] = team

    if "_display_id" in ident.columns:
        ident = ident.drop(columns=["_display_id"])
    return ident, stable
