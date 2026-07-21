#!/usr/bin/env python3
"""Recompute match event summaries after manual corrections.

Does not re-run detection/tracking/ReID/calibration. Applies review actions to
``match_events.parquet`` / ``events.parquet`` and regenerates summary CSVs and
Excel sheets that depend on events.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_analytics.events.event_review import (  # noqa: E402
    Correction,
    CorrectionKind,
    ReviewLog,
    apply_review,
)
from football_analytics.events.schemas import EventStatus, EventType, MatchEvent  # noqa: E402
from football_analytics.events.event_summary import summarize_events  # noqa: E402
from football_analytics.export.excel_exporter import export_excel_workbook  # noqa: E402
from football_analytics.utils.io import write_json  # noqa: E402


def _events_from_parquet(path: Path) -> list[MatchEvent]:
    if not path.is_file():
        return []
    frame = pd.read_parquet(path)
    events: list[MatchEvent] = []
    for row in frame.itertuples(index=False):
        events.append(
            MatchEvent(
                event_id=str(row.event_id),
                event_type=EventType(str(row.event_type)),
                status=EventStatus(str(row.status)),
                timestamp_ms=float(row.timestamp_ms),
                team_id=None if pd.isna(getattr(row, "team_id", None)) else int(row.team_id),
                scorer_track_id=(
                    None
                    if pd.isna(getattr(row, "scorer_track_id", None))
                    else int(row.scorer_track_id)
                ),
                assist_track_id=(
                    None
                    if pd.isna(getattr(row, "assist_track_id", None))
                    else int(row.assist_track_id)
                ),
                confidence=float(getattr(row, "confidence", 0.0) or 0.0),
            )
        )
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--corrections",
        type=Path,
        default=None,
        help="JSON list of corrections; default <run-dir>/event_corrections.json",
    )
    args = parser.parse_args()
    run_dir = args.run_dir
    corrections_path = args.corrections or (run_dir / "event_corrections.json")

    events_path = run_dir / "events.parquet"
    if not events_path.is_file() and (run_dir / "match_events.parquet").is_file():
        events_path = run_dir / "match_events.parquet"
    events = _events_from_parquet(events_path)

    log = ReviewLog()
    if corrections_path.is_file():
        payload = json.loads(corrections_path.read_text(encoding="utf-8"))
        for item in payload.get("corrections", payload if isinstance(payload, list) else []):
            log.add(
                Correction(
                    event_id=str(item["event_id"]),
                    kind=CorrectionKind(str(item["kind"])),
                    value=item.get("value"),
                    attribute=item.get("attribute"),
                    note=str(item.get("note") or ""),
                    reviewer=str(item.get("reviewer") or ""),
                )
            )
    result = apply_review(events, log)
    updated = result.events

    rows = [
        {
            "event_id": e.event_id,
            "event_type": e.event_type.value,
            "status": e.status.value,
            "timestamp_ms": e.timestamp_ms,
            "team_id": e.team_id,
            "scorer_track_id": e.scorer_track_id,
            "assist_track_id": e.assist_track_id,
            "confidence": e.confidence,
        }
        for e in updated
    ]
    frame = pd.DataFrame(rows)
    frame.to_parquet(run_dir / "events.parquet", index=False)
    if (run_dir / "match_events.parquet").is_file() or True:
        frame.to_parquet(run_dir / "match_events.parquet", index=False)

    summary = summarize_events(updated)
    confirmed = sum(1 for e in updated if e.status.value in {"auto_confirmed", "manually_confirmed"})
    candidates = sum(1 for e in updated if e.status.value == "candidate_review_required")
    pd.DataFrame(
        [
            {
                "confirmed_goals": sum(summary.confirmed_goals_by_team.values()),
                "pending_review": len(summary.pending_review_event_ids),
                "unresolved": len(summary.unresolved_event_ids),
            }
        ]
    ).to_csv(run_dir / "player_event_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "team_id": team_id,
                "goals_confirmed": count,
            }
            for team_id, count in summary.confirmed_goals_by_team.items()
        ]
    ).to_csv(run_dir / "team_event_summary.csv", index=False)

    quality_path = run_dir / "quality_report.json"
    if quality_path.is_file():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        quality["confirmed_events"] = confirmed
        quality["candidate_events"] = candidates
        write_json(quality_path, quality)

    # Refresh Excel event sheets when workbook exists.
    excel_path = run_dir / "full_match_report.xlsx"
    if excel_path.is_file():
        from openpyxl import load_workbook

        # Keep other sheets; rewrite event-related ones via exporter merge is heavy.
        # Minimal: rewrite Match Events sheet content using export helper on a stub.
        sheets = {
            name: pd.DataFrame()
            for name in (
                "Match Summary",
                "Player Summary",
                "Goalkeeper Summary",
                "Team Summary",
                "Visibility Quality",
                "Camera Coverage",
                "Jersey Results",
                "Global Identity Mapping",
                "Identity Consistency",
                "Chunk Status",
                "Errors and Warnings",
                "Configuration",
                "Match Events",
                "Goals and Assists",
                "Shots",
                "Substitutions",
                "Officials",
                "Manual Corrections",
            )
        }
        # Prefer preserving existing non-event sheets when possible.
        try:
            existing = pd.read_excel(excel_path, sheet_name=None)
            for name, value in existing.items():
                if name in sheets:
                    sheets[name] = value
        except Exception:  # noqa: BLE001
            pass
        sheets["Match Events"] = frame
        sheets["Goals and Assists"] = frame[
            frame["event_type"].isin(["goal", "assist"])
        ] if not frame.empty else frame
        sheets["Shots"] = frame[frame["event_type"].eq("shot")] if not frame.empty else frame
        sheets["Manual Corrections"] = pd.DataFrame(
            [
                {
                    "correction_id": index,
                    "kind": c.kind.value,
                    "note": c.note,
                    "event_id": c.event_id,
                }
                for index, c in enumerate(log.corrections)
            ]
        )
        export_excel_workbook(excel_path, sheets)

    write_json(
        run_dir / "recompute_match_events_report.json",
        {
            "status": "PASS",
            "confirmed_events": confirmed,
            "candidate_events": candidates,
            "corrections_applied": len(log.corrections),
            "detection_rerun": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "confirmed_events": confirmed,
                "candidate_events": candidates,
                "corrections_applied": len(log.corrections),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
