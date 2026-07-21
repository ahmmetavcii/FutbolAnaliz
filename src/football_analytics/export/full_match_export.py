"""Bundle a validated full-match run into interchange formats."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from football_analytics.export.csv_exporter import export_csv
from football_analytics.export.excel_exporter import export_excel_workbook
from football_analytics.export.json_exporter import export_json
from football_analytics.export.parquet_exporter import export_parquet
from football_analytics.export.tactical_map_exporter import export_tactical_map_video


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_dataframe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.DataFrame()


def _copy_if_present(src: Path, dst: Path, *, overwrite: bool) -> str | None:
    if not src.is_file():
        return None
    if dst.exists() and not overwrite:
        return str(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def export_full_match_results(
    run_dir: Path,
    output_dir: Path,
    *,
    formats: Sequence[str] | None = None,
    include_video: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export artifacts from *run_dir* into *output_dir*.

    Supported formats: ``canonical``, ``json``, ``csv``, ``parquet``, ``soccernet``.
    Missing optional model tables are exported as empty schemas rather than invented rows.
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = list(formats or ["canonical"])
    artifacts: dict[str, Any] = {"status": "PASS", "formats": selected, "files": []}

    report = _load_json(run_dir / "run_report.json")
    manifest = _load_json(run_dir / "match_manifest.json")
    if not manifest:
        manifest = _load_json(run_dir / "prepared" / "match_manifest.json")
    chunk_status = _load_json(run_dir / "chunk_status.json")
    events = _maybe_dataframe(run_dir / "events.parquet")
    if events.empty:
        events = _maybe_dataframe(run_dir / "exports" / "events.parquet")
    player_summary = _maybe_dataframe(run_dir / "player_summary.parquet")
    team_summary = _maybe_dataframe(run_dir / "team_summary.parquet")
    identities = _maybe_dataframe(run_dir / "global_identities.parquet")

    sheets: dict[str, pd.DataFrame] = {
        "Match Summary": pd.DataFrame(
            [
                {
                    "match_id": manifest.get("match_id") or report.get("match_id"),
                    "camera_count": manifest.get("camera_count"),
                    "status": report.get("status", "UNKNOWN"),
                }
            ]
        ),
        "Player Summary": player_summary if not player_summary.empty else pd.DataFrame(),
        "Goalkeeper Summary": pd.DataFrame(),
        "Team Summary": team_summary if not team_summary.empty else pd.DataFrame(),
        "Visibility Quality": pd.DataFrame(),
        "Camera Coverage": pd.DataFrame(),
        "Jersey Results": pd.DataFrame(),
        "Global Identity Mapping": identities if not identities.empty else pd.DataFrame(),
        "Identity Consistency": pd.DataFrame(),
        "Chunk Status": pd.DataFrame(chunk_status.get("chunks", [])),
        "Errors and Warnings": pd.DataFrame(report.get("errors", [])),
        "Configuration": pd.DataFrame([{"key": k, "value": str(v)} for k, v in (report.get("config") or {}).items()]),
        "Match Events": events if not events.empty else pd.DataFrame(),
        "Goals and Assists": events[events["event_type"].isin(["goal", "assist"])]
        if not events.empty and "event_type" in events.columns
        else pd.DataFrame(),
        "Shots": events[events["event_type"] == "shot"]
        if not events.empty and "event_type" in events.columns
        else pd.DataFrame(),
        "Substitutions": events[events["event_type"] == "substitution"]
        if not events.empty and "event_type" in events.columns
        else pd.DataFrame(),
        "Officials": pd.DataFrame(),
        "Manual Corrections": pd.DataFrame(),
    }

    if "canonical" in selected or "json" in selected:
        payload = {
            "match_manifest": manifest,
            "run_report": report,
            "chunk_status": chunk_status,
            "honesty": {
                "invented_events": False,
                "invented_identities": False,
                "note": "Empty tables mean evidence was unavailable, not zero real events.",
            },
        }
        result = export_json(output_dir / "full_match_export.json", payload)
        artifacts["files"].append(result)

    if "csv" in selected or "canonical" in selected:
        for name, frame in (
            ("events", events),
            ("player_summary", player_summary),
            ("team_summary", team_summary),
            ("global_identities", identities),
        ):
            target = output_dir / "csv" / f"{name}.csv"
            artifacts["files"].append(export_csv(target, frame))

    if "parquet" in selected or "canonical" in selected:
        for name, frame in (
            ("events", events),
            ("player_summary", player_summary),
            ("team_summary", team_summary),
            ("global_identities", identities),
        ):
            target = output_dir / "parquet" / f"{name}.parquet"
            artifacts["files"].append(export_parquet(target, frame))

    if "canonical" in selected:
        xlsx_path = output_dir / "full_match_report.xlsx"
        artifacts["files"].append(export_excel_workbook(xlsx_path, sheets))

    if "soccernet" in selected:
        sn_dir = output_dir / "soccernet"
        sn_dir.mkdir(parents=True, exist_ok=True)
        copied = _copy_if_present(
            run_dir / "soccernet_gamestate.json",
            sn_dir / "predictions.json",
            overwrite=overwrite,
        )
        artifacts["files"].append(
            {
                "path": copied or str(sn_dir / "predictions.json"),
                "status": "COPIED" if copied else "MISSING_SOURCE",
            }
        )
        if copied is None:
            export_json(
                sn_dir / "predictions.json",
                {
                    "predictions": [],
                    "note": "No SoccerNet GS predictions were present in the run directory.",
                },
            )

    if include_video:
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        annotated_src = run_dir / "analytics_annotated.mp4"
        if annotated_src.is_file():
            artifacts["files"].append(
                {
                    "path": _copy_if_present(
                        annotated_src,
                        video_dir / "annotated.mp4",
                        overwrite=overwrite,
                    )
                }
            )
        else:
            # Optional synthetic tactical map from positions if available.
            positions = _maybe_dataframe(run_dir / "game_state.parquet")
            if not positions.empty and "frame_id" in positions:
                frame_count = int(positions["frame_id"].max()) + 1
            else:
                frame_count = 30
            artifacts["files"].append(
                export_tactical_map_video(
                    video_dir / "tactical_map.mp4",
                    positions,
                    fps=25.0,
                    frame_count=frame_count,
                )
            )

    artifacts["output_dir"] = str(output_dir)
    return artifacts


export_run = export_full_match_results
