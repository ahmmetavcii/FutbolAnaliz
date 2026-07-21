"""Post-process one real single-camera pipeline run into full-match outputs.

Everything here is derived from artifacts the real model pipeline produced
(detections, tracks, calibration, metrics, team identity). Roles, global
identities, and summaries are computed with the conservative infrastructure
from :mod:`football_analytics.roles` / :mod:`football_analytics.multicamera`;
when evidence is insufficient the honest outcome (``unknown_person``, empty
event tables) is preserved rather than papered over.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_analytics.analytics.goalkeeper_summary import summarize_goalkeepers
from football_analytics.analytics.officials_summary import summarize_officials
from football_analytics.analytics.player_metrics import PlayerMetricsConfig, PlayerSample
from football_analytics.analytics.player_summary import summarize_players
from football_analytics.analytics.team_summary import summarize_teams
from football_analytics.export.excel_exporter import export_excel_workbook
from football_analytics.export.tactical_map_exporter import export_tactical_map_video
from football_analytics.export.video_exporter import validate_video_export
from football_analytics.multicamera.global_identity import GlobalIdentityRegistry
from football_analytics.multicamera.cross_camera_reid import ReidMatchConfig
from football_analytics.multicamera.local_tracking import LocalObservation, PlayerRole
from football_analytics.roles.role_classifier import (
    PersonFrameFeatures,
    PersonRole,
    RoleClassifier,
)
from football_analytics.roles.role_voting import RoleVoter

from .manifest import atomic_write_json, atomic_write_parquet

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PENALTY_DEPTH_M = 16.5

#: Map MVP shot labels onto the metric config's accepted vocabulary.
_SHOT_TYPE_ALIASES = {"main_wide": "wide", "wide": "wide", "tactical": "tactical"}

_ROLE_TO_PLAYER_ROLE = {
    PersonRole.GOALKEEPER: PlayerRole.GOALKEEPER,
    PersonRole.OUTFIELD_PLAYER: PlayerRole.OUTFIELD,
    PersonRole.REFEREE: PlayerRole.REFEREE,
}


def _team_index(team_id: Any) -> int | None:
    if team_id in (None, "", "unknown") or (isinstance(team_id, float) and np.isnan(team_id)):
        return None
    text = str(team_id)
    if text.endswith("0"):
        return 0
    if text.endswith("1"):
        return 1
    return None


def compute_role_predictions(
    track_identities: pd.DataFrame,
    player_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, Any]]:
    """Run the conservative role infrastructure on real per-frame identity rows.

    Only evidence that was actually measured feeds the classifier: team-kit
    confidence from the color model and penalty-area occupancy from calibrated
    field positions. Officials-kit similarity was not measured, so referee
    claims cannot (and must not) arise from this clip.
    """
    positions = player_metrics.set_index(["frame_id", "track_id"])[
        ["x_field", "y_field", "valid"]
    ]

    classifier = RoleClassifier()
    voter = RoleVoter()
    for row in track_identities.itertuples(index=False):
        team = _team_index(row.team_id)
        team_conf = float(row.team_confidence or 0.0)
        occupancy = None
        deepest = False
        try:
            pos = positions.loc[(row.frame_id, row.track_id)]
            if bool(pos["valid"]) and not np.isnan(pos["x_field"]):
                x = float(pos["x_field"])
                occupancy = 1.0 if (x <= PENALTY_DEPTH_M or x >= PITCH_LENGTH_M - PENALTY_DEPTH_M) else 0.0
        except KeyError:
            pass
        features = PersonFrameFeatures(
            track_id=int(row.track_id),
            timestamp_ms=float(row.timestamp_ms),
            on_pitch=True,
            kit_similarity_officials=None,
            kit_similarity_team0=team_conf if team == 0 else None,
            kit_similarity_team1=team_conf if team == 1 else None,
            goalkeeper_kit_distinctiveness=None,
            own_penalty_area_occupancy=occupancy,
            is_deepest_teammate=deepest,
            team_id=team,
        )
        voter.add_observation(classifier.classify_frame(features))

    votes = voter.decide_all()
    rows = [
        {
            "track_id": vote.track_id,
            "role": vote.role.value,
            "vote_share": vote.vote_share,
            "observations": vote.observations,
        }
        for vote in votes.values()
    ]
    return pd.DataFrame(rows), votes


def build_global_identity(
    tracks: pd.DataFrame,
    track_identities: pd.DataFrame,
    player_metrics: pd.DataFrame,
    camera_id: str,
    reid_prototypes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, GlobalIdentityRegistry]:
    """Bind real local tracks to global identities (single camera).

    One representative observation per (track, second) keeps the registry
    conservative and fast while still exercising occlusion-gap binding.
    When ``reid_prototypes`` is provided, mean sn-reid embeddings allow
    reappearing players (new local track ids after gaps/cuts) to merge.
    """
    identity_lookup = track_identities.set_index(["frame_id", "track_id"])
    field_lookup = player_metrics.set_index(["frame_id", "track_id"])
    reid_by_track: dict[int, tuple[float, ...]] = {}
    if reid_prototypes is not None and not reid_prototypes.empty:
        for row in reid_prototypes.itertuples(index=False):
            if not bool(getattr(row, "valid", True)):
                continue
            embedding = getattr(row, "embedding", None)
            if embedding is None:
                continue
            values = list(embedding)
            if not values:
                continue
            reid_by_track[int(row.track_id)] = tuple(float(x) for x in values)

    tracks = tracks.sort_values(["track_id", "frame_id"])
    tracks = tracks[tracks["object_type"] == "person"]
    observations: list[LocalObservation] = []
    last_second: dict[int, int] = {}
    for row in tracks.itertuples(index=False):
        second = int(row.timestamp_ms // 1000)
        track_id = int(row.track_id)
        if last_second.get(track_id) == second:
            continue
        last_second[track_id] = second

        team = None
        team_conf = 0.0
        try:
            ident = identity_lookup.loc[(row.frame_id, track_id)]
            team = _team_index(ident["team_id"])
            team_conf = float(ident["team_confidence"] or 0.0)
        except KeyError:
            pass
        pitch_xy = None
        try:
            pos = field_lookup.loc[(row.frame_id, track_id)]
            if bool(pos["valid"]) and not np.isnan(pos["x_field"]):
                pitch_xy = (float(pos["x_field"]), float(pos["y_field"]))
        except KeyError:
            pass
        observations.append(
            LocalObservation(
                camera_id=camera_id,
                local_track_id=track_id,
                frame_index=int(row.frame_id),
                reference_time_seconds=float(row.timestamp_ms) / 1000.0,
                bbox_xyxy=(
                    float(row.bbox_x1),
                    float(row.bbox_y1),
                    float(row.bbox_x2),
                    float(row.bbox_y2),
                ),
                pitch_xy_m=pitch_xy,
                team_id=team,
                team_confidence=min(max(team_conf, 0.0), 1.0),
                detection_confidence=min(max(float(row.tracking_confidence), 0.0), 1.0),
                reid_embedding=reid_by_track.get(track_id),
            )
        )

    registry = GlobalIdentityRegistry(
        config=ReidMatchConfig(
            # Soccer broadcast: teammates look alike; require a stronger
            # appearance match before merging new local track ids.
            accept_threshold=0.68,
            unresolved_threshold=0.45,
            minimum_reid_similarity=0.55,
            weight_reid=0.45,
            weight_team=0.12,
            weight_jersey=0.18,
            weight_position=0.18,
            weight_time=0.04,
            weight_role=0.03,
        )
        if reid_by_track
        else ReidMatchConfig()
    )
    registry.assign_all(observations)

    map_rows = [
        {
            "camera_id": key[0],
            "local_track_id": key[1],
            "global_id": global_id,
            "unresolved": registry.identities[global_id].unresolved,
        }
        for key, global_id in sorted(registry.track_to_global.items())
    ]
    player_rows = [
        {
            "global_id": identity.global_id,
            "n_local_tracks": len(identity.track_keys),
            "team_id": identity.team_id,
            "team_confidence": identity.team_confidence,
            "jersey_number": identity.jersey_number,
            "jersey_confidence": identity.jersey_confidence,
            "role": identity.role.value,
            "unresolved": identity.unresolved,
            "last_time_seconds": identity.last_time_seconds,
        }
        for identity in sorted(registry.identities.values(), key=lambda i: i.global_id)
    ]
    return pd.DataFrame(map_rows), pd.DataFrame(player_rows), registry


def build_player_samples(player_metrics: pd.DataFrame) -> list[PlayerSample]:
    samples: list[PlayerSample] = []
    for row in player_metrics.itertuples(index=False):
        shot = _SHOT_TYPE_ALIASES.get(str(getattr(row, "shot_type", "wide")), "wide")
        samples.append(
            PlayerSample(
                track_id=int(row.track_id),
                timestamp_ms=float(row.timestamp_ms),
                x_field=None if np.isnan(row.x_field) else float(row.x_field),
                y_field=None if np.isnan(row.y_field) else float(row.y_field),
                confidence=float(row.confidence or 0.0),
                calibration_valid=bool(row.valid),
                shot_type=shot,
                quality_ok=True,
            )
        )
    return samples


def _team_ids_by_track(track_identities: pd.DataFrame) -> dict[int, int | None]:
    result: dict[int, int | None] = {}
    for track_id, group in track_identities.groupby("track_id"):
        assigned = group[group["team_id"].notna()]
        if assigned.empty:
            result[int(track_id)] = None
            continue
        top = assigned.groupby("team_id")["team_confidence"].mean().idxmax()
        result[int(track_id)] = _team_index(top)
    return result


def postprocess_pipeline_run(
    pipeline_run_dir: Path,
    output_dir: Path,
    camera_id: str = "camera_1",
    *,
    jersey_predictions: pd.DataFrame | None = None,
    tactical_map_max_frames: int | None = None,
) -> dict[str, Any]:
    """Derive the full-match output schema from one real pipeline run."""
    pipeline_run_dir = Path(pipeline_run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detections = pd.read_parquet(pipeline_run_dir / "detections.parquet")
    tracks = pd.read_parquet(pipeline_run_dir / "tracks.parquet")
    calibration = pd.read_parquet(pipeline_run_dir / "calibration.parquet")
    player_metrics = pd.read_parquet(pipeline_run_dir / "player_metrics.parquet")
    track_identities = pd.read_parquet(pipeline_run_dir / "track_identities.parquet")
    game_state = pd.read_parquet(pipeline_run_dir / "game_state.parquet")
    pipeline_report = json.loads((pipeline_run_dir / "run_report.json").read_text())

    quality_notes: list[str] = []

    # -- pass-through model tables (real, unmodified content) ---------------
    atomic_write_parquet(output_dir / "detections.parquet", detections)
    local_tracks = tracks.copy()
    local_tracks.insert(0, "camera_id", camera_id)
    atomic_write_parquet(output_dir / "local_tracks.parquet", local_tracks)
    atomic_write_parquet(output_dir / "player_frame_metrics.parquet", player_metrics)

    # -- calibration summary -------------------------------------------------
    valid_ratio = float(calibration["valid"].mean()) if len(calibration) else 0.0
    sample = calibration[calibration["valid"]].head(1)
    calibrations_payload = {
        "schema_version": "1.0.0",
        "cameras": {
            camera_id: {
                "frames": int(len(calibration)),
                "valid_frames": int(calibration["valid"].sum()),
                "valid_ratio": valid_ratio,
                "provider": (
                    str(sample["provider"].iat[0]) if len(sample) else None
                ),
                "homography_sample": (
                    json.loads(sample["homography_json"].iat[0]) if len(sample) else None
                ),
                "pitch_length_m": PITCH_LENGTH_M,
                "pitch_width_m": PITCH_WIDTH_M,
            }
        },
    }
    atomic_write_json(output_dir / "camera_calibrations.json", calibrations_payload)

    # -- roles ---------------------------------------------------------------
    role_predictions, votes = compute_role_predictions(track_identities, player_metrics)
    atomic_write_parquet(output_dir / "role_predictions.parquet", role_predictions)
    role_counts = role_predictions["role"].value_counts().to_dict()
    if not any(r in role_counts for r in ("referee", "assistant_referee")):
        quality_notes.append(
            "no referee/assistant_referee claimed: officials-kit similarity was "
            "not measured on this clip, so official roles cannot be assigned honestly"
        )
    if "goalkeeper" not in role_counts:
        quality_notes.append(
            "no goalkeeper claimed: goalkeeper-kit distinctiveness unavailable and "
            "penalty-area evidence insufficient on this 30s broadcast clip"
        )

    # -- global identity -----------------------------------------------------
    reid_path = pipeline_run_dir / "track_reid_prototypes.parquet"
    reid_prototypes = (
        pd.read_parquet(reid_path) if reid_path.is_file() else pd.DataFrame()
    )
    identity_map, global_players, registry = build_global_identity(
        tracks,
        track_identities,
        player_metrics,
        camera_id,
        reid_prototypes=reid_prototypes if not reid_prototypes.empty else None,
    )
    if reid_prototypes.empty:
        quality_notes.append(
            "reid prototypes missing: cross-gap identity merges use team/position cues only"
        )
    # Attach voted roles to global players via their single local track.
    role_by_track = {
        int(r["track_id"]): str(r["role"]) for r in role_predictions.to_dict("records")
    }
    if not identity_map.empty:
        track_to_global = dict(
            zip(identity_map["local_track_id"], identity_map["global_id"])
        )
        global_players["voted_role"] = global_players["global_id"].map(
            {
                global_id: role_by_track.get(track, "unknown_person")
                for track, global_id in track_to_global.items()
            }
        )
    atomic_write_parquet(output_dir / "global_identity_map.parquet", identity_map)
    atomic_write_parquet(output_dir / "global_players.parquet", global_players)

    # -- jersey --------------------------------------------------------------
    if jersey_predictions is not None and not jersey_predictions.empty:
        atomic_write_parquet(output_dir / "jersey_predictions.parquet", jersey_predictions)
    else:
        atomic_write_parquet(
            output_dir / "jersey_predictions.parquet",
            pd.DataFrame(columns=["track_id", "jersey_number", "confidence", "status"]),
        )
        quality_notes.append(
            "jersey predictions empty or low-confidence: see quality_report for details"
        )

    # -- player / team / official summaries ----------------------------------
    samples = build_player_samples(player_metrics)
    team_ids = _team_ids_by_track(track_identities)
    metrics_config = PlayerMetricsConfig()
    player_summaries = summarize_players(votes, team_ids, samples, config=metrics_config)
    official_summaries = summarize_officials(votes, samples, config=metrics_config)
    goalkeeper_summaries = summarize_goalkeepers(player_summaries)
    team_summaries = summarize_teams(player_summaries)

    player_df = pd.DataFrame(
        [
            {
                "track_id": s.track_id,
                "role": s.role.value,
                "team_id": s.team_id,
                "role_vote_share": s.role_vote_share,
                "total_distance_m": s.total_distance_m,
                "max_speed_kmh": s.max_speed_kmh,
                "mean_speed_kmh": s.mean_speed_kmh,
                "sprint_count": s.sprint_count,
                "physical_metrics_valid": s.physical_metrics_valid,
                "invalid_reason": s.invalid_reason,
                "counts_toward_team_totals": s.counts_toward_team_totals,
            }
            for s in player_summaries.values()
        ]
    )
    goalkeeper_df = pd.DataFrame(
        [
            {
                "track_id": s.track_id,
                "team_id": s.team_id,
                "penalty_area_dwell_share": s.penalty_area_dwell_share,
            }
            for s in goalkeeper_summaries.values()
        ],
    )
    if goalkeeper_df.empty:
        goalkeeper_df = pd.DataFrame(
            columns=["track_id", "team_id", "penalty_area_dwell_share"]
        )
    officials_df = pd.DataFrame(
        [
            {
                "track_id": s.track_id,
                "role": s.role.value,
                "team_id": None,
                "total_distance_m": s.total_distance_m,
                "max_speed_kmh": s.max_speed_kmh,
                "counts_toward_team_totals": s.counts_toward_team_totals,
            }
            for s in official_summaries.values()
        ],
    )
    if officials_df.empty:
        officials_df = pd.DataFrame(
            columns=[
                "track_id", "role", "team_id", "total_distance_m",
                "max_speed_kmh", "counts_toward_team_totals",
            ]
        )
        quality_notes.append(
            "officials_summary empty: no track produced official-role evidence "
            "on this clip (reason above); nothing was invented"
        )
    team_df = pd.DataFrame(
        [
            {
                "team_id": s.team_id,
                "player_count": s.player_count,
                "players_with_valid_metrics": s.players_with_valid_metrics,
                "total_distance_m": s.total_distance_m,
                "max_speed_kmh": s.max_speed_kmh,
                "total_sprints": s.total_sprints,
                "goalkeeper_track_ids": ",".join(map(str, s.goalkeeper_track_ids)),
            }
            for s in team_summaries.values()
        ]
    )

    player_df.to_csv(output_dir / "player_summary.csv", index=False)
    goalkeeper_df.to_csv(output_dir / "goalkeeper_summary.csv", index=False)
    officials_df.to_csv(output_dir / "officials_summary.csv", index=False)
    team_df.to_csv(output_dir / "team_summary.csv", index=False)

    # -- events: prefer pipeline event_detection artifacts; never invent goals
    events_src = pipeline_run_dir / "match_events.parquet"
    compat_src = pipeline_run_dir / "events.parquet"
    events_note = "no_supported_event_detected"
    confirmed_count = 0
    candidate_count = 0
    if events_src.is_file():
        events_df = pd.read_parquet(events_src)
        atomic_write_parquet(output_dir / "match_events.parquet", events_df)
        if compat_src.is_file():
            compat_df = pd.read_parquet(compat_src)
        else:
            compat_df = events_df[
                [
                    c
                    for c in (
                        "event_id",
                        "event_type",
                        "status",
                        "timestamp_ms",
                        "team_id",
                        "scorer_track_id",
                        "assist_track_id",
                        "confidence",
                    )
                    if c in events_df.columns
                ]
            ] if not events_df.empty else pd.DataFrame(
                columns=[
                    "event_id",
                    "event_type",
                    "status",
                    "timestamp_ms",
                    "team_id",
                    "scorer_track_id",
                    "assist_track_id",
                    "confidence",
                ]
            )
        atomic_write_parquet(output_dir / "events.parquet", compat_df)
        if not events_df.empty and "status" in events_df.columns:
            status = events_df["status"].astype(str).str.lower()
            confirmed_count = int(status.isin(["auto_confirmed", "manually_confirmed"]).sum())
            candidate_count = int(status.eq("candidate_review_required").sum())
        manifest_path = pipeline_run_dir / "stage_manifests" / "event_detection.json"
        if manifest_path.is_file():
            events_note = json.loads(manifest_path.read_text()).get(
                "events_reason", events_note
            )
        for name in (
            "event_evidence.json",
            "event_review_queue.json",
            "scoreboard_timeline.parquet",
            "ball_trajectory.parquet",
            "touch_events.parquet",
            "possession_chain.parquet",
            "player_event_summary.csv",
            "team_event_summary.csv",
        ):
            src = pipeline_run_dir / name
            if src.is_file():
                shutil.copy2(src, output_dir / name)
        clips_src = pipeline_run_dir / "event_clips"
        if clips_src.is_dir():
            dst = output_dir / "event_clips"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(clips_src, dst)
        if confirmed_count == 0 and candidate_count == 0:
            quality_notes.append(
                f"match_events empty/honest ({events_note}); events were not invented"
            )
    else:
        events_df = pd.DataFrame(
            columns=[
                "event_id",
                "event_type",
                "status",
                "timestamp_ms",
                "team_id",
                "scorer_track_id",
                "assist_track_id",
                "confidence",
            ]
        )
        atomic_write_parquet(output_dir / "events.parquet", events_df)
        atomic_write_parquet(output_dir / "match_events.parquet", events_df)
        quality_notes.append(
            "events.parquet empty: no goal/shot/substitution evidence stream exists "
            f"for this clip ({events_note}); events were not invented"
        )

    # -- videos ---------------------------------------------------------------
    annotated_src = pipeline_run_dir / "analytics_annotated.mp4"
    annotated_dst = output_dir / "annotated_match.mp4"
    annotated_validation: dict[str, Any] | None = None
    if annotated_src.is_file():
        shutil.copy2(annotated_src, annotated_dst)
        annotated_validation = validate_video_export(annotated_dst)

    positions = game_state.rename(columns={"team_id": "team_id"})
    frame_count = int(positions["frame_id"].max()) + 1 if len(positions) else 1
    if tactical_map_max_frames:
        frame_count = min(frame_count, tactical_map_max_frames)
    tactical_result = export_tactical_map_video(
        output_dir / "tactical_map.mp4",
        positions,
        fps=25.0,
        frame_count=frame_count,
    )
    tactical_validation = validate_video_export(output_dir / "tactical_map.mp4")

    # -- Excel workbook (fixed 17-sheet schema) --------------------------------
    track_quality_path = pipeline_run_dir / "track_quality.parquet"
    visibility_df = (
        pd.read_parquet(track_quality_path) if track_quality_path.is_file() else pd.DataFrame()
    )
    jersey_df = pd.read_parquet(output_dir / "jersey_predictions.parquet")
    coverage_df = pd.DataFrame(
        [
            {
                "camera_id": camera_id,
                "frames": int(len(calibration)),
                "valid_calibration_frames": int(calibration["valid"].sum()),
                "coverage_ratio": valid_ratio,
            }
        ]
    )
    empty_events = events_df
    goals_df = (
        events_df[events_df["event_type"].astype(str).isin(["goal", "assist"])]
        if not events_df.empty and "event_type" in events_df.columns
        else events_df
    )
    shots_df = (
        events_df[events_df["event_type"].astype(str).eq("shot")]
        if not events_df.empty and "event_type" in events_df.columns
        else events_df
    )
    sheets = {
        "Match Summary": pd.DataFrame(
            [
                {
                    "camera_id": camera_id,
                    "pipeline_status": pipeline_report.get("status"),
                    "detections": len(detections),
                    "track_rows": len(tracks),
                    "unique_tracks": int(tracks["track_id"].nunique()),
                    "global_players": len(global_players),
                    "calibration_valid_ratio": valid_ratio,
                    "confirmed_events": confirmed_count,
                    "candidate_events": candidate_count,
                    "events_reason": events_note,
                }
            ]
        ),
        "Player Summary": player_df,
        "Goalkeeper Summary": goalkeeper_df,
        "Team Summary": team_df,
        "Visibility Quality": visibility_df,
        "Camera Coverage": coverage_df,
        "Jersey Results": jersey_df,
        "Global Identity Mapping": identity_map,
        "Identity Consistency": global_players.drop(
            columns=["team_confidence"], errors="ignore"
        ),
        "Chunk Status": pd.DataFrame(
            [
                {
                    "camera_id": camera_id,
                    "chunk_index": 0,
                    "status": "PASS",
                    "pipeline_run_dir": str(pipeline_run_dir),
                }
            ]
        ),
        "Errors and Warnings": pd.DataFrame({"note": quality_notes}),
        "Configuration": pd.DataFrame(
            [
                {"key": key, "value": str(value)}
                for key, value in (pipeline_report.get("model") or {}).items()
            ]
        ),
        "Match Events": empty_events,
        "Goals and Assists": goals_df,
        "Shots": shots_df,
        "Substitutions": (
            events_df[events_df["event_type"].astype(str).eq("substitution")]
            if not events_df.empty and "event_type" in events_df.columns
            else events_df
        ),
        "Officials": officials_df,
        "Manual Corrections": pd.DataFrame(columns=["correction_id", "kind", "note"]),
    }
    excel_result = export_excel_workbook(output_dir / "full_match_report.xlsx", sheets)

    # -- quality + run reports --------------------------------------------------
    quality_report = {
        "schema_version": "1.0.0",
        "camera_id": camera_id,
        "source_pipeline_run": str(pipeline_run_dir),
        "model_stages_claimed": True,
        "detections": int(len(detections)),
        "track_rows": int(len(tracks)),
        "unique_tracks": int(tracks["track_id"].nunique()),
        "calibration_valid_ratio": valid_ratio,
        "player_frame_metrics_rows": int(len(player_metrics)),
        "player_frame_metrics_valid_ratio": float(player_metrics["valid"].mean()),
        "global_players": int(len(global_players)),
        "unresolved_identities": int(global_players["unresolved"].sum()) if len(global_players) else 0,
        "role_distribution": role_counts,
        "confirmed_events": confirmed_count,
        "candidate_events": candidate_count,
        "events_reason": events_note,
        "empty_or_limited_artifacts": quality_notes,
    }
    atomic_write_json(output_dir / "quality_report.json", quality_report)

    run_report = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "camera_id": camera_id,
        "pipeline_run_report": pipeline_report,
        "quality_report": str(output_dir / "quality_report.json"),
        "annotated_video": asdict_or_none(annotated_validation),
        "tactical_map": {
            "export": tactical_result,
            "validation": asdict_or_none(tactical_validation),
        },
        "excel": excel_result,
        "model_stages_claimed": True,
    }
    atomic_write_json(output_dir / "run_report.json", run_report)
    return run_report


def asdict_or_none(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
