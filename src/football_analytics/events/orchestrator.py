"""End-to-end event detection orchestrator over existing pipeline artifacts."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from football_analytics.events.assist_detector import AssistDetector, AssistDetectorConfig
from football_analytics.events.ball_trajectory import (
    BallTrajectoryConfig,
    load_ball_trajectory_from_ball_state,
    write_ball_trajectory,
)
from football_analytics.events.duplicate_suppression import suppress_replay_duplicates
from football_analytics.events.event_clips import ClipConfig, build_clips, export_clip_mp4
from football_analytics.events.event_detector import EventDetectorConfig
from football_analytics.events.event_summary import summarize_events
from football_analytics.events.goal_detector import GoalDetector, GoalDetectorConfig, GoalSignals
from football_analytics.events.possession_chain import (
    PossessionChainConfig,
    build_possession_chain,
    passes_from_chain,
)
from football_analytics.events.schemas import EventStatus, EventType, MatchEvent, is_confirmed
from football_analytics.events.scoreboard_ocr import ScoreboardOCRConfig, run_scoreboard_ocr
from football_analytics.events.scorer_detector import ScorerDetectorConfig, detect_scorer
from football_analytics.events.shot_detector import ShotDetector, ShotDetectorConfig, ShotSignals
from football_analytics.events.spotting_adapter import SpottingAdapter, SpottingAdapterConfig
from football_analytics.events.touch_inference import TouchInferenceConfig, infer_touches
from football_analytics.utils.io import write_json


MATCH_EVENT_COLUMNS = [
    "event_id",
    "event_type",
    "period",
    "match_second",
    "video_timestamp",
    "team_id",
    "primary_global_player_id",
    "secondary_global_player_id",
    "scorer_global_player_id",
    "assist_global_player_id",
    "shooter_global_player_id",
    "primary_jersey_number",
    "secondary_jersey_number",
    "confidence",
    "status",
    "source_models",
    "source_camera_ids",
    "source_frame_start",
    "source_frame_end",
    "evidence_json",
    "manually_reviewed",
    "reviewer_action",
    "created_at",
    "timestamp_ms",
    "scorer_track_id",
    "assist_track_id",
]


def _default_config() -> dict[str, Any]:
    path = Path("/home/ahmet/projects/football-analytics/configs/events/goal_detection.yaml")
    if path.is_file():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def _team_index(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    if text.endswith("0"):
        return 0
    if text.endswith("1"):
        return 1
    return None


def _spotting_goal_score(candidates: list[Any], timestamp_ms: float, window_ms: float = 2500.0) -> float | None:
    best = 0.0
    found = False
    for item in candidates:
        if str(item.event_type).lower() != "goal":
            continue
        ts_ms = float(item.timestamp) * 1000.0
        if abs(ts_ms - timestamp_ms) <= window_ms:
            best = max(best, float(item.confidence))
            found = True
    return best if found else None


def _goal_line_candidates(ball: pd.DataFrame, cfg: dict[str, Any]) -> list[GoalSignals]:
    if ball is None or ball.empty:
        return []
    pitch_len = float(cfg.get("pitch_length_m", 105.0))
    tol = float(cfg.get("goal_line_tolerance_m", 1.0))
    half_w = float(cfg.get("goal_mouth_half_width_m", 3.66))
    center_y = 34.0
    signals: list[GoalSignals] = []
    ordered = ball.sort_values("frame_id")
    prev = None
    for row in ordered.itertuples(index=False):
        if not bool(getattr(row, "visible", False)):
            prev = row
            continue
        px = float(getattr(row, "pitch_x", float("nan")))
        py = float(getattr(row, "pitch_y", float("nan")))
        if px != px or py != py:
            prev = row
            continue
        crossed = False
        if prev is not None:
            ppx = float(getattr(prev, "pitch_x", float("nan")))
            if ppx == ppx:
                if (ppx < pitch_len - tol <= px) or (ppx > tol >= px):
                    crossed = abs(py - center_y) <= half_w + 1.0
        if crossed:
            signals.append(
                GoalSignals(
                    timestamp_ms=float(row.timestamp_ms),
                    ball_crossed_line_score=0.55,
                    from_replay=False,
                )
            )
        prev = row
    return signals


def match_events_frame(events: list[MatchEvent], *, created_at: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        attrs = dict(event.attributes)
        frame_start = int(attrs.get("source_frame_start", max(0, event.timestamp_ms / 40.0)))
        frame_end = int(attrs.get("source_frame_end", frame_start))
        rows.append(
            {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "period": attrs.get("period"),
                "match_second": float(event.timestamp_ms) / 1000.0,
                "video_timestamp": float(event.timestamp_ms) / 1000.0,
                "team_id": event.team_id,
                "primary_global_player_id": attrs.get("scorer_global_player_id"),
                "secondary_global_player_id": attrs.get("assist_global_player_id"),
                "scorer_global_player_id": attrs.get("scorer_global_player_id"),
                "assist_global_player_id": attrs.get("assist_global_player_id"),
                "shooter_global_player_id": attrs.get("shooter_global_player_id"),
                "primary_jersey_number": attrs.get("scorer_jersey_number"),
                "secondary_jersey_number": attrs.get("assist_jersey_number"),
                "confidence": float(event.confidence),
                "status": event.status.value,
                "source_models": json.dumps(attrs.get("source_models", [])),
                "source_camera_ids": json.dumps(attrs.get("source_camera_ids", ["camera_1"])),
                "source_frame_start": frame_start,
                "source_frame_end": frame_end,
                "evidence_json": json.dumps(
                    [
                        {
                            "source": item.source,
                            "score": item.score,
                            "timestamp_ms": item.timestamp_ms,
                            "from_replay": item.from_replay,
                            "description": item.description,
                        }
                        for item in event.evidence.items
                    ]
                ),
                "manually_reviewed": False,
                "reviewer_action": None,
                "created_at": created_at,
                "timestamp_ms": float(event.timestamp_ms),
                "scorer_track_id": event.scorer_track_id,
                "assist_track_id": event.assist_track_id,
            }
        )
    if not rows:
        return pd.DataFrame(columns=MATCH_EVENT_COLUMNS)
    return pd.DataFrame(rows, columns=MATCH_EVENT_COLUMNS)


def _section(cfg: dict[str, Any], name: str, cls: type) -> Any:
    raw = cfg.get(name) or {}
    return cls(**{key: raw[key] for key in cls.__dataclass_fields__ if key in raw})


def run_event_detection(
    run_dir: Path,
    *,
    config: dict[str, Any] | None = None,
    video_path: Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = _default_config()
    if config:
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                merged = dict(cfg[key])
                merged.update(value)
                cfg[key] = merged
            else:
                cfg[key] = value

    started = dt.datetime.now(dt.timezone.utc)
    video = Path(video_path) if video_path else run_dir / "input" / "test_clip.mp4"
    stage_dir = run_dir / "stages" / "event_detection"
    stage_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = run_dir / "event_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    def _read(name: str) -> pd.DataFrame:
        path = run_dir / name
        return pd.read_parquet(path) if path.is_file() else pd.DataFrame()

    ball_state = _read("ball_state.parquet")
    tracks = _read("tracks.parquet")
    identities = _read("track_identities.parquet")
    shots = _read("shot_segments.parquet")
    global_map = _read("global_identity_map.parquet")

    spotting_cfg = _section(cfg, "spotting", SpottingAdapterConfig)
    adapter = SpottingAdapter(spotting_cfg)
    spotting_json = stage_dir / "spotting_predictions.json"
    spotting_candidates: list[Any] = []
    spotting_error = None
    try:
        if video.is_file() and spotting_cfg.enabled:
            spotting_candidates = adapter.predict_events(video, spotting_json)
            adapter.validate_outputs(spotting_candidates)
    except Exception as exc:  # noqa: BLE001
        spotting_error = str(exc)
        spotting_candidates = []
        write_json(spotting_json, {"status": "ERROR", "error": spotting_error, "candidates": []})

    ball_traj = load_ball_trajectory_from_ball_state(
        ball_state, config=_section(cfg, "ball_trajectory", BallTrajectoryConfig)
    )
    write_ball_trajectory(run_dir / "ball_trajectory.parquet", ball_traj)
    touches = infer_touches(
        ball_traj,
        tracks,
        identities=identities,
        global_map=global_map if not global_map.empty else None,
        config=_section(cfg, "touch_inference", TouchInferenceConfig),
    )
    touches.to_parquet(run_dir / "touch_events.parquet", index=False)
    chain = build_possession_chain(
        touches, config=_section(cfg, "possession_chain", PossessionChainConfig)
    )
    chain.to_parquet(run_dir / "possession_chain.parquet", index=False)
    scoreboard = run_scoreboard_ocr(
        video, config=_section(cfg, "scoreboard_ocr", ScoreboardOCRConfig)
    )
    scoreboard.to_parquet(run_dir / "scoreboard_timeline.parquet", index=False)

    goal_cfg_raw = cfg.get("goal_detection") or {}
    goal_detector = GoalDetector(
        GoalDetectorConfig(
            detector=EventDetectorConfig(
                auto_confirm_score=float(goal_cfg_raw.get("auto_confirm_score", 0.85)),
                candidate_score=float(goal_cfg_raw.get("candidate_score", 0.55)),
                unresolved_floor=float(goal_cfg_raw.get("unresolved_floor", 0.20)),
            )
        )
    )

    events: list[MatchEvent] = []
    geometry_signals = _goal_line_candidates(ball_traj, goal_cfg_raw)
    for signal in geometry_signals:
        spotting_score = _spotting_goal_score(spotting_candidates, signal.timestamp_ms)
        scoreboard_score = (
            0.6 if (not scoreboard.empty and bool(scoreboard["score_change"].fillna(False).any())) else None
        )
        scorer = detect_scorer(
            goal_timestamp_ms=signal.timestamp_ms,
            touches=touches,
            config=_section(cfg, "scorer", ScorerDetectorConfig),
        )
        enriched = GoalSignals(
            timestamp_ms=signal.timestamp_ms,
            team_id=_team_index(scorer.scorer_team_id),
            ball_crossed_line_score=signal.ball_crossed_line_score,
            scoreboard_change_score=scoreboard_score,
            kickoff_restart_score=None,
            celebration_break_score=spotting_score,
            from_replay=False,
            scorer_track_id=scorer.scorer_track_id,
            scorer_attribution_score=scorer.scorer_confidence,
        )
        event = goal_detector.detect(enriched)
        if event is None:
            continue
        attrs = dict(event.attributes)
        attrs.update(
            {
                "scorer_global_player_id": scorer.scorer_global_player_id,
                "scorer_jersey_number": scorer.scorer_jersey_number,
                "scorer_status": scorer.scorer_status,
                "scorer_evidence": scorer.scorer_evidence,
                "source_models": (
                    ["geometry", "PTS-baseline/E2E-Spot"]
                    if spotting_score is not None
                    else ["geometry"]
                ),
            }
        )
        events.append(replace(event, attributes=attrs))

    for item in spotting_candidates:
        if str(item.event_type).lower() != "goal":
            continue
        ts_ms = float(item.timestamp) * 1000.0
        if any(abs(existing.timestamp_ms - ts_ms) < 2000 for existing in events):
            continue
        # Single spotting cue alone cannot reach auto-confirm (single_source_cap).
        weak = GoalSignals(
            timestamp_ms=ts_ms,
            celebration_break_score=float(item.confidence),
            from_replay=False,
        )
        event = goal_detector.detect(weak)
        if event is None:
            continue
        attrs = dict(event.attributes)
        attrs["source_models"] = ["PTS-baseline/E2E-Spot"]
        events.append(replace(event, attributes=attrs))

    shot_cfg = cfg.get("shot_detection") or {}
    shot_detector = ShotDetector(
        ShotDetectorConfig(
            detector=EventDetectorConfig(
                auto_confirm_score=float(shot_cfg.get("auto_confirm_score", 0.85)),
                candidate_score=float(shot_cfg.get("candidate_score", 0.50)),
                unresolved_floor=float(shot_cfg.get("unresolved_floor", 0.20)),
            )
        )
    )
    for item in spotting_candidates:
        label = str(item.event_type).lower()
        on_target = "shots on target" in label
        off_target = "shots off target" in label
        if not (on_target or off_target):
            continue
        signal = ShotSignals(
            timestamp_ms=float(item.timestamp) * 1000.0,
            ball_toward_goal_score=float(item.confidence),
            from_replay=False,
            on_target=True if on_target else False,
        )
        event = shot_detector.detect(signal)
        if event is None:
            continue
        attrs = dict(event.attributes)
        attrs["source_models"] = ["PTS-baseline/E2E-Spot"]
        attrs["outcome"] = "on_target" if on_target else "off_target"
        events.append(replace(event, attributes=attrs))

    events = suppress_replay_duplicates(events, shot_segments=shots)

    assist_raw = cfg.get("assist") or {}
    assist_detector = AssistDetector(
        AssistDetectorConfig(
            max_link_window_ms=float(assist_raw.get("max_link_window_ms", 15_000.0)),
            min_pass_confidence=float(assist_raw.get("min_pass_confidence", 0.50)),
        )
    )
    passes = passes_from_chain(chain)
    for goal in list(events):
        if goal.event_type is not EventType.GOAL or not is_confirmed(goal.status):
            continue
        assist = assist_detector.detect_for_goal(goal, passes)
        if assist is not None:
            events.append(assist)

    created_at = started.isoformat()
    match_df = match_events_frame(events, created_at=created_at)
    compat_cols = [
        "event_id",
        "event_type",
        "status",
        "timestamp_ms",
        "team_id",
        "scorer_track_id",
        "assist_track_id",
        "confidence",
    ]
    compat = match_df[compat_cols] if not match_df.empty else pd.DataFrame(columns=compat_cols)
    match_df.to_parquet(run_dir / "match_events.parquet", index=False)
    compat.to_parquet(run_dir / "events.parquet", index=False)

    confirmed = [e for e in events if is_confirmed(e.status)]
    candidates = [e for e in events if e.status is EventStatus.CANDIDATE_REVIEW_REQUIRED]
    unresolved = [e for e in events if e.status is EventStatus.UNRESOLVED]
    reason = "no_supported_event_detected" if not events else "events_detected"

    write_json(
        run_dir / "event_evidence.json",
        {
            "spotting": adapter.last_payload,
            "spotting_error": spotting_error,
            "goal_signal_count": len(geometry_signals),
            "scoreboard_changes": int(scoreboard["score_change"].fillna(False).sum())
            if not scoreboard.empty
            else 0,
            "events": [
                {
                    "event_id": e.event_id,
                    "type": e.event_type.value,
                    "status": e.status.value,
                    "confidence": e.confidence,
                }
                for e in events
            ],
        },
    )
    write_json(
        run_dir / "event_review_queue.json",
        {
            "items": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "timestamp_ms": e.timestamp_ms,
                    "confidence": e.confidence,
                    "status": e.status.value,
                    "suggested_scorer_track_id": e.attributes.get("suggested_scorer_track_id"),
                    "suggested_assist_track_id": e.attributes.get("suggested_assist_track_id"),
                }
                for e in candidates
            ]
        },
    )

    summary = summarize_events(events)
    pd.DataFrame(
        [
            {
                "confirmed_goals": sum(summary.confirmed_goals_by_team.values()),
                "unattributed_confirmed_goals": summary.unattributed_confirmed_goals,
                "pending_review": len(summary.pending_review_event_ids),
                "unresolved": len(summary.unresolved_event_ids),
            }
        ]
    ).to_csv(run_dir / "player_event_summary.csv", index=False)

    team_rows = []
    for team_id in sorted({e.team_id for e in confirmed if e.team_id is not None}):
        team_events = [e for e in confirmed if e.team_id == team_id]
        team_rows.append(
            {
                "team_id": team_id,
                "goals_confirmed": sum(1 for e in team_events if e.event_type is EventType.GOAL),
                "goals_candidates": sum(
                    1 for e in candidates if e.event_type is EventType.GOAL and e.team_id == team_id
                ),
                "assists_confirmed": sum(1 for e in team_events if e.event_type is EventType.ASSIST),
                "shots_confirmed": sum(1 for e in team_events if e.event_type is EventType.SHOT),
                "shots_on_target_confirmed": sum(
                    1
                    for e in team_events
                    if e.event_type is EventType.SHOT and e.attributes.get("outcome") == "on_target"
                ),
            }
        )
    pd.DataFrame(team_rows).to_csv(run_dir / "team_event_summary.csv", index=False)

    clip_cfg = ClipConfig(
        pre_ms=float((cfg.get("clips") or {}).get("pre_seconds", 7.0)) * 1000.0,
        post_ms=float((cfg.get("clips") or {}).get("post_seconds", 8.0)) * 1000.0,
    )
    clip_paths: list[str] = []
    exportable = [
        e
        for e in events
        if e.status
        in {
            EventStatus.AUTO_CONFIRMED,
            EventStatus.CANDIDATE_REVIEW_REQUIRED,
            EventStatus.MANUALLY_CONFIRMED,
        }
    ]
    if bool((cfg.get("clips") or {}).get("export", True)) and video.is_file() and exportable:
        for window in build_clips(exportable, config=clip_cfg):
            out = clips_dir / f"{window.event_id}.mp4"
            try:
                export_clip_mp4(window, video, out)
                clip_paths.append(str(out))
            except Exception:  # noqa: BLE001
                continue

    finished = dt.datetime.now(dt.timezone.utc)
    if not tracks.empty:
        total_frames = int(tracks["frame_id"].max()) + 1
    elif not ball_traj.empty:
        total_frames = int(ball_traj["frame_id"].max()) + 1
    else:
        total_frames = 0

    manifest = {
        "status": "PASS",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "processed_frames": total_frames,
        "total_frames": total_frames,
        "confirmed_event_count": len(confirmed),
        "candidate_event_count": len(candidates),
        "unresolved_event_count": len(unresolved),
        "events_reason": reason,
        "model_sources": ["PTS-baseline/E2E-Spot", "ball_state", "touch_inference"],
        "artifact_paths": {
            "match_events": str(run_dir / "match_events.parquet"),
            "events": str(run_dir / "events.parquet"),
            "event_evidence": str(run_dir / "event_evidence.json"),
            "event_review_queue": str(run_dir / "event_review_queue.json"),
            "scoreboard_timeline": str(run_dir / "scoreboard_timeline.parquet"),
            "ball_trajectory": str(run_dir / "ball_trajectory.parquet"),
            "touch_events": str(run_dir / "touch_events.parquet"),
            "possession_chain": str(run_dir / "possession_chain.parquet"),
            "event_clips": str(clips_dir),
        },
        "error": spotting_error,
        "clip_count": len(clip_paths),
    }
    write_json(stage_dir / "stage_manifest.json", {**manifest, "stage": "event_detection"})
    manifests_dir = run_dir / "stage_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifests_dir / "event_detection.json", manifest)
    write_json(stage_dir / "metrics.json", manifest)
    return manifest
