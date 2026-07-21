# Full-match readiness report (2026-07-18, real short-video model run)

## Summary

The full-match orchestration stack now runs the **real** MVP-2 model pipeline
(YOLO11n detection, ByteTrack tracking, calibration, field coordinates,
speed/distance, team identity) through the chunk scheduler on
`football.mp4` (30 s, 1920x1080, 25 fps). `model_stages_claimed=true` is backed
by 12 691 real detections and 10 601 real track rows.

## Stabilized packages

`py_compile`, package import, CLI import, panel import, and pytest all pass for:

- `football_analytics.full_match` (schemas, probe, chunking, resume, scheduler,
  progress, health, consolidation, recompute, pipeline adapter, postprocess)
- `football_analytics.multicamera` (sync, calibration manager, global identity,
  fusion, duplicate suppression, coverage)
- `football_analytics.roles` (classifier, referee/goalkeeper specialists,
  temporal voting, active-player state)
- `football_analytics.events` (schemas, evidence, detectors, review, clips,
  summaries)
- `football_analytics.export` (JSON/CSV/Parquet/XLSX/video/tactical map)

No module downloads models or starts heavy work at import time. Broken imports
remaining: **0**.

## Real pipeline adapter

`football_analytics.full_match.pipeline_adapter` provides
`ExistingPipelineAdapter` with `validate_inputs / prepare / run_chunk /
validate_outputs / get_artifact_manifest / cleanup` and a `ChunkContext`
carrying run_id, camera_id, period, chunk_index, source path, frame/time
range, config, and output directory. The scheduler binds to it through
`configs/full_match/existing_pipeline_adapter.yaml`
(`run_full_match --chunk-pipeline-config ...`). The adapter shells into the
proven `scripts/run_pipeline.py` (MVP-2 config, batch=1, workers=0), so the
old pipeline code is reused, not rewritten.

## Real short-video model run

- Run root: `/home/ahmet/workspace/full_match_runs/run_football_short_model`
- Orchestrated scheduler run: `orchestrated/` (1 chunk, PASS, retries=0)
- Underlying pipeline run:
  `chunk_artifacts/camera_1/chunk_00000/pipeline_runs/run_20260718_214418_5349f4`

Stage results (all exit code 0):

| Stage | Output | Count / result |
|---|---|---|
| video probe | prepared/match_manifest.json | 750 frames, first/middle/last decodable |
| shot segmentation | shot_segments.parquet | 750 rows (`main_wide`) |
| detection | detections.parquet | 12 691 rows, real YOLO11n, peak VRAM 50.8 MB |
| tracking | local_tracks.parquet | 10 601 rows, 225 track ids, peak VRAM 56.3 MB |
| calibration | camera_calibrations.json | 750 frames, 94.7 % valid |
| field coords + speed/distance | player_frame_metrics.parquet | 10 601 rows, 7.3 % calibrated-valid |
| team/role infra | role_predictions.parquet | 94 outfield_player, 131 unknown_person |
| jersey inference | jersey_predictions.parquet | 10 real tracklets scored, 0 confident (all unresolved) |
| global identity | global_identity_map.parquet / global_players.parquet | 224 global ids, duplicate-free mapping |
| events | events.parquet | empty, reason `no_supported_event_detected` |
| export | XLSX + MP4s + CSVs | all validated (reopen/decode) |

Peak RAM 2.60 GB, peak VRAM 56.3 MB, torch `2.11.0+cu128` (unchanged).

## Honesty rules honored

- Referee/assistant/goalkeeper roles were **not** claimed: the clip provides no
  officials-kit or goalkeeper-kit reference, so those tracks stay
  `unknown_person` and officials tables are empty with the reason recorded in
  `quality_report.json`.
- Events were **not** invented: confirmed=0, candidate=0,
  reason=`no_supported_event_detected`. Review/correction/recompute flow was
  exercised with an explicit "nothing to correct" record.
- Global-ID success is reported only as `GLOBAL_ID_SINGLE_CAMERA_PASS`.
- Jersey numbers below confidence threshold reported as unresolved, not filled.

## Panel integration

The panel (`apps/full_match_panel.py`) now constructs and launches the real
scheduler CLIs (`prepare_full_match.py`, `run_full_match.py`,
`resume_full_match.py`) with the real chunk-pipeline adapter as default, and
reads live chunk/stage progress from the scheduler's atomic manifests.
A panel-command end-to-end run completed with run id `panel-demo` at
`/mnt/c/football_data/results/panel-demo/run` (all model stages PASS).

## Tests

- Package suites: 253 passed (full_match / multicamera / roles / events / panel)
- New regression tests: 21 passed (adapter contract, real artifacts,
  model_stages_claimed consistency, referee exclusion, identity schema,
  empty-event honesty, XLSX reopen, MP4 decode, panel-scheduler integration,
  resume-without-duplicate)
- Whole project: **517 passed, 0 failed** (previous 25 + 239 all preserved)

## Status

- FULL_MATCH: `REAL_SHORT_VIDEO_PIPELINE_PASS`
- MULTICAMERA: `MULTICAMERA_INFRASTRUCTURE_PASS`
- EVENTS: `EVENT_DETECTION_INFRASTRUCTURE_PASS`
- GLOBAL ID: `GLOBAL_ID_SINGLE_CAMERA_PASS`
- `FULL_MATCH_90MIN_PASS`, `MULTICAMERA_REAL_VIDEO_PASS`,
  `MATCH_EVENTS_REAL_VIDEO_PASS`: **not claimed** (see unresolved_limitations.md)
