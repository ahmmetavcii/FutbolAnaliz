# MVP-2 Spatial Analytics Report

**Date:** 2026-07-18  
**Environment:** `ai-dev` (unchanged)  
**PyTorch:** `2.11.0+cu128`  
**GPU:** NVIDIA GeForce RTX 4060 Laptop GPU  
**Config:** `configs/pipeline/mvp2_spatial_analytics.yaml`

## Outcome

MVP-2 canonical orchestration, streaming analytics stages, uncertainty gates,
tests, and real-video artifacts are implemented. After the calibration provider
chain fix, the 30-second football clip produces real field coordinates and
physical speed/distance for tracks that pass quality and plausibility gates.
Overall spatial analytics status is **PARTIAL** (not every frame/track is
calibrated; ball/possession coverage remains low; `sn_calibration` is blocked).

## Source audit and license

- Reference repository:
  https://github.com/abdullahtarek/football_analysis
- Commit: `e4799632cdf271cf57f4eb0c4d872cf1b7ab0f17`
- Repository verification: clean, 13,491,870 bytes, `git fsck --full` PASS
- License: **LICENSE_NOT_VERIFIED**
- Audit: `docs/research/reference_football_analysis_audit.md`

No source file was copied. Independently reimplemented ideas:

- player bottom-centre foot geometry and ball bbox centre;
- temporal shirt-colour evidence;
- sparse optical flow, upgraded to forward/backward LK + RANSAC affine;
- homography only as a validated calibration primitive;
- player-ball proximity only as one temporal possession signal;
- ellipse/triangle visual cues with resolution-relative overlays.

Rejected reference behavior includes fixed paths, pickle stubs, all-frame RAM
loading, batch 20/confidence 0.1 constants, unlimited ball interpolation+bfill,
ball ID 1, 70-pixel possession, player 91 exception, fixed 24 FPS/window,
fixed four pixels/23.32 m, 1920×1080 overlays, fixed optical-flow columns,
single-max-feature motion, and indefinite possession carry-forward.

## Calibration diagnosis and fix

Root cause on `run_20260718_033654_77a8a7` before the fix:

- Stage only tried `MetadataCalibrationProvider` against `video_manifest.json`.
- Manifest has no `calibration` key → `invalid_reason="metadata: unavailable"`.
- Config `provider_priority: [sn_calibration, pnlcalib, ...]` was never wired;
  the chain rejected any non-metadata/manual provider.
- Provider attempts were not recorded in the stage directory.

Fix:

1. **A — Manual calibration tool**
   `scripts/create_manual_calibration.py` (interactive click mode +
   `--points-file`). Validates non-collinearity, homography, reprojection
   error and coverage; writes `DEMO_MANUAL_FALLBACK` JSON. Generated
   `configs/calibration/manual_football_frame100.json`
   (7 points, reprojection 0.202 m, coverage 0.215, conf 0.975).
2. **B — Real automatic PnLCalib**
   Out-of-process worker in `sn-banner-mmseg` (shapely + torch/CUDA). Worker
   returns only detected image/pitch correspondences; in-process
   `calibration_from_mapping` applies every gate. Per-frame sampling with
   stride/hold. `sn_calibration` recorded as **BLOCKED** with no placeholder.
3. **Resume**
   `--rerun-from calibration` with `--resume-run-dir` force-reruns from that
   stage; earlier stages skip after checksum/output validation (detection and
   tracking were not recomputed).

## Pipeline and contracts

Implemented orchestration:

`ingest → shot classification → detection → tracking → track quality → team/role
identity → camera motion → calibration/field coordinates → ball trajectory →
possession → player/team metrics → analytics rendering`

Each stage validates inputs/outputs and writes a checksum/config-hash manifest.
`--resume-run-dir` validates completed artifact checksums and skips valid stages.
`--rerun-from <stage>` forces recomputation from that stage onward.
Video stages stream frames; configured chunk boundaries are at most 300 seconds.

Canonical schemas/artifacts:

- `shot_segments.parquet`
- `track_identities.parquet`
- `camera_motion.parquet`
- `calibration.parquet`
- `game_state.parquet`
- `ball_state.parquet`
- `possession_timeline.parquet`
- `track_quality.parquet`
- `player_metrics.parquet`
- `team_metrics.parquet`

All MVP-2 schemas include `schema_version`, `run_id`, `match_id`, `frame_id`,
`timestamp_ms`, `source_method`, `confidence`, and `valid`. Unknown values are
nullable.

## Created/updated implementation files

- Config/contracts: `configs/pipeline/mvp2_spatial_analytics.yaml`,
  `configs/calibration/manual_football_frame100.json`,
  `src/football_analytics/contracts/schemas.py`
- Geometry/video:
  `src/football_analytics/geometry/bbox.py`,
  `src/football_analytics/video/streaming.py`
- Analytics under `src/football_analytics/analytics/`
- Stages under `src/football_analytics/stages/` (calibration rewritten)
- Rendering: `src/football_analytics/visualization/analytics_renderer.py`
- Orchestration/resume:
  `src/football_analytics/orchestration/runner.py`,
  `src/football_analytics/stages/base.py`,
  `scripts/run_pipeline.py`
- Calibration tooling:
  `scripts/create_manual_calibration.py`,
  `scripts/pnlcalib_worker.py`
- Validation: `scripts/validate_mvp2_outputs.py`
- Tests under `tests/`
- Governance/docs: `external_repos.lock.yaml`, `THIRD_PARTY_NOTICES.md`,
  research/setup reports

## Tests

- Unit tests (subset after calibration change): **30 passed**
  (`test_canonical_schemas`, `test_player_metrics`, `test_stage_resume`,
  `test_bbox_geometry`)
- Manual tool rejects collinear four-point sets (exit 1, no file written)
- Detection/tracking manifests unchanged across the calibration resume
  (completed_at still `03:37` / `03:38`; calibration/metrics/render at `04:11+`)

## Primary 30-second real-video run (post-calibration fix)

Input:
`/mnt/c/football_data/videos/test_clips/football.mp4`

- Duration/format: 30.0 s, 1920×1080, 25 FPS, 750 frames
- Run: `/home/ahmet/workspace/runs/run_20260718_033654_77a8a7`
  (resumed with `--rerun-from calibration`)
- Model/tracker: `yolo11n.pt`, device 0, FP16, 640 px, batch 1; ByteTrack
- Calibration provider attempts (recorded):
  - `sn_calibration`: **BLOCKED**
  - `pnlcalib`: **valid** — 90/151 sampled frames passed gates
    (stride=5, hold=10, worker 87.1 s on `cuda:0`)
  - `metadata` / `manual_json` / `demo_four_point`: not attempted (earlier success)

Measured quality after resume:

| Signal | Result |
|---|---:|
| Shot classification | 750/750 `main_wide` |
| Track quality usable | 23 tracks |
| Calibration valid frames | **682 / 750 (90.9%)** |
| Mean / max reprojection error (valid) | 1.800 m / 3.857 m |
| `game_state` x_field/y_field filled | **7922 / 10601 (74.7%)** |
| `player_metrics` valid rows | **804 / 10601** |
| Players in `player_speed_summary.csv` | **22 data rows** |
| Max smoothed speed | 37.57 km/h (filter 38.0; **0 rows > 38**) |
| Sum of per-player `total_distance_m` | 95.5 m |
| Team metrics valid | 692 / 1500 |

Top `total_distance_m` (per track, measurable segments only):

| track_id | total_distance_m |
|---:|---:|
| 64 | 14.8 |
| 199 | 12.3 |
| 4 | 9.9 |
| 1 | 9.7 |
| 194 | 7.0 |

Physical speed/distance remain gated by calibration confidence, track quality,
shot type, sample count and the maximum-plausible-speed filter. Rows that fail
those gates stay invalid/null; they are not filled with pixel motion.

## Status by feature

| Feature | Status | Evidence / limitation |
|---|---|---|
| Bounding-box/foot geometry | PASS | clipped/validated; crop and foot confidence tested |
| Shot classifier | PARTIAL | heuristic labels/cuts tested; no trained replay model |
| Team identity | PASS baseline | temporal LAB/HSV, quality/outlier filters; measured coverage |
| Role identity | PARTIAL | API/gates tested; COCO model supplies no referee/goalkeeper labels |
| Camera motion | PASS | streaming FB-LK + RANSAC affine and resets |
| Calibration (PnLCalib) | **PASS** | real out-of-process inference; 682/750 valid; attempts logged |
| Calibration (sn-calibration) | **BLOCKED** | weights/env unavailable; no placeholder written |
| Manual calibration tool | **PASS** | interactive + points-file; gates reject bad sets |
| Field coordinates | **PASS** | 7922/10601 filled under valid calibration |
| Ball trajectory | PASS baseline | bounded prediction/reset; low detector coverage retained |
| Possession | PASS baseline | metre/normalized-distance state machine with unknown timeout |
| Player physical metrics | **PARTIAL** | 804 valid rows / 22 tracks with distance; coverage limited by gates |
| Team spatial metrics | **PARTIAL** | 692/1500 valid under identity+calibration gates |
| Analytics rendering | PASS | videos/PNG/CSV produced and reopened |
| Full-match chunk restart | PARTIAL | streaming + stage resume; partitioned in-stage restart deferred |

Overall: **PARTIAL**.

## Key outputs

Primary updated run:

- `/home/ahmet/workspace/runs/run_20260718_033654_77a8a7/`
- `calibration.parquet`, `game_state.parquet`, `player_metrics.parquet`
- `player_speed_summary.csv` (22 data rows)
- `analytics_annotated.mp4`, `tactical_preview.mp4`
- `stages/calibration/provider_attempts.json`
- `stages/calibration/pnlcalib_frames.json`
- Manual JSON:
  `configs/calibration/manual_football_frame100.json`

Logs:

- `/home/ahmet/workspace/staging/calibration/rerun_calibration.log`
- `/home/ahmet/workspace/staging/calibration/rerun_render.log`
- `/home/ahmet/workspace/staging/calibration/pnl_probe.json`

## Next development order

1. Register a licensed football-specific detector with role/ball classes.
2. Tighten PnLCalib temporal consistency (homography smoothing, denser sampling
   near scene cuts) and evaluate against labelled landmarks.
3. Evaluate and tune team identity on labelled tracklets.
4. Train/evaluate shot/replay classification.
5. Partition canonical Parquet by chunk and add in-stage chunk checkpointing.
6. Benchmark tracking, ball, calibration and possession against labelled data.

See `docs/setup/mvp2_known_limitations.md` for conservative scope details.
