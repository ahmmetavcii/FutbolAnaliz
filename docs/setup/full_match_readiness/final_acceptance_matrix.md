# Final acceptance matrix (2026-07-18)

| Item | Status | Evidence |
|---|---|---|
| Package API stabilization (full_match/multicamera/roles/events) | PASS | py_compile + import + CLI/panel import + 253 package tests |
| Broken imports | 0 | import sweep over all package modules |
| Pipeline adapter interface (validate/prepare/run_chunk/validate_outputs/manifest/cleanup) | PASS | `full_match/pipeline_adapter.py`, contract tests |
| Real detection on football.mp4 | PASS | 12 691 rows, YOLO11n, `detections.parquet` |
| Real tracking | PASS | 10 601 rows, 225 ids, ByteTrack, `local_tracks.parquet` |
| Shot segmentation | PASS | 750 rows `shot_segments.parquet` (in pipeline run dir) |
| Calibration + field coordinates | PASS | 94.7 % valid frames, `camera_calibrations.json` |
| Speed/distance metrics | PASS | 10 601 rows `player_frame_metrics.parquet` (7.3 % calibrated-valid, honest gating) |
| Team/role infrastructure on real tracks | PASS (conservative) | 94 outfield_player, 131 unknown_person; officials/goalkeeper honestly unclaimed |
| Referee excluded from team totals | PASS | officials_summary empty, team totals recomputed from countable players only (test-enforced) |
| Jersey inference on possible tracks | PASS (unresolved) | 10 real tracklets scored with trained checkpoint; 0 above threshold |
| Global identity (single camera) | GLOBAL_ID_SINGLE_CAMERA_PASS | 224 global ids, duplicate-free local→global mapping, unresolved supported |
| Event schema/review/clips/recompute | EVENT_DETECTION_INFRASTRUCTURE_PASS | confirmed=0, candidate=0, reason=no_supported_event_detected; correction+recompute exercised |
| model_stages_claimed | true | consistency test ties it to nonzero real detections/tracks |
| annotated_match.mp4 | PASS | ffprobe+OpenCV decode validation (30 s, 1920x1080) |
| tactical_map.mp4 | PASS | rendered from real field coordinates; decode-validated |
| full_match_report.xlsx | PASS | 17 sheets, reopen-validated |
| Panel: real scheduler start/progress/resume | PASS | run id `panel-demo`, /mnt/c/football_data/results/panel-demo/run, 100 % chunks PASS |
| Resume without duplicate processing | PASS | resume after PASS: 0 executed / 1 skipped (0.02 s) |
| Regression: prior 25 + 239 tests | PASS | full suite 517 passed / 0 failed |
| torch/CUDA unchanged | PASS | 2.11.0+cu128 |
| FULL_MATCH_90MIN_PASS | NOT CLAIMED | no 45/90-minute video available |
| MULTICAMERA_REAL_VIDEO_PASS | NOT CLAIMED | no same-match multi-camera recording |
| MATCH_EVENTS_REAL_VIDEO_PASS | NOT CLAIMED | no manually confirmed goal/assist footage |
