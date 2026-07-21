# Full-match conversion status (2026-07-18)

## Pipeline map (completed)

Reusable foundation confirmed: staged MVP-2 runner, schemas, streaming reader,
calibration chain, jersey standalone model, SoccerNet clean-room exporters.
Main production gaps remain clip truncation, detection/tracking double work,
coarse resume, and missing jersey/ReID wiring.

## What is working now

- CLI scripts: `prepare_full_match`, `run_full_match`, `resume_full_match`,
  `validate_full_match_run`, `export_full_match_results`, sync/calibrate/recompute stubs
- Streamlit panel scaffold: `apps/full_match_panel.py`
- Export package: JSON/CSV/Parquet/Excel/video exporters
- Short real-clip infrastructure run:
  - prepare: `/home/ahmet/workspace/full_match_runs/prep_football_short`
  - run: `/home/ahmet/workspace/full_match_runs/run_football_short_infra`
  - validate: `PASS` with `model_stages_claimed=false`
- Status snapshot: `/home/ahmet/workspace/full_match_runs/status.json`
- Torch verified unchanged: `2.11.0+cu128`

## Blocked (honest)

| Claim | Status | Reason |
|-------|--------|--------|
| `FULL_MATCH_90MIN_PASS` | **NO** | No 45/90-minute input video |
| `MULTICAMERA_REAL_VIDEO_PASS` | **NO** | No same-match multi-camera pair |
| `MULTICAMERA_GLOBAL_ID_REAL_VIDEO_PASS` | **NO** | No real multicamera video |
| `MATCH_EVENTS_REAL_VIDEO_PASS` | **NO** | No event-labeled long match video |

Only short real clip available:
`/mnt/c/football_data/videos/test_clips/football.mp4` (30s).

## In flight

Background package implementations are still converging under
`src/football_analytics/{full_match,multicamera,roles,events}/`. File churn is
expected until those writers finish; CLI APIs are kept behind lazy imports.
