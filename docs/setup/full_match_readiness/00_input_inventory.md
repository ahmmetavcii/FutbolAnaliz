# Full-match input inventory

Date: 2026-07-18

## Available real video

| Path | Duration | Resolution | FPS | Notes |
|------|----------|------------|-----|-------|
| `/mnt/c/football_data/videos/test_clips/football.mp4` | 30s | 1920x1080 | 25 | Only confirmed real short clip |

## Missing inputs (blockers)

- No 45-minute half video
- No 90-minute full-match video
- No same-match multi-camera pair (2 or 4 cameras)
- `/mnt/c/football_data/uploads` may be empty / unused for long matches

## Honest status implication

Without suitable inputs, the following statuses **must not** be claimed:

- `FULL_MATCH_90MIN_PASS`
- `MULTICAMERA_REAL_VIDEO_PASS`
- `MULTICAMERA_GLOBAL_ID_REAL_VIDEO_PASS`
- `MATCH_EVENTS_REAL_VIDEO_PASS`

Infrastructure / short-clip / unit-test passes remain valid when evidence exists.
