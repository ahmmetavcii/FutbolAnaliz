# Event Detection Infrastructure Report

Generated: 2026-07-20

## Status

**EVENTS: EVENT_DETECTION_INFRASTRUCTURE_PASS**

**MATCH_EVENTS_REAL_VIDEO_PASS: NOT CLAIMED** (no labeled goal clip validated)

## football.mp4 honesty

- confirmed_event_count: **0**
- candidate_event_count: **0**
- unresolved_event_count: **0**
- events_reason: `no_supported_event_detected`
- spotting (PTS-baseline E2E-Spot official checkpoint): **PASS**, candidates: **0**
- Artifacts: `/home/ahmet/workspace/soccernet_blocker_remediation/event_detection_football_smoke/`

## Components

| Component | Result |
|---|---|
| Event stage pipeline | Wired as `event_detection` after `analytics_render` |
| Spotting model | PTS-baseline / E2E-Spot (`checkpoint_088.pt`) via `sn-pts-baseline` |
| Ball trajectory | From `ball_state.parquet` → `ball_trajectory.parquet` |
| Touch inference | Conservative foot/ball proximity |
| Scoreboard OCR | Lightweight; **no invented score changes** |
| Goal / scorer / assist / shot detectors | Existing library + orchestrator fusion |
| Replay duplicate suppression | Enabled |
| Panel Match Events | Results section + candidate actions + recompute |
| Recompute | `scripts/recompute_match_events.py` (no detection rerun) |

## Remaining blockers

1. No labeled real-goal video validation → cannot claim `MATCH_EVENTS_REAL_VIDEO_PASS`
2. Scoreboard OCR has no digit model (intentionally does not invent scores)
3. Market/OSNet ReID and jersey remain separate from event attribution quality on broadcast crops
