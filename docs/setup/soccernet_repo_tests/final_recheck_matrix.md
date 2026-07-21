# SoccerNet Final Regression Recheck Matrix

Generated: 2026-07-18 (recheck only; no downloads, no re-training, no env changes).
Source: `artifacts/soccernet_repo_tests/final_recheck_summary.json`

| Repository | technical_status | original_implementation | compatible_implementation | recheck |
|---|---|---|---|---|
| SoccerNet SDK | PASS | PASS | N/A | PASS |
| sn-trackeval | PASS | PASS | PASS | PASS |
| sn-echoes | PASS | PASS_DEVKIT | PASS | PASS |
| sn-calibration | PASS | ORIGINAL_MODEL_PASS | COMPATIBLE_REPLACEMENT_PASS | PASS |
| sn-jersey | PASS | N/A_README_ONLY | PASS (clean-room) | PASS |
| sn-gamestate | PASS | BLOCKED_RUNTIME_OOM | COMPATIBLE_IMPLEMENTATION_PASS | PASS |

## External / separated statuses (unchanged)

| Item | Status |
|---|---|
| sn-trackeval official leaderboard | NOT_RUN_NO_OFFICIAL_GT |
| sn-gamestate original TrackLab pipeline | BLOCKED_RUNTIME_OOM |
| sn-echoes license | LICENSE_REVIEW_REQUIRED |
| sn-calibration / sn-jersey license | LICENSE_REVIEW_REQUIRED |

## Recheck test counts

| Suite | Passed | Failed |
|---|---:|---:|
| Final recheck assertions (this run) | 40 | 0 |
| pytest unit/integration (this run) | 125 | 0 |

## Key verified numbers

| Metric | Value |
|---|---|
| Echoes JSON files | 4752 |
| Echoes total segments | 3,566,675 |
| Echoes streaming reader (one match) | 1880 = 1880 raw |
| Calibration original valid frames | 5/5 |
| Calibration compatible 750-run valid | 682/750 |
| Jersey known-20 argmax accuracy | 0.55 |
| Jersey val vs random | beats_random = true |
| Game State predictions | 10,601 |
| Game State distinct frames | 750 |
