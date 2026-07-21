# SoccerNet Six-Component Final Status Matrix

Generated from remediation verification on 2026-07-18.
Machine-readable source: `docs/setup/soccernet_repo_tests/status.json`.

| Repository | technical_status | original_implementation_status | compatible_implementation_status | license_status | environment | checkpoint | inference | evaluation |
|---|---|---|---|---|---|---|---|---|
| SoccerNet SDK | PASS | PASS | N/A | MIT | ai-dev | N/A | N/A | PASS |
| sn-trackeval | PASS | PASS | PASS | MIT | sn-trackeval-env | N/A | N/A | PASS |
| sn-echoes | PASS | PASS_DEVKIT | PASS | LICENSE_REVIEW_REQUIRED | ai-dev | N/A | N/A (dataset/devkit) | PASS |
| sn-calibration | PASS | ORIGINAL_MODEL_PASS | COMPATIBLE_REPLACEMENT_PASS | LICENSE_REVIEW_REQUIRED | sn-calibration | PASS | PASS | PASS |
| sn-jersey | PASS | N/A_README_ONLY | PASS | LICENSE_REVIEW_REQUIRED | sn-jersey-env | PASS | PASS | PASS |
| sn-gamestate | PASS | BLOCKED_RUNTIME_OOM | COMPATIBLE_IMPLEMENTATION_PASS | GPL-3.0 | sn-gamestate-env / ai-dev | PARTIAL open assets | PASS compatible | PASS compatible |

## Official / external extras

| Item | Status |
|---|---|
| sn-trackeval official SoccerNet GS leaderboard | NOT_RUN_NO_OFFICIAL_GT |
| sn-gamestate original TrackLab end-to-end inference | BLOCKED_RUNTIME_OOM |
| sn-echoes ASR generation on new video | Out of scope (not FAIL) |

## Test counts

| Suite | Passed | Failed |
|---|---:|---:|
| Component regression checks (`test_all_soccernet_components.py`) | 6 | 0 |
| Unit/integration pytest cases (aggregated) | 44 | 0 |
| SDK regression suite (artifact-backed) | 1 | 0 |
| **Total recorded automated checks** | **51** | **0** |

## Key artifacts

| Component | Path |
|---|---|
| SDK | `artifacts/soccernet_repo_tests/SoccerNet/sdk_regression.json` |
| TrackEval | `artifacts/soccernet_repo_tests/sn-trackeval/remediation/scenario_summary.json` |
| Echoes | `artifacts/soccernet_repo_tests/sn-echoes/remediation/full_validation.json` |
| Calibration original | `artifacts/soccernet_repo_tests/sn-calibration/original_model/inference_results.json` |
| Calibration compatible | `artifacts/soccernet_repo_tests/sn-calibration/compatible_replacement/` |
| Jersey | `artifacts/soccernet_repo_tests/sn-jersey/remediation/inference_eval_summary.json` |
| Jersey checkpoint | `/home/ahmet/models/jersey_recognition_v1_best.pt` |
| Game State prediction | `/mnt/c/football_data/results/soccernet_gamestate_prediction.json` |
| Regression JSON/MD/JUnit | `artifacts/soccernet_repo_tests/regression/` |
