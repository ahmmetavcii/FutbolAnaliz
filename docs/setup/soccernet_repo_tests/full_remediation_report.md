# SoccerNet Full Remediation Report

Date: 2026-07-18  
Project: `/home/ahmet/projects/football-analytics`  
Repos: `/home/ahmet/projects/soccernet`  
Data root: `/mnt/c/football_data`  
Reference run: `/home/ahmet/workspace/runs/run_20260718_033654_77a8a7`  
Primary env protected: `ai-dev` (PyTorch/CUDA unchanged)

## Executive summary

All six SoccerNet components reach real technical capability PASS under the
stated policy. Original vs compatible paths are reported separately so a
replacement never masquerades as an original model PASS.

| Repo | technical | original | compatible |
|---|---|---|---|
| SoccerNet SDK | PASS | PASS | N/A |
| sn-trackeval | PASS | PASS | PASS |
| sn-echoes | PASS | PASS_DEVKIT | PASS |
| sn-calibration | PASS | ORIGINAL_MODEL_PASS | COMPATIBLE_REPLACEMENT_PASS |
| sn-jersey | PASS | N/A_README_ONLY | PASS |
| sn-gamestate | PASS | BLOCKED_RUNTIME_OOM | COMPATIBLE_IMPLEMENTATION_PASS |

Automated checks: 6/6 component regression suites PASS; 44 pytest cases PASS;
SDK artifact regression PASS.

---

## 1. SoccerNet SDK

**What works:** import, offline Downloader construction, jersey train/test GT
parse, full image inventory (1,297,548), 100/100 random decode, 0 zero-byte
files, no network required for config tests.

- Environment: `ai-dev`
- Original/compatible: original SDK package
- Dataset: `/mnt/c/football_data/datasets/SoccerNet/jersey-2023`
- Artifact: `artifacts/soccernet_repo_tests/SoccerNet/sdk_regression.json`
- License: MIT
- Blockers: none

## 2. sn-trackeval

**What works:** production adapter
`src/football_analytics/evaluation/trackeval_adapter.py` converts canonical
tracks/GT to MOTChallenge, runs real TrackEval metrics (HOTA, CLEAR, Identity,
Count), and separates synthetic perfect / ID-switch / FP / FN / localization
scenarios. GT absence does not invent scores.

- Environment: `sn-trackeval-env`
- Official leaderboard: `NOT_RUN_NO_OFFICIAL_GT`
- Artifact: `artifacts/soccernet_repo_tests/sn-trackeval/remediation/scenario_summary.json`
- License: MIT
- Blockers: official SoccerNet GS GT not local

## 3. sn-echoes

**What works:** clean-room reader
`src/football_analytics/integrations/sn_echoes_reader.py` streams all 4752 JSON
files (3,566,675 segments), schema/timestamp checks, Whisper variant inventory,
SRT/timeline export. Upstream `stats.py` exit 0 in clean run.

- Environment: `ai-dev`
- Not an inference model; ASR-on-new-video absence is not FAIL
- Artifacts: `artifacts/soccernet_repo_tests/sn-echoes/remediation/`
- License: `LICENSE_REVIEW_REQUIRED` (no source copied into `src/`)
- Blockers: redistribution license review

## 4. sn-calibration

### Original model

Official checkpoint located and verified:

- Path: `/home/ahmet/models/sn-calibration/soccer_pitch_segmentation.pth`
- SHA256: `e51b3f076f9e7791cfe5d8d97b496abb4f6a9d544308a25fd85af856472eaa75`
- Source: official Google Drive asset from sn-calibration README

5 frames from `football.mp4` produced valid cameras/homographies
(`ORIGINAL_MODEL_PASS`).

### Compatible replacement

`src/football_analytics/integrations/sn_calibration_compatible.py` maps PnLCalib
parquet to sn-calibration-compatible contract (105×68 pitch, invertible
homography, orientation, confidence, reprojection, valid/invalid).

- 750-frame run validated; 5 selected frames 5/5 valid
- Artifact dirs under `artifacts/soccernet_repo_tests/sn-calibration/`
- Environment: `sn-calibration`
- License: `LICENSE_REVIEW_REQUIRED`

## 5. sn-jersey

Upstream repo is README-only. Clean-room implementation lives under
`src/football_analytics/jersey/` with scripts/config:

- Temporal MobileNetV3-small + quality-weighted pooling
- Official torchvision ImageNet init (`mobilenet_v3_small-047dcff4.pth`)
- Deterministic tracklet split, no leakage
- Trained 20 epochs on known jersey labels from jersey-2023 train

**Metrics (held-out known validation):**

- Val accuracy (argmax): **57.3%** on 157 tracklets
- Random baseline: **1.3%**
- Infer-20 raw accuracy: **55%**
- Checkpoint: `/home/ahmet/models/jersey_recognition_v1_best.pt`
- SHA256: `7060d422cf30b3ae7ebeb8b09a3fa126d72aaece65f03487ff04a4ec14cebfbc`

Artifacts:

- `artifacts/soccernet_repo_tests/sn-jersey/remediation/predictions.csv`
- `artifacts/soccernet_repo_tests/sn-jersey/remediation/jersey_preview_annotated.mp4`
- `artifacts/soccernet_repo_tests/sn-jersey/remediation/inference_eval_summary.json`

Environment: `sn-jersey-env`  
Status: `compatible_implementation_status=PASS`, `technical_status=PASS`

## 6. sn-gamestate

### Original TrackLab baseline

- Environment: `sn-gamestate-env` (Python 3.9, TrackLab/OpenMMLab pins)
- Open checkpoints downloaded (ReID Zenodo, SV_kp/SV_lines, YOLO11m)
- Pipeline starts on short football clip but is killed with **exit 137 (OOM)**
- Local runtime fix applied in upstream checkout for pandas Series access
  (`nbjw_calib.py` `.iloc[0]`); not claimed as original PASS
- Status: `BLOCKED_RUNTIME_OOM`

### Compatible implementation

`src/football_analytics/integrations/sn_gamestate_compatible.py` +
`scripts/export_soccernet_gamestate.py` export SoccerNet GS-compatible JSON from
MVP-2 run artifacts.

- Output: `/mnt/c/football_data/results/soccernet_gamestate_prediction.json`
- 10,601 predictions, frames 0–749, 7,922 pitch-valid
- Jersey currently null/unknown (no jersey preds wired into that run export)
- Validation: `artifacts/soccernet_repo_tests/sn-gamestate/compatible_implementation/validation_summary.json`
- Status: `COMPATIBLE_IMPLEMENTATION_PASS` → overall technical PASS

License: GPL-3.0 isolated; no GPL source copied into `src/football_analytics`.

---

## Environments

Manifests under `artifacts/soccernet_repo_tests/environments/`:

| Env | Role |
|---|---|
| ai-dev | SDK, echoes reader, compatible exporters (protected) |
| sn-trackeval-env | TrackEval adapter tests |
| sn-calibration | Original calibration inference |
| sn-jersey-env | Jersey training/inference |
| sn-gamestate-env | Original TrackLab attempt |

Each has environment.yml / pip-freeze / pip-check / smoke where created.

## Licenses / notices

- `THIRD_PARTY_NOTICES.md`
- `third_party/manifest.json`
- `third_party/licenses/`

Policy: MIT retained with notice; GPL kept isolated; unverified licenses marked
`LICENSE_REVIEW_REQUIRED` without auto-downgrading technical PASS.

## Common test runner

```bash
python scripts/test_all_soccernet_components.py
```

Outputs:

- `artifacts/soccernet_repo_tests/regression/soccernet_regression.json`
- `artifacts/soccernet_repo_tests/regression/soccernet_regression.md`
- `artifacts/soccernet_repo_tests/regression/soccernet_regression.junit.xml`

## Related docs

- `docs/setup/soccernet_repo_tests/status.json`
- `docs/setup/soccernet_repo_tests/final_status_matrix.md`
- `docs/setup/soccernet_repo_tests/remaining_external_blockers.md`
