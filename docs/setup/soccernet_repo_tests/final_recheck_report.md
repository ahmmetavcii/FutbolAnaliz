# SoccerNet Final Regression Recheck Report

Date: 2026-07-18  
Scope: verification-only regression of the six SoccerNet components. No dataset,
model, or checkpoint was re-downloaded; no environment was rebuilt; `ai-dev`
package versions were untouched; no runs or artifacts were deleted.

- Overall: **PASS**
- Recheck assertions: **40 passed / 0 failed**
- pytest unit/integration: **125 passed / 0 failed**
- Machine-readable: `artifacts/soccernet_repo_tests/final_recheck_summary.json`

Original and compatible implementation statuses are reported separately; no
failing feature is reported as PASS.

---

## 1. SoccerNet SDK — PASS
- `import SoccerNet` and `SoccerNetDownloader` import: OK
- train/test image dirs present; train_gt.json / test_gt.json parse OK
- Deterministic 10-image OpenCV decode: 10/10 decoded
- 0-byte file count: 0

## 2. sn-trackeval — PASS (original PASS, compatible PASS)
- Synthetic scenario artifacts present and parse: perfect, id_switch, fp, fn
- HOTA / MOTA / IDF1 present in scenario metrics
- `scenario_summary.json` present with metrics
- Official leaderboard: **NOT_RUN_NO_OFFICIAL_GT** (kept separate)

## 3. sn-echoes — PASS (license separated)
- JSON files: **4752** (expected 4752)
- Total segments: **3,566,675** (from full_validation artifact)
- `stats.py` exit code: 0
- Streaming reader lazily iterated one match: 1880 segments == 1880 raw segments
- License: **LICENSE_REVIEW_REQUIRED** (kept separate; does not lower technical status)

## 4. sn-calibration — PASS (original + compatible, separated)
### Original model (re-run this recheck)
- Checkpoint `/home/ahmet/models/sn-calibration/soccer_pitch_segmentation.pth`
- 5/5 real frames valid; per-frame homography finite and invertible (|det| > 1e-12)
- `original_implementation_status = ORIGINAL_MODEL_PASS`
### Compatible replacement (PnLCalib → contract)
- 5 selected frames: homography finite + invertible, confidence + reprojection finite
- 750-frame run: 682 valid preserved
- `compatible_implementation_status = COMPATIBLE_REPLACEMENT_PASS`

## 5. sn-jersey — PASS (clean-room implementation)
This is explicitly a **clean-room implementation in football-analytics**; the
upstream sn-jersey repo is README-only.
- Dataset train/test verified
- Checkpoint `/home/ahmet/models/jersey_recognition_v1_best.pt` loads OK
- Validation accuracy artifact parsed: beats_random = true
- ≥20 known validation tracklets inference: 20 tracklets, argmax accuracy 0.55
- Preview MP4 inference on `jersey_tracklet_preview.mp4` (tracklet 0, GT=10):
  - Extracted frames and ran real inference
  - Prediction: **-1 (unknown)**, confidence 0.13 — below the 0.35 acceptance
    threshold, so it correctly abstains on the letterboxed/annotated preview
    render (domain-shifted from tight training crops). GT=10.
  - Reported honestly: inference ran and produced prediction + confidence vs GT;
    not counted as a jersey-number match.

## 6. sn-gamestate — PASS via compatible (original separated)
- Existing JSON validated **without re-running the exporter**:
  `/mnt/c/football_data/results/soccernet_gamestate_prediction.json`
- Parses; predictions = **10,601**; distinct frames = **750**
- bbox valid (non-negative w/h); pitch coordinates finite on valid rows
- `compatible_implementation_status = COMPATIBLE_IMPLEMENTATION_PASS`
- `original_implementation_status = BLOCKED_RUNTIME_OOM` (preserved; original not
  marked PASS until OOM is resolved)

---

## Cross checks
- Prior regression artifacts present (JSON/MD/JUnit): OK
- All JSON/CSV artifacts open cleanly: OK (0 bad files)
- All preview/annotated MP4s open: OK
- Contact sheet reopens: OK
- No XLSX artifacts present (none expected)
- status.json: all six technical_status = PASS; calibration original = ORIGINAL_MODEL_PASS;
  gamestate original = BLOCKED_RUNTIME_OOM — consistent with reports
- Jersey dataset download report present and `Overall: PASS`
- Preview + contact sheet present in `test_clips`

## Remaining real blockers
1. sn-gamestate original TrackLab pipeline OOM (exit 137) — needs reduced module
   set / smaller batch / lower-resolution clip to attempt original PASS.
2. sn-trackeval official leaderboard needs official SoccerNet GT (external/NDA).
3. LICENSE_REVIEW_REQUIRED for sn-echoes, sn-calibration, sn-jersey before redistribution.

## Fix priority order
1. (Optional, if original baseline PASS required) Retry sn-gamestate original with
   lower VRAM footprint.
2. Acquire official GT to run sn-trackeval leaderboard verification.
3. Complete license review for the three LICENSE_REVIEW_REQUIRED repos.
