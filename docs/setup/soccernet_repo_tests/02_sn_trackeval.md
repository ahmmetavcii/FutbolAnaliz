# sn-trackeval — Repo Verification Report (02)

- **Repo:** `/home/ahmet/projects/soccernet/sn-trackeval`
- **Remote:** https://github.com/SoccerNet/sn-trackeval.git
- **Branch:** detached HEAD
- **Commit:** `9c25232f6f2b56c9f203f1eb55784ff1e97df683`
- **Package version:** 0.4.0 (TrackEval + SoccerNet extensions)
- **Environment:** `ai-dev` (Python 3.10.20, torch 2.11.0+cu128, CUDA available: True)
- **Date:** 2026-07-18
- **Overall result:** PARTIAL
- **Passed test groups:** 7 / 7
- **Failed test groups:** 0
- **Log:** `/home/ahmet/projects/football-analytics/logs/soccernet_repo_tests/02_sn_trackeval.log`
- **Artifacts:** `/home/ahmet/projects/football-analytics/artifacts/soccernet_repo_tests/sn-trackeval`

## Summary

The core TrackEval evaluator (HOTA, CLEAR, Identity, Count) was run **for real** on a
self-built synthetic MOTChallenge-format fixture and behaved correctly, cleanly
separating a perfect tracker from id-switch / false-positive / missed-detection variants,
and produced native result artifacts. Overall status is **PARTIAL** (not PASS) because the
**official SoccerNet Game-State (GS) benchmark and format could not be validated** — the
NDA/official SoccerNet tracking ground-truth dataset is not present on this machine.

## Test Results

### 1. Repo integrity — PASS
- Folder exists: YES
- Remote: `https://github.com/SoccerNet/sn-trackeval.git`
- Branch: detached HEAD; Commit: `9c25232f6f2b56c9f203f1eb55784ff1e97df683`
- Dirty status: clean
- `git fsck --full`: clean
- Disk size: 3.1M total (2.4M excluding `.git`)
- Submodules: none (no `.gitmodules`)

### 2. Environment — PASS
- python: `/home/ahmet/miniconda3/envs/ai-dev/bin/python`, Python 3.10.20
- torch 2.11.0+cu128, CUDA available: True
- numpy 2.2.6, scipy 1.15.3, pandas 2.3.3, opencv 5.0.0 (read-only, **not modified**)
- `pip check`: No broken requirements found
- Note: `minimum_requirements.txt` pins ancient `scipy==1.4.1`, `numpy==1.18.1`; these were
  **not** applied (would conflict with ai-dev). The current numpy 2.x works with the core
  metric path (no deprecated `np.float`/`np.int` usage there).

### 3. Code audit — PASS
- Supported tracking formats / datasets: MOTChallenge 2D box, MOTS, KITTI, KITTI-MOTS,
  DAVIS, BDD100K, TAO, YouTube-VIS, BURST, HeadTracking, PersonPath22, RobMOTS, and
  **SoccerNetGS** (SoccerNet game-state).
- Metrics present: **HOTA, CLEAR, Identity, VACE, Count, IDEucl, JAndF, TrackMAP**.
- SoccerNet GS ground-truth expectation: MOTChallenge-style tracks plus game-state
  attributes (roles, teams, jersey numbers, pitch/image space) — see `run_soccernet_gs.py`
  options (`USE_ROLES`, `USE_TEAMS`, `USE_JERSEY_NUMBERS`, `EVAL_SPACE`, ...).
- Prediction format: MOTChallenge CSV `frame,id,bb_left,bb_top,bb_w,bb_h,conf,x,y,z`.
- API/CLI: Python `trackeval.Evaluator` + dataset + metrics; CLI via `scripts/run_*.py`.
- `python -m compileall -q trackeval scripts tests` → exit 0.
- `import trackeval` OK; `from trackeval.datasets import SoccerNetGS` OK.
- Self-contained unit tests: `pytest tests/test_metrics.py` → **9 passed** (synthetic
  CLEAR/Identity/VACE reference cases).
- CLI help: `python scripts/run_soccernet_gs.py --help` → exit 0.
- `test_mot17.py`, `test_davis.py`, `test_mots.py`, `test_all_quick.py` require a
  `data/tests/...` reference-data folder that is **not present** → not run (missing data).
- Missing optional dependency `tabulate` (listed in `pyproject`) only affects the BURST
  dataset/script (gracefully skipped); irrelevant to SoccerNet/core metrics. Env left
  unchanged (not installed).

### 4. Synthetic fixture — PASS
Built a minimal MOTChallenge fixture (kept in `/home/ahmet/workspace/staging/sn_trackeval_smoke`, not added to repo):
- 10 frames, 2 ground-truth players.
- 4 tracker scenarios: `perfect`, `idswitch`, `falsepos`, `missed`.

### 5. Metric behavior — PASS
Full `trackeval.Evaluator` + `MotChallenge2DBox` + HOTA/CLEAR/Identity/Count:

| scenario | HOTA | MOTA | IDSW | CLR_FP | CLR_FN | IDF1 | IDR | AssA |
|---|---|---|---|---|---|---|---|---|
| perfect  | 1.0000 | 1.00 | 0 | 0 | 0 | 1.0000 | 1.00 | 1.0000 |
| idswitch | 0.5774 | 0.90 | 2 | 0 | 0 | 0.5000 | 0.50 | 0.3333 |
| falsepos | 0.8165 | 0.50 | 0 | 10 | 0 | 0.8000 | 1.00 | 1.0000 |
| missed   | 0.7382 | 0.65 | 0 | 0 | 7 | 0.7879 | 0.65 | 0.8385 |

- **A) Perfect** → max HOTA=1.0, IDSW=0, FP=0, FN=0. ✓
- **B) ID switch** → HOTA drops (0.577 < 1.0), IDSW rises to 2. ✓
- **C) False positive** → CLR_FP=10, MOTA penalized to 0.5. ✓
- **D) Missed detection** → recall IDR=0.65 and CLR_FN=7 (recall dropped). ✓

### 6. Artifact validation — PASS
- Per tracker, native TrackEval outputs `pedestrian_summary.txt` and
  `pedestrian_detailed.csv` were produced, all **non-empty** and **parseable**
  (detailed.csv: 3 rows × 276 columns each).
- Perfect vs broken summaries are clearly different.
- Copied to `/home/ahmet/projects/football-analytics/artifacts/soccernet_repo_tests/sn-trackeval`
  (plus `smoke_summary.json`).

### 7. football-analytics `tracks.parquet` compatibility — PASS (schema only)
- Inspected `/home/ahmet/workspace/runs/run_20260718_033654_77a8a7/tracks.parquet`
  (10,601 rows).
- Columns: `frame_id`(0-based), `track_id`, corner bbox `bbox_x1..y2`,
  `tracking_confidence`, `object_type`, `class_id`, etc.
- **Adapter required:** corner→`left,top,width,height` and `frame_id+1` (1-based).
  Full mapping in `sn_trackeval_adapter_mapping.md`.
- **No ground truth** exists in/with `tracks.parquet`; evaluation is impossible on it and
  **no fake metric was produced**. A real benchmark needs SoccerNet GT (see blocker).
- No production adapter was implemented (by design).

## Overall: PARTIAL

- Core evaluator + generic MOT metric families (HOTA/CLEAR/Identity/Count, plus VACE via
  the repo's own unit tests) are fully functional and behaviorally correct on synthetic
  data, with real artifacts.
- The official SoccerNet Game-State benchmark and format remain **unverified**.

## Open Blocker
- **SoccerNet Game-State / tracking ground-truth dataset (NDA / official) is not present**
  on this machine. Without it, the official `run_soccernet_gs.py` benchmark and the
  SoccerNet-specific GT format cannot be validated, and no metric can be computed on our
  own `tracks.parquet` (predictions only, no GT).

## Cleanup
- The synthetic fixture + generated outputs were created under
  `/home/ahmet/workspace/staging/sn_trackeval_smoke`, validated, and their result artifacts
  copied to the artifacts dir; the temporary staging directory was then **removed**.
- No repo files were modified (repo `git status` clean); ai-dev packages unchanged.
