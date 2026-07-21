# sn-gamestate — Repo Verification Report

**Overall status: BLOCKED** (code is sound and architecture/output-schema fully documented, but there is no functional environment, no model checkpoints, and no official dataset — so imports, model forward, real inference, and GS-HOTA evaluation cannot run)

- **Repo:** `/home/ahmet/projects/soccernet/sn-gamestate`
- **Remote:** `https://github.com/SoccerNet/sn-gamestate.git`
- **Branch/commit:** detached HEAD @ `1c958345067218297d221e45e1a6405f975f83e0`
- **Environment used:** none functional. `ai-dev` (Py3.10/torch2.11) verified but **incompatible** with repo (`requires-python >=3.9,<3.10`, `torch==1.13.1`) and lacks TrackLab. Dedicated `sn-gamestate-python` (Py3.9.23) has the **correct Python but no dependencies installed**.
- **Tested:** 2026-07-18

---

## 1. Repo integrity

| Item | Value |
|------|-------|
| Remote | https://github.com/SoccerNet/sn-gamestate.git |
| Branch/commit | detached HEAD @ `1c958345067218297d221e45e1a6405f975f83e0` |
| Dirty (before/after) | clean → clean (only gitignored `__pycache__` from compileall; no tracked files changed) |
| `git diff --stat` | empty (`repo_initial_diff.patch`, 0 bytes) |
| `git fsck` | clean |
| Submodules / LFS | none / none |
| Tracked files | 123 |
| Disk size | 196M w/ `.git`, 98M without (mostly `examples_predictions/*.zip` 69M + images/gifs) |
| License | **GPL-3.0** (LICENSE present) |

## 2. Purpose & output contract

**SoccerNet Game State Reconstruction (GSR)** baseline built on **TrackLab** (`tracklab==1.3.24`, Hydra/OmegaConf modular pipeline). Pipeline (`configs/soccernet.yaml`):
`bbox_detector (YOLOv8) → reid (prtreid) → track (bpbreid_strong_sort) → pitch (nbjw_calib) → calibration (nbjw_calib HRNet) → jersey_number_detect (mmocr) → tracklet_agg (voting_role_jn) → team (kmeans_embeddings) → team_side (mean_position)`.

**Output** (per-player, all fields supported): `bbox_image {x,y,w,h}`, `bbox_pitch {x/y_bottom_left/right/middle}` (meters), `track_id`, `role`, `team`, `jersey`, `image_id`, `video_id`, `category_id`. (Schema extracted read-only from `examples_predictions/SoccerNetGS-test.zip`.)

- **Input:** SoccerNet video dataset (SNGS clips); `configs/dataset/youtube.yaml` also supports arbitrary single video.
- **Entrypoints:** inference `tracklab -cn soccernet`; eval `eval=gs_hota` (**GS-HOTA** metric).
- **Dataset required:** official **SoccerNetGS** (`data/SoccerNetGS`).
- **Checkpoints:** none in repo (only calibration `mean/std/Radar`); all model weights **auto-downloaded by TrackLab at runtime**.
- **Constraints/assumptions:** Python 3.9 only; `torch==1.13.1`, `numpy==1.26.4`, `mmocr==1.0.1`, `mmdet~=3.1.0` (openmmlab); default `nvid=1`; Hydra chdir to `outputs/`; runtime weight auto-download.

## 3. Environment

- **ai-dev:** Python 3.10.20, torch 2.11.0+cu128, `pip check` clean — but violates repo `<3.10` constraint and lacks tracklab. Not modified.
- **sn-gamestate-python:** Python 3.9.23 (correct), but only `pip/setuptools/uv/wheel` installed — **no torch/numpy/tracklab/mmocr/prtreid/easyocr**. Installing them needs internet + heavy openmmlab/git builds → **forbidden**. → No runnable environment.

## 4. Code / import / CLI

- `compileall` (sn_gamestate + plugins): **PASS** (syntax OK).
- Imports: `sn_gamestate`, `sn_gamestate.config_finder` → OK (pure Python). All functional modules → **FAIL** with `ModuleNotFoundError` (`tracklab`, `numpy`, `pandas`, `PIL`) = **missing dependencies**, not syntax/path errors.
- No unit tests in repo. `tracklab` CLI unavailable.

## 5. Checkpoint inventory

Repo ships no model weights (only `calibration/mean.npy`, `std.npy`, `Radar.png`). Required weights (YOLOv8, prtreid/bpbreid, nbjw_calib HRNet, mmocr) are auto-downloaded at runtime (forbidden). No compatible local checkpoints (only unrelated `yolo11n.pt`, `sn-banner/*.pth`). → **BLOCKED_CHECKPOINT_MISSING** (no random weights used).

## 6. Dataset / samples

Official **SoccerNetGS not present** (no `data/` dir). Only `examples_predictions/SoccerNetGS-test.zip` (per-video prediction JSONs) — used read-only for schema. No local GT for GS-HOTA.

## 7–9. Model forward / real inference / evaluation

- **Model forward smoke: BLOCKED** — no torch/tracklab; model classes not constructable. Not run with random weights.
- **Real inference: BLOCKED** — no env + no checkpoints + no dataset; every substage (detector/reid/track/calibration/jersey/team) BLOCKED; 20s clip not run.
- **Evaluation (GS-HOTA): BLOCKED** — needs tracklab TrackEval + official dataset/GT; evaluator could not run. Output schema documented from example predictions instead.

## 10. football-analytics compatibility

See `sn_gamestate_integration_mapping.md`. sn-gamestate output is **schema-compatible with MVP-2 parquet via a thin adapter**: `bbox_image` ← tracks xyxy→xywh; `bbox_pitch` ← game_state `x_field/y_field` (pitch-origin/orientation alignment needed); `track_id`, `role`, `team_id` direct/mapped; **`jersey` is a gap** (MVP-2 has no jersey stage). Both use meters on a 105×68 pitch but different origin/orientation. No fake benchmark produced (no GT).

## 11. Artifacts (all present, non-empty, parseable)

`/home/ahmet/projects/football-analytics/artifacts/soccernet_repo_tests/sn-gamestate/`
- `repo_inventory.json`, `environment_summary.json`, `checkpoint_inventory.json`, `dataset_inventory.json`, `import_test_results.json`, `model_forward_results.json`, `inference_results.json`, `evaluation_smoke_results.json`, `output_schema.json`, `repo_initial_diff.patch` (empty = clean).

## 12. Result

| Metric | Value |
|--------|-------|
| Tests passed | 5 (integrity, architecture, compileall, output-schema, parquet-compat) |
| Tests failed | 0 |
| Import/CLI | PARTIAL (pure-Python only; functional modules blocked by missing deps) |
| Checkpoint | BLOCKED_CHECKPOINT_MISSING |
| Dataset | official SoccerNetGS absent |
| Model forward | BLOCKED |
| Real inference | BLOCKED |
| Evaluation | BLOCKED |
| Overall | **BLOCKED** |

**Open blockers:**
1. No functional environment (sn-gamestate-python deps not installed; ai-dev wrong Python + no tracklab; offline install of torch1.13+openmmlab+git deps forbidden).
2. Model checkpoints missing (auto-download forbidden).
3. Official SoccerNetGS dataset + GT absent.

**Log:** `/home/ahmet/projects/football-analytics/logs/soccernet_repo_tests/06_sn_gamestate.log`
