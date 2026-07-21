# sn-jersey — Repo Verification Report

**Overall status: PARTIAL** (dataset + loader + evaluation metric fully validated; but the repo ships **no baseline code** and no checkpoint, so real jersey-number recognition is BLOCKED)

- **Repo:** `/home/ahmet/projects/soccernet/sn-jersey`
- **Remote:** `https://github.com/SoccerNet/sn-jersey.git`
- **Branch/commit:** detached HEAD (main) @ `2f43b48c59eefe0bb5d948888db07f55f51208ad` ("Update README.md")
- **Environment used:** `ai-dev` (Python 3.10.20, torch 2.11.0+cu128, numpy 2.2.6, pandas 2.3.3, opencv 5.0.0, sklearn 1.7.2, CUDA available). No dedicated `sn-jersey` conda env exists. ai-dev not modified.
- **Tested:** 2026-07-18

---

## 1. Repo integrity

| Item | Value |
|------|-------|
| Folder exists | YES |
| Remote | https://github.com/SoccerNet/sn-jersey.git |
| Branch | detached HEAD (main); all refs point to same commit |
| Commit SHA | `2f43b48c59eefe0bb5d948888db07f55f51208ad` |
| Dirty (before/after) | clean → clean (repo not modified by test) |
| `git diff --stat` | empty (`repo_initial_diff.patch`, 0 bytes) |
| `git fsck` | clean |
| Submodules | none |
| Git-LFS pointers | none (`.gitattributes` absent) |
| Disk size | ~208K with `.git`, ~20K without |
| Tracked files | **1** (`README.md` only) |
| License | **LICENSE_NOT_VERIFIED** — no LICENSE/COPYING, no license text in README |

## 2. Purpose & architecture

- **Repo type:** **README-only** challenge/dataset specification. **No baseline code** (no loader, model, training, inference, or evaluation scripts; no notebooks; no unit tests) on any ref.
- **Task:** Jersey Number **Recognition** — classify each player **tracklet** (folder of thumbnail crops) into an integer jersey number; **`-1` = not visible**. Not localization/detection.
- **Input:** multi-frame tracklet (number visible only in a subset of low-res/blurred frames → temporal aggregation needed).
- **GT JSON contract:** flat dict `{player_id(str): jersey_number(int)}`, `-1` = not visible.
- **Submission/metric:** same dict format; **accuracy** (EvalAI). No official eval script in repo.
- **Model / checkpoint / entrypoints:** **none in repo.**
- **Download:** SoccerNet pip `downloadDataTask(task="jersey-2023")` (not used; data already local).
- **Hardcoded paths / RAM / OS assumptions:** N/A (no code).

## 3. Environment

ai-dev verified (`pip check` clean; `timm` not installed). No `sn-jersey` conda env. No packages changed.

## 4. Code / compile / tests

No `.py`, no notebooks → `compileall` exit 0 (nothing to compile), nothing to import, no CLI, **no import-time side effects or auto-downloads possible.**

## 5. Checkpoint inventory

Scanned repo + `/home/ahmet/models` + `/home/ahmet/workspace` + `/home/ahmet/projects/soccernet` for `.pt/.pth/.ckpt/.onnx/.safetensors/.pkl` and jersey-specific weights. **No jersey-recognition checkpoint found.** Only unrelated models (`yolo11n.pt`, `sn-banner/SV_*.pth`, spotting `.pkl`) — not usable. Repo defines no model, so no checkpoint could be compatible. → **BLOCKED_CHECKPOINT_MISSING** (no random/unrelated weights used).

## 6–7. Dataset structure & GT JSON validation

`/mnt/c/football_data/datasets/SoccerNet/jersey-2023` (download marked COMPLETE), matches the documented format.

| Split | Tracklets (=GT=folders) | Known # | `-1` | ≥20 frames | Empty | Total images |
|-------|------|------|------|------|------|------|
| train | 1427 | 1024 | 403 | 1419 | 0 | 733,001 |
| test  | 1211 | 856  | 355 | 1197 | 0 | 564,547 |

- GT schema: keys all `str`, values all `int`, `-1` handled, known numbers within 0–99. **Exact GT↔image-folder match** for both splits (no orphan folders, no missing folders).
- Deterministic sample (100+100 tracklets, ≤5 imgs each = 1000 images decoded): **0 zero-byte, 0 corrupt, 1000/1000 decoded OK.** Variable thumbnail sizes (train w[3,151]/h[8,188], test w[5,171]/h[6,201]) — extreme low-res crops, handled.
- Frame-count distribution supports the **≥20-frame filter** (train 1419/1427, test 1197/1211 qualify).

## 8. Loader real-run test (independent reference reader — repo has none)

A football-analytics-side reference `Dataset`/`DataLoader` loaded **real train & test** tracklets into batched tensors:
- bs=1/4, num_workers=0/2 → shape `[B, 8, 3, 64, 64]`, dtype `float32`, correct labels (jersey numbers) + player_ids.
- **Deterministic:** same item → identical tensor. (This validates data-loading only, not recognition.)

## 9. Model forward smoke

**MODEL_ARCHITECTURE_SMOKE_PASS** — generic `torchvision.resnet18(weights=None)` (random init, **no download**) forward on a real batch `[32,3,64,64]` → `[32,1000]` on **CPU and CUDA**, no NaN/Inf; `TORCH_HOME` confirmed empty afterward. This is **not** sn-jersey's model and **not** recognition.

## 10. Real recognition inference

**BLOCKED_CHECKPOINT_MISSING** — repo defines no recognition model and no compatible checkpoint exists. Preview video (`jersey_tracklet_preview.mp4`) inference not run for the same reason. Not run with random weights.

## 11. Evaluation metric smoke

Reimplemented the documented **accuracy** metric over `{player_id: number}` on a real `test[:50]` subset. Correctly separates scenarios: **perfect=1.0, wrong_number=0.3, all_-1=0.3, missing_half=0.5** → `separates_perfect_from_broken = True`. (Validates the evaluation contract, not recognition.)

## 12. football-analytics compatibility

See `sn_jersey_integration_mapping.md`. Dataset is production-usable now; a jersey stage would need crop-generation from `tracks.parquet`, ≥20-frame/min-size filtering, temporal aggregation → `{track_id: number, confidence}` joined to `track_identities.parquet`, with `-1`/unknown and fragmentation/scene-cut handling. **Blocked today** by the absent model+checkpoint; recommendation: train a recognition model on jersey-2023 (data available).

## 13. Artifacts (all present, non-empty, parseable)

`/home/ahmet/projects/football-analytics/artifacts/soccernet_repo_tests/sn-jersey/`
- `repo_inventory.json`, `checkpoint_inventory.json`, `dataset_schema.json`, `dataset_sample_validation.json`, `label_distribution.json`, `loader_test_results.json`, `model_forward_results.json`, `evaluation_smoke_results.json`, `inference_results.json`, `sample_contact_sheet.jpg` (12 cells, GT numbers), `repo_initial_diff.patch` (empty = clean).

## 14. Result

| Metric | Value |
|--------|-------|
| Tests passed | 11 |
| Tests failed | 0 |
| Loader (train/test) | PASS (reference reader) |
| Dataset | train 1427 / test 1211 tracklets, valid GT, 0 corrupt sampled |
| Checkpoint | **BLOCKED_CHECKPOINT_MISSING** (repo has no model) |
| Model forward | MODEL_ARCHITECTURE_SMOKE_PASS (generic, weights=None) |
| Real recognition inference | **BLOCKED** |
| Evaluation | PASS (synthetic; separates perfect/broken) |
| Overall | **PARTIAL** |

**Open blockers:**
1. Repo is **README-only** — no baseline loader/model/training/inference/evaluation code.
2. **No recognition model + no compatible checkpoint** → real jersey-number inference impossible.
3. LICENSE_NOT_VERIFIED.

**Log:** `/home/ahmet/projects/football-analytics/logs/soccernet_repo_tests/05_sn_jersey.log`
