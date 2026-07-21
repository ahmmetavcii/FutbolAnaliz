# SoccerNet Blocker Remediation — Full Report

_Generated: 2026-07-18T17:00:38.674666+00:00_

ai-dev PyTorch/CUDA left untouched (`2.11.0+cu128` / CUDA 12.8).
Original and compatible statuses are never mixed.
Synthetic CUDA forward ≠ real video inference; evaluator PASS ≠ model PASS.

## Per-repository

### sn-banner
- Previous original: `BLOCKED_DEPENDENCY_CONFLICT` → New: **`ORIGINAL_INFERENCE_PASS`**
- Compatible: `NOT_IMPLEMENTED`
- Env: `sn-banner-runtime`
- Real inference: YES
- Checkpoint: official HF best_mIoU_iter_10935.pth (833MB)
- Peak RAM/VRAM: None GB / 3.33 GB
- Remaining blocker: none
- Notes: OpenMMLab mmseg 1.2.2 + mmcv 2.1.0 isolated env; official Mask2Former checkpoint on 5 real football frames; billboard masks non-empty, finite.
- Artifacts: `/home/ahmet/workspace/soccernet_blocker_remediation/artifacts/sn-banner`

### sn-caption
- Previous original: `BLOCKED_DEPENDENCY_CONFLICT` → New: **`PASS_DEVKIT`**
- Compatible: `NOT_IMPLEMENTED`
- Env: `sn-caption-eval`
- Real inference: no
- Checkpoint: CHECKPOINT_NOT_PUBLISHED
- Peak RAM/VRAM: None GB / 0 GB
- Remaining blocker: CHECKPOINT_NOT_PUBLISHED
- Notes: conda-forge OpenJDK 11 in env; ORIGINAL DenseVideoCaptioning evaluator; perfect>partial>wrong on all metrics. Evaluator PASS ≠ model inference.
- Artifacts: `/home/ahmet/workspace/soccernet_blocker_remediation/artifacts/sn-caption`

### sn-nvs
- Previous original: `BLOCKED_BUILD_NO_NVCC` → New: **`ORIGINAL_SMOKE_PASS`**
- Compatible: `N/A`
- Env: `sn-nvs-build`
- Real inference: no
- Checkpoint: CHECKPOINT_NOT_PUBLISHED (requires training on SN-NVS dataset)
- Peak RAM/VRAM: None GB / 0.018 GB
- Remaining blocker: CHECKPOINT_NOT_PUBLISHED + EXTERNAL_BLOCKER_DATA_ACCESS (SN-NVS scenes)
- Notes: conda cuda-nvcc 12.1 built ORIGINAL diff_gaussian_rasterization + simple-knn; real GPU kernel forward on 5000 synthetic Gaussians -> 3x360x640 finite render. Not scene-quality NVS inference.
- Artifacts: `/home/ahmet/workspace/soccernet_blocker_remediation/artifacts/sn-nvs`

### sn-gamestate
- Previous original: `BLOCKED_RUNTIME_OOM` → New: **`ORIGINAL_INFERENCE_PASS`**
- Compatible: `COMPATIBLE_IMPLEMENTATION_PASS`
- Env: `sn-gamestate-env`
- Real inference: YES
- Checkpoint: existing open TrackLab assets (YOLO/PRTReId/NBJW)
- Peak RAM/VRAM: 4.5 GB / 4.5 GB
- Remaining blocker: none
- Notes: ORIGINAL TrackLab pipeline exit 0 on 20f and 50f @640w with low-memory Hydra overrides (batch_size=1/4, visualizer off). 50f: 725 dets, 23 tracks, pitch coords 100%, roles present. Compatible status unchanged.
- Artifacts: `/home/ahmet/workspace/soccernet_blocker_remediation/artifacts/sn-gamestate`

### PTS-baseline
- Previous original: `SOURCE_ONLY_NO_MODEL` → New: **`ORIGINAL_INFERENCE_PASS`**
- Compatible: `NOT_IMPLEMENTED`
- Env: `sn-pts-baseline`
- Real inference: YES
- Checkpoint: official e2e-spot-models soccer_rny002gsm_gru_rgb/checkpoint_088.pt (18MB)
- Peak RAM/VRAM: None GB / None GB
- Remaining blocker: none
- Notes: Official published checkpoint from jhong93/e2e-spot-models (linked in README). ORIGINAL E2EModel.predict on 100 real football frames; finite softmax scores (100,18).
- Artifacts: `/home/ahmet/workspace/soccernet_blocker_remediation/artifacts/PTS-baseline`

### sn-teamspotting
- Previous original: `SOURCE_ONLY_NO_MODEL` → New: **`ORIGINAL_INFERENCE_PASS`**
- Compatible: `NOT_IMPLEMENTED`
- Env: `sn-teamspotting`
- Real inference: YES
- Checkpoint: official Drive SoccerNetBall_baseline/checkpoint_best.pt (48MB)
- Peak RAM/VRAM: None GB / 0 GB
- Remaining blocker: none
- Notes: Official T-DEED baseline checkpoint from README Drive folder. ORIGINAL TDEEDModel.predict on 100 real frames (796x448) CPU; finite pred (1,100).
- Artifacts: `/home/ahmet/workspace/soccernet_blocker_remediation/artifacts/sn-teamspotting`

### sn-depth
- Previous original: `SOURCE_ONLY_NO_MODEL` → New: **`ORIGINAL_INFERENCE_PASS`**
- Compatible: `NOT_IMPLEMENTED`
- Env: `sn-depth-runtime`
- Real inference: YES
- Checkpoint: official Drive ZoeDepthN_football.pt (3.9GB)
- Peak RAM/VRAM: None GB / 2.23 GB
- Remaining blocker: none
- Notes: Official SoccerNet-Depth weights + ORIGINAL ZoeDepth.infer_pil on real football frame; depth 540x960 finite, min/max/mean recorded, PNG reopened.
- Artifacts: `/home/ahmet/workspace/soccernet_blocker_remediation/artifacts/sn-depth`

### SoccerNet-v3
- Previous original: `SOURCE_ONLY_NO_MODEL` → New: **`CHECKPOINT_NOT_PUBLISHED`**
- Compatible: `N/A`
- Env: `ai-dev (static)`
- Real inference: no
- Checkpoint: CHECKPOINT_NOT_PUBLISHED (dataset/devkit, no model)
- Peak RAM/VRAM: None GB / 0 GB
- Remaining blocker: CHECKPOINT_NOT_PUBLISHED + EXTERNAL_BLOCKER_DATA_ACCESS (Frames-v3)
- Notes: Dataset/devkit overlapping SoccerNet SDK; no published model checkpoint.
- Artifacts: `/home/ahmet/workspace/soccernet_blocker_remediation/artifacts/SoccerNet-v3`

### sn-mvfoul
- Previous original: `PASS_DEVKIT` → New: **`PASS_DEVKIT`**
- Compatible: `NOT_IMPLEMENTED`
- Env: `sn-eval-env`
- Real inference: no
- Checkpoint: Drive folder exists but multi-view video NDA/access
- Peak RAM/VRAM: None GB / 0 GB
- Remaining blocker: EXTERNAL_BLOCKER_DATA_ACCESS
- Notes: Devkit evaluator remains PASS. Real model inference blocked by multi-view NDA/data access. Not a local dependency issue.
- Artifacts: `/home/ahmet/workspace/soccernet_blocker_remediation/artifacts/sn-mvfoul`

### sn-trackeval
- Previous original: `PASS (evaluator) / NOT_RUN_NO_OFFICIAL_GT` → New: **`PASS_DEVKIT`**
- Compatible: `EXISTING_COMPATIBLE_PASS`
- Env: `sn-trackeval-env / ai-dev`
- Real inference: no
- Checkpoint: N/A
- Peak RAM/VRAM: None GB / 0 GB
- Remaining blocker: EXTERNAL_BLOCKER_OFFICIAL_GT
- Notes: Core evaluator works. Official leaderboard GT not available locally — EXTERNAL_BLOCKER_OFFICIAL_GT. Do not claim leaderboard PASS.
- Artifacts: `/home/ahmet/workspace/soccernet_blocker_remediation/artifacts/sn-trackeval`

### sn-jersey
- Previous original: `N/A_README_ONLY` → New: **`N/A_README_ONLY`**
- Compatible: `COMPATIBLE_CLEANROOM_PASS`
- Env: `sn-jersey-env / ai-dev`
- Real inference: no
- Checkpoint: clean-room /home/ahmet/models/jersey_recognition_v1_best.pt (untouched)
- Peak RAM/VRAM: None GB / 0 GB
- Remaining blocker: none
- Notes: Original repo is README-only — not a dependency problem. Compatible clean-room remains PASS. Original checkpoint not touched.
- Artifacts: `/home/ahmet/workspace/soccernet_blocker_remediation/artifacts/sn-jersey`

### sn-echoes
- Previous original: `PASS (dataset/devkit)` → New: **`PASS_DEVKIT`**
- Compatible: `EXISTING_COMPATIBLE_PASS`
- Env: `ai-dev`
- Real inference: no
- Checkpoint: N/A (no ASR model)
- Peak RAM/VRAM: None GB / 0 GB
- Remaining blocker: none
- Notes: Dataset/devkit — not an ASR inference model. Do not treat as model remediation target.
- Artifacts: `/home/ahmet/workspace/soccernet_blocker_remediation/artifacts/sn-echoes`

