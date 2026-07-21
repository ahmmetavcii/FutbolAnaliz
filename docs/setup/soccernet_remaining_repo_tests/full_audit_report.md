# Remaining 13 SoccerNet Repositories — Full Audit Report

_Generated: 2026-07-18T15:47:42.171691+00:00_

All 13 repos are fully cloned with real source code and **100% py_compile pass**. 
Statuses separate ORIGINAL vs COMPATIBLE and never mark unverified features as PASS.

## Summary counts

- Total repos: 13
- Cloned: 13
- With source code: 13
- README-only: 0
- Compile pass: 13/13

## Original status distribution

- **BLOCKED_BUILD_NO_NVCC**: sn-nvs
- **BLOCKED_DEPENDENCY_CONFLICT**: sn-banner, sn-caption
- **ORIGINAL_SMOKE_PASS**: ActiveSpotting
- **PASS_DEVKIT**: sn-mvfoul, sn-spotting, sn-reid, sn-tracking, sn-grounding
- **SOURCE_ONLY_NO_MODEL**: sn-teamspotting, sn-depth, PTS-baseline, SoccerNet-v3

## Per-repository detail

### sn-banner  (`f6d50b24a33d`)
- Purpose: Broadcast banner replacement (segmentation + calibration + compositing)
- License: GPL-3.0
- Environment: ai-dev (static/compile) — real run needs isolated env
- Compile: COMPILE_PASS (108/108)
- Original status: **BLOCKED_DEPENDENCY_CONFLICT**
- Compatible status: NOT_IMPLEMENTED
- Real model inference: no
- Checkpoint: official available
- Evidence: COMPILE_PASS 108/108; official checkpoint downloaded (833MB, SHA recorded) but load requires mmengine/mmseg (OpenMMLab) which is not installed and conflicts with ai-dev; LOAD_ERR No module named 'mmengine'.
- Artifacts: `/home/ahmet/workspace/soccernet_remaining_repo_tests/artifacts/sn-banner`

### sn-nvs  (`1655ab19b3bd`)
- Purpose: Novel View Synthesis (Gaussian/Triangle splatting)
- License: LICENSE_NOT_FOUND (root); submodules Gaussian/Triangle-Splatting research non-commercial
- Environment: ai-dev (static/compile) — real run needs isolated env
- Compile: COMPILE_PASS (107/107)
- Original status: **BLOCKED_BUILD_NO_NVCC**
- Compatible status: N/A
- Real model inference: no
- Checkpoint: none found/published
- Evidence: COMPILE_PASS; requires compiled CUDA rasterizers (diff_gaussian_rasterization) — NOT_BUILT; nvcc absent (NVCC_ABSENT). Cannot build per policy.
- Artifacts: `/home/ahmet/workspace/soccernet_remaining_repo_tests/artifacts/sn-nvs`

### sn-teamspotting  (`091fed2fc35c`)
- Purpose: Team/ball action spotting model (T-DEED)
- License: GPL-3.0
- Environment: ai-dev (static/compile) — real run needs isolated env
- Compile: COMPILE_PASS (15/15)
- Original status: **SOURCE_ONLY_NO_MODEL**
- Compatible status: NOT_IMPLEMENTED
- Real model inference: no
- Checkpoint: none found/published
- Evidence: COMPILE_PASS 15/15; model needs timm (missing) + published checkpoint (none found in README); shared ActionSpotting evaluator validated (perfect vs imperfect).
- Artifacts: `/home/ahmet/workspace/soccernet_remaining_repo_tests/artifacts/sn-teamspotting`

### sn-depth  (`9f6636fafb11`)
- Purpose: Monocular depth estimation for football (ZoeDepth baseline)
- License: LICENSE_NOT_FOUND (root); baseline ZoeDepth MIT
- Environment: ai-dev (static/compile) — real run needs isolated env
- Compile: COMPILE_PASS (56/56)
- Original status: **SOURCE_ONLY_NO_MODEL**
- Compatible status: NOT_IMPLEMENTED
- Real model inference: no
- Checkpoint: none found/published
- Evidence: COMPILE_PASS 56/56; needs zoedepth+timm (missing) and ZoeDepth checkpoint (not fetched); no direct public checkpoint link in repo README.
- Artifacts: `/home/ahmet/workspace/soccernet_remaining_repo_tests/artifacts/sn-depth`

### sn-mvfoul  (`502fb44a76c2`)
- Purpose: Multi-view foul recognition (VARS) + evaluator
- License: GPL-3.0
- Environment: sn-eval-env (SoccerNet devkit)
- Compile: COMPILE_PASS (16/16)
- Original status: **PASS_DEVKIT**
- Compatible status: NOT_IMPLEMENTED
- Real model inference: no
- Checkpoint: none found/published
- Evidence: COMPILE_PASS 16/16; official MV_FoulRecognition evaluator validated on synthetic perfect vs imperfect (action acc 100->62.5, offence/severity 100->lower). Model needs multi-view video data + checkpoint (BLOCKED_DATA_ACCESS/MISSING_CHECKPOINT).
- Artifacts: `/home/ahmet/workspace/soccernet_remaining_repo_tests/artifacts/sn-mvfoul`

### sn-caption  (`c05973d4f008`)
- Purpose: Dense video captioning benchmark + evaluator
- License: LICENSE_NOT_FOUND (root); benchmark TemporallyAwarePooling license present
- Environment: ai-dev (static/compile) — real run needs isolated env
- Compile: COMPILE_PASS (10/10)
- Original status: **BLOCKED_DEPENDENCY_CONFLICT**
- Compatible status: NOT_IMPLEMENTED
- Real model inference: no
- Checkpoint: none found/published
- Evidence: COMPILE_PASS 10/10; DenseVideoCaptioning evaluator imports (pycocoevalcap present) but METEOR/PTBTokenizer require Java (JAVA_ABSENT) -> full metric run blocked; model needs checkpoint+features.
- Artifacts: `/home/ahmet/workspace/soccernet_remaining_repo_tests/artifacts/sn-caption`

### sn-spotting  (`9842826f94e1`)
- Purpose: Action spotting benchmark (CALF/NetVLAD++/…) + evaluator
- License: MIT
- Environment: sn-eval-env (SoccerNet devkit)
- Compile: COMPILE_PASS (72/72)
- Original status: **PASS_DEVKIT**
- Compatible status: NOT_IMPLEMENTED
- Real model inference: no
- Checkpoint: none found/published
- Evidence: COMPILE_PASS 72/72; official ActionSpotting evaluator validated (perfect a_mAP high vs imperfect lower). Baseline models need torch_geometric (missing) + checkpoints (not present; only PCA feature pkls on disk).
- Artifacts: `/home/ahmet/workspace/soccernet_remaining_repo_tests/artifacts/sn-spotting`

### sn-reid  (`621e2b0f2d2a`)
- Purpose: Player re-identification benchmark (torchreid fork) + evaluator
- License: MIT
- Environment: sn-eval-env (SoccerNet devkit)
- Compile: COMPILE_PASS (171/171)
- Original status: **PASS_DEVKIT**
- Compatible status: NOT_IMPLEMENTED
- Real model inference: no
- Checkpoint: none found/published
- Evidence: COMPILE_PASS 171/171; official ReIdentification evaluator validated (mAP 100% perfect vs 33% bad; rank-1 100% vs 0%). Model needs torchreid build (Cython) + checkpoint.
- Artifacts: `/home/ahmet/workspace/soccernet_remaining_repo_tests/artifacts/sn-reid`

### sn-tracking  (`b0bbba35e07f`)
- Purpose: Multi-object tracking benchmark (DeepSORT/ByteTrack) + trackeval eval
- License: LICENSE_NOT_FOUND (root); benchmarks DeepSORT/ByteTrack MIT
- Environment: sn-eval-env (SoccerNet devkit)
- Compile: COMPILE_PASS (10/10)
- Original status: **PASS_DEVKIT**
- Compatible status: EXISTING_COMPATIBLE_PASS
- Real model inference: no
- Checkpoint: none found/published
- Evidence: COMPILE_PASS 10/10; trackeval MOT evaluator validated (perfect HOTA 1.0/IDSW 0 vs id-switch HOTA 0.816/IDSW 2). Detector (YOLOX) needs yolox+torch2trt+checkpoint. football-analytics already has a production trackeval_adapter (compatible).
- Artifacts: `/home/ahmet/workspace/soccernet_remaining_repo_tests/artifacts/sn-tracking`

### ActiveSpotting  (`33a81cb83497`)
- Purpose: Active-learning action spotting (NetVLAD)
- License: MIT
- Environment: ai-dev (pure torch forward)
- Compile: COMPILE_PASS (6/6)
- Original status: **ORIGINAL_SMOKE_PASS**
- Compatible status: NOT_IMPLEMENTED
- Real model inference: YES
- Checkpoint: none found/published
- Evidence: COMPILE_PASS 6/6; real NetVLAD & NetRVLAD GPU forward on synthetic [2,120,512] -> [2,32768], finite, peak VRAM ~10.6MB. Full pipeline needs features+labels; no published checkpoint.
- Artifacts: `/home/ahmet/workspace/soccernet_remaining_repo_tests/artifacts/ActiveSpotting`

### PTS-baseline  (`af2ea8234e0c`)
- Purpose: Precise temporal spotting baseline (E2E-Spot family)
- License: BSD-3-Clause
- Environment: ai-dev (static/compile) — real run needs isolated env
- Compile: COMPILE_PASS (39/39)
- Original status: **SOURCE_ONLY_NO_MODEL**
- Compatible status: NOT_IMPLEMENTED
- Real model inference: no
- Checkpoint: none found/published
- Evidence: COMPILE_PASS 39/39; model needs timm (missing) + trained checkpoint (none in README). Shared ActionSpotting metric validated.
- Artifacts: `/home/ahmet/workspace/soccernet_remaining_repo_tests/artifacts/PTS-baseline`

### sn-grounding  (`910bf859ac6d`)
- Purpose: Replay grounding benchmark + evaluator
- License: MIT
- Environment: sn-eval-env (SoccerNet devkit)
- Compile: COMPILE_PASS (34/34)
- Original status: **PASS_DEVKIT**
- Compatible status: NOT_IMPLEMENTED
- Real model inference: no
- Checkpoint: none found/published
- Evidence: COMPILE_PASS 34/34; official ReplayGrounding evaluator validated (perfect a_mAP >= imperfect). Baseline model (CALF) needs features + checkpoint.
- Artifacts: `/home/ahmet/workspace/soccernet_remaining_repo_tests/artifacts/sn-grounding`

### SoccerNet-v3  (`7d483a85ad62`)
- Purpose: SoccerNet-v3 frame dataset devkit (dataloader/statistics/visualize)
- License: MIT
- Environment: ai-dev (static/compile) — real run needs isolated env
- Compile: COMPILE_PASS (3/3)
- Original status: **SOURCE_ONLY_NO_MODEL**
- Compatible status: N/A
- Real model inference: no
- Checkpoint: none found/published
- Evidence: COMPILE_PASS 3/3; dataloader imports torch + SoccerNet.Evaluation.utils, which are not co-installed in one env (ai-dev lacks Evaluation; sn-eval-env lacks torch). Dataset devkit overlapping SoccerNet SDK.
- Artifacts: `/home/ahmet/workspace/soccernet_remaining_repo_tests/artifacts/SoccerNet-v3`

