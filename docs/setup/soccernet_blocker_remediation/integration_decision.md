# Integration Decision

_Generated: 2026-07-18T17:00:38.674666+00:00_

## sn-banner — DO_NOT_USE
- Needed for main pipeline? DO_NOT_USE
- Overlaps existing modules? see notes
- License risk: GPL-3.0
- Fits RTX 4060 / 16GB? peak VRAM 3.33 GB
- Notes: OpenMMLab mmseg 1.2.2 + mmcv 2.1.0 isolated env; official Mask2Former checkpoint on 5 real football frames; billboard masks non-empty, finite.

## sn-caption — DO_NOT_USE
- Needed for main pipeline? DO_NOT_USE
- Overlaps existing modules? see notes
- License risk: LICENSE_NOT_FOUND (root)
- Fits RTX 4060 / 16GB? peak VRAM 0 GB
- Notes: conda-forge OpenJDK 11 in env; ORIGINAL DenseVideoCaptioning evaluator; perfect>partial>wrong on all metrics. Evaluator PASS ≠ model inference.

## sn-nvs — RESEARCH_ONLY
- Needed for main pipeline? RESEARCH_ONLY
- Overlaps existing modules? see notes
- License risk: LICENSE_NOT_FOUND (root); GS research non-commercial
- Fits RTX 4060 / 16GB? peak VRAM 0.018 GB
- Notes: conda cuda-nvcc 12.1 built ORIGINAL diff_gaussian_rasterization + simple-knn; real GPU kernel forward on 5000 synthetic Gaussians -> 3x360x640 finite render. Not scene-quality NVS inference.

## sn-gamestate — OPTIONAL
- Needed for main pipeline? OPTIONAL
- Overlaps existing modules? see notes
- License risk: GPL-3.0
- Fits RTX 4060 / 16GB? peak VRAM 4.5 GB
- Notes: ORIGINAL TrackLab pipeline exit 0 on 20f and 50f @640w with low-memory Hydra overrides (batch_size=1/4, visualizer off). 50f: 725 dets, 23 tracks, pitch coords 100%, roles present. Compatible status unchanged.

## PTS-baseline — OPTIONAL
- Needed for main pipeline? OPTIONAL
- Overlaps existing modules? see notes
- License risk: BSD-3-Clause
- Fits RTX 4060 / 16GB? peak VRAM None GB
- Notes: Official published checkpoint from jhong93/e2e-spot-models (linked in README). ORIGINAL E2EModel.predict on 100 real football frames; finite softmax scores (100,18).

## sn-teamspotting — OPTIONAL
- Needed for main pipeline? OPTIONAL
- Overlaps existing modules? see notes
- License risk: GPL-3.0
- Fits RTX 4060 / 16GB? peak VRAM 0 GB
- Notes: Official T-DEED baseline checkpoint from README Drive folder. ORIGINAL TDEEDModel.predict on 100 real frames (796x448) CPU; finite pred (1,100).

## sn-depth — RESEARCH_ONLY
- Needed for main pipeline? RESEARCH_ONLY
- Overlaps existing modules? see notes
- License risk: LICENSE_NOT_FOUND (root); ZoeDepth MIT
- Fits RTX 4060 / 16GB? peak VRAM 2.23 GB
- Notes: Official SoccerNet-Depth weights + ORIGINAL ZoeDepth.infer_pil on real football frame; depth 540x960 finite, min/max/mean recorded, PNG reopened.

## SoccerNet-v3 — DO_NOT_USE
- Needed for main pipeline? DO_NOT_USE
- Overlaps existing modules? see notes
- License risk: MIT
- Fits RTX 4060 / 16GB? peak VRAM 0 GB
- Notes: Dataset/devkit overlapping SoccerNet SDK; no published model checkpoint.

## sn-mvfoul — DO_NOT_USE
- Needed for main pipeline? DO_NOT_USE
- Overlaps existing modules? see notes
- License risk: GPL-3.0
- Fits RTX 4060 / 16GB? peak VRAM 0 GB
- Notes: Devkit evaluator remains PASS. Real model inference blocked by multi-view NDA/data access. Not a local dependency issue.

## sn-trackeval — USE (via existing adapter)
- Needed for main pipeline? USE (via existing adapter)
- Overlaps existing modules? see notes
- License risk: MIT
- Fits RTX 4060 / 16GB? peak VRAM 0 GB
- Notes: Core evaluator works. Official leaderboard GT not available locally — EXTERNAL_BLOCKER_OFFICIAL_GT. Do not claim leaderboard PASS.

## sn-jersey — USE (clean-room)
- Needed for main pipeline? USE (clean-room)
- Overlaps existing modules? see notes
- License risk: README-only (no source)
- Fits RTX 4060 / 16GB? peak VRAM 0 GB
- Notes: Original repo is README-only — not a dependency problem. Compatible clean-room remains PASS. Original checkpoint not touched.

## sn-echoes — USE (reader)
- Needed for main pipeline? USE (reader)
- Overlaps existing modules? see notes
- License risk: LICENSE_REVIEW_REQUIRED
- Fits RTX 4060 / 16GB? peak VRAM 0 GB
- Notes: Dataset/devkit — not an ASR inference model. Do not treat as model remediation target.

