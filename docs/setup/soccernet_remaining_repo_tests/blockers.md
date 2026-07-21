# Remaining External / Runtime Blockers

_Generated: 2026-07-18T15:47:42.171691+00:00_

## nvcc / CUDA build

- **sn-nvs**: COMPILE_PASS; requires compiled CUDA rasterizers (diff_gaussian_rasterization) — NOT_BUILT; nvcc absent (NVCC_ABSENT). Cannot build per policy.

## Dependency conflict (OpenMMLab / Java)

- **sn-banner**: COMPILE_PASS 108/108; official checkpoint downloaded (833MB, SHA recorded) but load requires mmengine/mmseg (OpenMMLab) which is not installed and conflicts with ai-dev; LOAD_ERR No module named 'mmengine'.
- **sn-caption**: COMPILE_PASS 10/10; DenseVideoCaptioning evaluator imports (pycocoevalcap present) but METEOR/PTBTokenizer require Java (JAVA_ABSENT) -> full metric run blocked; model needs checkpoint+features.

## No published checkpoint / needs isolated model env

- **sn-teamspotting**: COMPILE_PASS 15/15; model needs timm (missing) + published checkpoint (none found in README); shared ActionSpotting evaluator validated (perfect vs imperfect).
- **sn-depth**: COMPILE_PASS 56/56; needs zoedepth+timm (missing) and ZoeDepth checkpoint (not fetched); no direct public checkpoint link in repo README.
- **PTS-baseline**: COMPILE_PASS 39/39; model needs timm (missing) + trained checkpoint (none in README). Shared ActionSpotting metric validated.
- **SoccerNet-v3**: COMPILE_PASS 3/3; dataloader imports torch + SoccerNet.Evaluation.utils, which are not co-installed in one env (ai-dev lacks Evaluation; sn-eval-env lacks torch). Dataset devkit overlapping SoccerNet SDK.

## Devkit works, model needs checkpoint/data

- **sn-mvfoul**: COMPILE_PASS 16/16; official MV_FoulRecognition evaluator validated on synthetic perfect vs imperfect (action acc 100->62.5, offence/severity 100->lower). Model needs multi-view video data + checkpoint (BLOCKED_DATA_ACCESS/MISSING_CHECKPOINT).
- **sn-spotting**: COMPILE_PASS 72/72; official ActionSpotting evaluator validated (perfect a_mAP high vs imperfect lower). Baseline models need torch_geometric (missing) + checkpoints (not present; only PCA feature pkls on disk).
- **sn-reid**: COMPILE_PASS 171/171; official ReIdentification evaluator validated (mAP 100% perfect vs 33% bad; rank-1 100% vs 0%). Model needs torchreid build (Cython) + checkpoint.
- **sn-tracking**: COMPILE_PASS 10/10; trackeval MOT evaluator validated (perfect HOTA 1.0/IDSW 0 vs id-switch HOTA 0.816/IDSW 2). Detector (YOLOX) needs yolox+torch2trt+checkpoint. football-analytics already has a production trackeval_adapter (compatible).
- **sn-grounding**: COMPILE_PASS 34/34; official ReplayGrounding evaluator validated (perfect a_mAP >= imperfect). Baseline model (CALF) needs features + checkpoint.

## Notes

- nvcc/CUDA Toolkit intentionally NOT installed (policy).
- ai-dev PyTorch/CUDA/torchvision left untouched (policy).
- No NDA/paywalled SoccerNet videos were downloaded.
- sn-caption METEOR/PTBTokenizer need a JRE (system apt); not installed (policy).
