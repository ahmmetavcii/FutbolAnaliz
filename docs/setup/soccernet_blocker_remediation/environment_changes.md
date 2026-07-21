# Environment Changes

_Generated: 2026-07-18T17:00:38.674666+00:00_

## ai-dev (PROTECTED — unchanged)
- Python 3.10.20, torch `2.11.0+cu128`, CUDA 12.8
- Evidence: `system_before.txt` vs `ai_dev_torch_after.txt`

## New / updated isolated envs
| Env | Purpose | Key packages |
|---|---|---|
| sn-banner-runtime | OpenMMLab Mask2Former | py3.11, torch 2.1.2+cu121, mmcv 2.1.0, mmseg 1.2.2 |
| sn-caption-eval | Caption evaluator + Java | py3.10, conda-forge openjdk 11, SoccerNet, pycocoevalcap |
| sn-nvs-build | CUDA extension build | py3.10, torch 2.1.2+cu121, conda cuda-nvcc 12.1 |
| sn-depth-runtime | ZoeDepth | py3.9, torch 1.13.1+cu117, timm 0.6.12, numpy 1.23.5 |
| sn-pts-baseline | PTS E2E-Spot (updated) | py3.10, torch upgraded to 2.1.2+cu121 (env-only), timm 0.6.13 |
| sn-teamspotting | T-DEED (existing) | py3.10, torch 2.5.0+cpu, timm 1.0.11 |
| sn-gamestate-env | TrackLab (existing) | py3.9, torch 1.13.1+cu117 |

System NVIDIA driver and Windows driver untouched. System-wide CUDA Toolkit NOT installed.
nvcc lives only inside `sn-nvs-build` conda env.
