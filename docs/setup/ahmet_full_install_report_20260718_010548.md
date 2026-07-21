# Ahmet Full Football Install Report

**Session:** `20260717_234122`  
**Generated:** 2026-07-18T01:05:48.214187+03:00  
**Overall:** Partially complete — core AI stack READY; SoccerNet matrix mixed by design

## 1. System summary

| Item | Value |
|---|---|
| User | ahmet /home/ahmet |
| Ubuntu | 22.04.5 LTS |
| Kernel | 6.18.33.2-microsoft-standard-WSL2 |
| CPU | Intel Core i7-13700HX (24 threads) |
| WSL RAM observed | 7.6 GiB (host target 16 GB) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8188 MiB |
| Driver | 610.74 (CUDA capability 13.3) |
| Storage policy | code in /home/ahmet ; data in /mnt/c/football_data |

## 2. Disk usage

### Before
```
Filesystem      Size  Used Avail Use% Mounted on
none            3.9G     0  3.9G   0% /usr/lib/modules/6.18.33.2-microsoft-standard-WSL2
none            3.9G  4.0K  3.9G   1% /mnt/wsl
drivers         464G  206G  258G  45% /usr/lib/wsl/drivers
/dev/sdd       1007G  2.4G  954G   1% /
none            3.9G   40K  3.9G   1% /mnt/wslg
none            3.9G     0  3.9G   0% /usr/lib/wsl/lib
rootfs          3.9G  2.8M  3.9G   1% /init
none            3.9G  576K  3.9G   1% /run
none            3.9G     0  3.9G   0% /run/lock
none            3.9G     0  3.9G   0% /run/shm
none            3.9G   80K  3.9G   1% /mnt/wslg/versions.txt
none            3.9G   80K  3.9G   1% /mnt/wslg/doc
C:\             464G  206G  258G  45% /mnt/c
tmpfs           781M  8.0K  781M   1% /run/user/1000
```

### After
```
Filesystem      Size  Used Avail Use% Mounted on
none            3.9G     0  3.9G   0% /usr/lib/modules/6.18.33.2-microsoft-standard-WSL2
none            3.9G  4.0K  3.9G   1% /mnt/wsl
drivers         464G  250G  215G  54% /usr/lib/wsl/drivers
/dev/sdd       1007G   46G  911G   5% /
none            3.9G   40K  3.9G   1% /mnt/wslg
none            3.9G     0  3.9G   0% /usr/lib/wsl/lib
rootfs          3.9G  2.8M  3.9G   1% /init
none            3.9G  580K  3.9G   1% /run
none            3.9G     0  3.9G   0% /run/lock
none            3.9G     0  3.9G   0% /run/shm
none            3.9G   80K  3.9G   1% /mnt/wslg/versions.txt
none            3.9G   80K  3.9G   1% /mnt/wslg/doc
C:\             464G  250G  215G  54% /mnt/c
tmpfs           781M  8.0K  781M   1% /run/user/1000
```

## 3. Main ai-dev environment

| Package | Version |
|---|---|
| python | 3.10.20 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| torchaudio | 2.11.0+cu128 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| opencv | 5.0.0.93 |
| ultralytics | 8.4.91 |
| SoccerNet | 0.1.62 |
| cuda_available | True |
| pip_check | clean |
| check_env | PASS |
| jupyter_kernel | Python (ai-dev) |

Artifacts:
- `requirements/ai-dev-freeze.txt`
- `requirements/ai-dev-conda-full.yml`
- `requirements/ai-dev-conda-history.yml`

## 4. GPU / CUDA verification

- `torch.cuda.is_available() = True`
- Device name verified as RTX 4060 Laptop GPU
- Small CUDA tensor + matmul PASS
- YOLO nano image/video tracking smoke PASS
- Peak VRAM during smoke: 50818048 bytes
- Processing FPS: 25.34

## 5. Jupyter / Git / folders

- Jupyter kernel: `Python (ai-dev)` registered
- Git: user.name=`Ahmet Avcı`, email=`avciahmet1001@gmail.com`, LFS initialized
- Linux tree under `/home/ahmet/{dev-check,projects,models,workspace,logs}`
- Windows SSD tree under `/mnt/c/football_data/...`

## 6. Repository lock

| Repo | Full commit | Status |
|---|---|---|
| sn-tracking | `b0bbba35e07ff58010b6313ef8aa59ef663ad392` | CLONED_NOT_INSTALLED |
| sn-trackeval | `9c25232f6f2b56c9f203f1eb55784ff1e97df683` | EVALUATION_READY |
| sn-spotting | `9842826f94e1419580a9d17219c11aca7225f7ce` | CLONED_NOT_INSTALLED |
| sn-calibration | `ab38f461bec729fead86b6986839de1bb826f16d` | READY_FOR_NONVIDEO_USE |
| sn-reid | `621e2b0f2d2a7a3e207b8dd747542b6608bf72db` | READY_FOR_NONVIDEO_USE |
| sn-grounding | `910bf859ac6d7aff2b80a6d66155956254f24c6b` | CLONED_NOT_INSTALLED |
| SoccerNet-v3 | `7d483a85ad62b5e98f59427eabee8cb87c710d7b` | CLONED_NOT_INSTALLED |
| sn-jersey | `2f43b48c59eefe0bb5d948888db07f55f51208ad` | DATASET_DEVKIT_READY |
| sn-caption | `c05973d4f00853e208d54965f4d6fa47364b8d66` | CLONED_NOT_INSTALLED |
| SoccerNet | `74461027ac2095ce2f8d4ee991eccb5dd5f42459` | READY_FOR_NONVIDEO_USE |
| PTS-baseline | `af2ea8234e0c887758ef674071da2a17e2bc6c61` | PARTIALLY_READY |
| sn-mvfoul | `502fb44a76c254e332394f095d54abc830131a44` | READY_FOR_VIDEO_INPUT |
| ActiveSpotting | `33a81cb834978ee474ecec0c5a76b6f3f99b4bf4` | PARTIALLY_READY |
| sn-gamestate | `1c958345067218297d221e45e1a6405f975f83e0` | READY_FOR_DATASET_VIDEO_SMOKE_TEST |
| sn-depth | `9f6636fafb11447a5bada765e197928ee9efc467` | CLONED_NOT_INSTALLED |
| sn-echoes | `7105a85b7a8c1c000a31a30d0c29c388105c3de5` | DATASET_DEVKIT_READY |
| sn-teamspotting | `091fed2fc35c33f7489f3596958a2fe385e37d65` | READY_FOR_VIDEO_INPUT |
| sn-banner | `f6d50b24a33d6705d4c04dc4d4d93ecd12b08e74` | PARTIALLY_READY |
| sn-nvs | `1655ab19b3bd78f624a96d0f0c27ec2c9f550f61` | BLOCKED_BUILD |
| TrackLab | `5767e86c32a6d6c68e2fc8ae7311f558fff6c7b2` | CLONED_NOT_INSTALLED |
| PnLCalib | `8c87391d6f4ea40c5e4d65e61529916c7a49ce62` | CLONED_NOT_INSTALLED |
| No-Bells-Just-Whistles | `bd993b31c2917096c23bb8aadf148314d17f8345` | PARTIALLY_READY |

## 7. Environments

| Env | Python | Disk |
|---|---|---|
| ai-dev | Python 3.10.20 | 8.1 GiB |
| sn-trackeval | Python 3.10.20 | 663.1 MiB |
| sn-gamestate-python | Python 3.9.23 | 197.3 MiB |
| sn-gamestate | Python 3.9.23 | 4.5 GiB |
| sn-calibration | Python 3.10.20 | 4.7 GiB |
| sn-teamspotting | Python 3.10.20 | 1.6 GiB |
| sn-mvfoul | Python 3.9.23 | 1.8 GiB |
| sn-pts-baseline | Python 3.10.20 | 1.4 GiB |
| sn-reid | Python 3.10.20 | 1.5 GiB |
| sn-echoes | Python 3.10.20 | 137.4 MiB |
| sn-active-spotting | Python 3.10.20 | 1.2 GiB |
| sn-banner-mmseg | Python 3.10.20 | 4.9 GiB |
| sn-banner-replacement | Python 3.10.20 | 1.6 GiB |
| sn-nvs | Python 3.10.20 | 137.4 MiB |

## 8. Repo results

| Repo | Status | Blocker |
|---|---|---|
| sn-trackeval | EVALUATION_READY | — |
| sn-gamestate | READY_FOR_DATASET_VIDEO_SMOKE_TEST | Zenodo/model weights and SoccerNetGS dataset not downloaded by policy |
| sn-echoes | DATASET_DEVKIT_READY | — |
| sn-teamspotting | READY_FOR_VIDEO_INPUT | Google Drive checkpoint + HF/NDA videos not downloaded |
| sn-mvfoul | READY_FOR_VIDEO_INPUT | NDA passworded dataset required; mvtorch PyPI package unavailable (not required for VARS model core) |
| sn-calibration | READY_FOR_NONVIDEO_USE | Google Drive network weights not downloaded |
| PTS-baseline | PARTIALLY_READY | No trained model/frame directory available for inference smoke |
| ActiveSpotting | PARTIALLY_READY | Legacy CUDA 10.1 wheels not forced; needs SoccerNet feature npy files |
| SoccerNet | READY_FOR_NONVIDEO_USE | — |
| sn-banner | PARTIALLY_READY | Mask2Former HF weight + BannerReplacement dataset not downloaded; TTA needs >=24GB VRAM |
| sn-reid | READY_FOR_NONVIDEO_USE | — |
| sn-nvs | BLOCKED_BUILD | Intentionally deferred: CUDA Toolkit / nvcc not installed to avoid risking the stable `ai-dev` stack; sn-nvs is out of MVP scope |
| sn-tracking | CLONED_NOT_INSTALLED | ByteTrack/DeepSORT external stacks not installed |
| sn-spotting | CLONED_NOT_INSTALLED | Multiple benchmark envs + features/videos not installed |
| sn-grounding | CLONED_NOT_INSTALLED | Dataset/features required |
| SoccerNet-v3 | CLONED_NOT_INSTALLED | SoccerNet-v3 annotations/images not downloaded |
| sn-jersey | DATASET_DEVKIT_READY | — |
| sn-caption | CLONED_NOT_INSTALLED | Dense caption data/features not installed |
| sn-depth | CLONED_NOT_INSTALLED | ZoeDepth env + synthetic depth dataset not installed |

## 9. Models

- **yolo11n.pt**: 5.4 MiB, sha256=`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`, status=DOWNLOADED_VERIFIED
- **SV_kp.pth**: 252.7 MiB, sha256=`7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113`, status=DOWNLOADED_VERIFIED
- **SV_lines.pth**: 252.6 MiB, sha256=`2751242917f8c0f858a396e0cfe4521be39fe07bf049590eb21714526acecac1`, status=DOWNLOADED_VERIFIED

## 10. Patches

- Applied `patches/sn-echoes-stats-syntax.patch` (trailing `s` syntax bug in `stats.py`).
- Apply/revert commands documented in `patches/sn-echoes-stats-syntax.md`.

## 11. Failed / blocked attempts

- sn-nvs remains **BLOCKED_BUILD** by design for this install: CUDA Toolkit 12.8 / `nvcc` was **not** installed so the working `ai-dev` (PyTorch cu128) environment is not put at risk. sn-nvs is not required for MVP detection/tracking. Resume only if/when an isolated toolkit install is explicitly requested.
- sn-calibration / sn-teamspotting public Google Drive weights not auto-downloaded.
- ActiveSpotting ancient CUDA 10.1 GPU wheels intentionally avoided; CPU-compatible torch 1.11 used instead.
- PTS NumPy ABI issue resolved in isolation (`numpy==1.23.5` + `torch==1.11.0+cpu`).

## 12. Restart / start commands

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-dev
python ~/dev-check/check_env.py
python ~/projects/football-analytics/scripts/check_project.py
python ~/projects/football-analytics/scripts/run_setup_smoke.py
```

Log directory: `/home/ahmet/logs/football_setup_20260717_234122`

*Only verified terminal results are marked PASS/READY.*
