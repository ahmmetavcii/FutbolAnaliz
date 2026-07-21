# sn-calibration — Repo Verification Report

**Overall status: BLOCKED** (missing model checkpoint → real neural calibration inference cannot run; all code/geometry tests pass)

- **Repo:** `/home/ahmet/projects/soccernet/sn-calibration`
- **Remote:** `https://github.com/SoccerNet/sn-calibration.git`
- **Branch/commit:** detached HEAD @ `ab38f461bec729fead86b6986839de1bb826f16d`
- **Environment used:** dedicated conda env `sn-calibration` (Python 3.10.20, torch 2.1.2+cu121, torchvision 0.16.2, numpy 1.26.4, opencv 4.8.1, CUDA available). `ai-dev` verified but **not** used/modified.
- **Tested:** 2026-07-18

---

## 1. Repo integrity

| Item | Value |
|------|-------|
| Folder exists | YES |
| Remote | https://github.com/SoccerNet/sn-calibration.git |
| Branch | detached HEAD |
| Commit SHA | `ab38f461bec729fead86b6986839de1bb826f16d` |
| Dirty | No (clean working tree) |
| `git diff --stat` | empty (saved as `repo_initial_diff.patch`, 0 bytes) |
| `git fsck` | clean (no errors) |
| Submodules | none (no `.gitmodules`) |
| Disk size | 5.7M (2.9M excluding `.git`) |
| Tracked files | 27 |
| License | **LICENSE_NOT_VERIFIED** — no LICENSE/COPYING file; README/ChallengeRules contain no license text |

## 2. Purpose & architecture

SoccerNet **Camera Calibration** challenge baseline. Two-stage pipeline:

1. **`src/detect_extremities.py`** — `SegmentationNetwork` = DeepLabV3-ResNet50 (29 classes) segments pitch lines, post-processed into per-class **line extremities** (normalized `{x,y}`). Loads `resources/soccer_pitch_segmentation.pth` (**MISSING**) + `mean.npy`/`std.npy`. Input resized to 640×360. Output: `extremities_<frame>.json`.
2. **`src/baseline_cameras.py`** — reads extremities → estimates plane homography from line correspondences → `Camera.from_homography` → `refine_camera` (`solvePnPRefineLM`) → camera params JSON. Pure numpy/opencv, **no checkpoint**. Assumes 960×540.

- **Evaluation:** `src/evaluate_extremities.py`, `src/evaluate_camera.py`, `src/evalai_camera.py` (need dataset GT).
- **Input:** SoccerNet-V3 calibration **dataset folder** (`per_match_info.json` + images), **not** video.
- **Output contract** (`Camera.to_json_parameters`): `pan/tilt/roll_degrees`, `position_meters`, `x/y_focal_length`, `principal_point`, `radial/tangential/thin_prism_distortion` — a **full camera model**, not a raw homography. Pitch coords are **centered** (105×68).
- **Hardcoded assumptions:** default `-s /home/fmg/data/...`, `-p /home/fmg/results/...`; checkpoint path `../resources/...` (run from `src/`); 640×360 / 960×540 resolutions; draw functions assume 540p.
- **GPU:** uses CUDA if available, else CPU (`nn.DataParallel`).

## 3. Environment

- **ai-dev** (verified, not used): torch 2.11.0+cu128, `pip check` clean.
- **sn-calibration** (dedicated, used): Python 3.10.20, torch 2.1.2+cu121, torchvision 0.16.2, numpy 1.26.4, opencv 4.8.1, PIL 9.5.0, tqdm 4.62.3, matplotlib 3.7.5 — close to repo requirements (`torch~=1.10` original; env is a newer compatible stack). No packages changed.

## 4. Code & import tests (sn-calibration env)

- `python -m compileall -q src` → **exit 0**.
- Imports OK: `camera`, `soccerpitch`, `baseline_cameras`, `detect_extremities` (no checkpoint load at import time — no import-time side effects).
- `python src/detect_extremities.py --help` → **exit 0**.
- No unit tests present in repo.

## 5. Checkpoint & asset inventory

- **Required:** `resources/soccer_pitch_segmentation.pth` — **MISSING**: declared in `.gitattributes` as Git-LFS, absent on disk, not in HEAD, not found anywhere under `/home/ahmet` or `/mnt/c/football_data`.
- **Repo assets present:** `resources/mean.npy`, `resources/std.npy` (152 B each).
- **Other model files found (NOT usable here):** `sn-banner/SV_lines.pth`, `sn-banner/SV_kp.pth` (different architecture), `yolo11n.pt` (detector). None match the DeepLabV3-ResNet50 29-class `checkpoint['model']` contract.
- Per rules: **no download, no dummy/random weights.**

## 6. Real frames

Extracted frames 100/200/300/400/500 from `football.mp4` (1920×1080, 25 fps, 750 frames) via OpenCV → all written, non-zero, reopenable, valid dimensions (`test_frames_manifest.json`).

## 7. Inference & geometry validation

- **Neural extremity detection: BLOCKED.** Instantiating `SegmentationNetwork` fails with `FileNotFoundError` on the missing checkpoint. Not substituted with random weights.
  - ⚠️ Side effect: constructing `deeplabv3_resnet50` auto-downloaded the torchvision **ResNet50 ImageNet backbone** (torchvision 0.16 default `weights_backbone`) — **unintended, NOT the calibration checkpoint, unused for any inference.**
- **Calibration geometry module: PASS (5/5).** Deterministic synthetic camera round-trip through the repo's own `src/camera.py` (`from_homography` + `project_point`), in the repo's centered pitch coordinate system:
  - `from_homography` succeeds for all 5 poses; homography **invertible**, determinant far from 0.
  - **Intrinsics recovered exactly** (focal relative error ~1e-14).
  - Outputs **finite** (no NaN/Inf); `to_json_parameters` serializable.
  - Pitch confirmed **105×68 m** (39 model points).
  - Rough reprojection 7.7–10.1 px (this is the *documented rough initializer* stage before LM refinement; not sub-pixel). This validates the calibration **math only**, not the neural pipeline.

## 8. Relationship to PnLCalib

- `provider_attempts.json` of run `run_20260718_033654_77a8a7` already records **`sn_calibration` → BLOCKED** ("baseline weights not retrievable (Google Drive) and sn-calibration env lacks required dependencies"), with **PnLCalib** as the working provider. This test independently reproduces that conclusion.
- PnLCalib `calibration.parquet`: 750 rows, **682 valid / 68 invalid**, mean reprojection 1.80, mean confidence 0.705, mean coverage 0.41, output = `homography_json`.
- **No structural/absolute comparison** performed: different coordinate conventions (centered vs corner-origin), different output types (camera model vs homography), different error units, and **no ground truth** → no accuracy claim about which is better.

## 9. Canonical pipeline compatibility

See `sn_calibration_adapter_mapping.md`. A future adapter would need: single-frame wrapper around `SegmentationNetwork.analyse_image`, camera-model→homography conversion (`H = K·[r1 r2 R(-C)]`), coordinate re-origin (centered→corner), and resolution scaling (960×540→1920×1080). **Blocked today** by the missing checkpoint. Recommendation: keep PnLCalib as the provider.

## 10. Artifacts (all present, non-empty, parseable)

`/home/ahmet/projects/football-analytics/artifacts/soccernet_repo_tests/sn-calibration/`
- `repo_inventory.json`, `checkpoint_inventory.json`, `test_frames_manifest.json`, `inference_results.json`, `geometry_validation.json`, `environment_summary.json`, `repo_initial_diff.patch` (empty = clean repo).

## 11. Result

| Metric | Value |
|--------|-------|
| Tests passed | 10 |
| Tests failed | 0 |
| Neural inference | **BLOCKED** (missing checkpoint) |
| Geometry module | PASS (5/5 synthetic cases) |
| Valid calibration frames (real) | 0 (neural pipeline could not run) |
| Overall | **BLOCKED** |

**Open blockers:**
1. `resources/soccer_pitch_segmentation.pth` missing (Git-LFS, not downloadable per rules) → real inference impossible.
2. Inference entrypoints accept only a SoccerNet dataset folder, not frames/video (wrapper needed even with checkpoint).
3. LICENSE_NOT_VERIFIED.

**Log:** `/home/ahmet/projects/football-analytics/logs/soccernet_repo_tests/04_sn_calibration.log`
