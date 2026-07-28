# Match-node-tracker Component Decisions

Upstream: https://github.com/ajwise9/Match-node-tracker  
Branch: `main`  
Commit: `2777aa3f1e9cc563eba07a675cebdf4bfd9306bf`  
Local: `third_party/authorized/match-node-tracker/` (immutable)  
License signals: Ultralytics AGPL-3.0 in checkpoints; repo has no separate LICENSE file.  
Git LFS: not used (`.gitattributes` is LF-normalization only; `.pt` files are real zip checkpoints).

Ground truth: **GT_INCOMPLETE** — no precision/recall/F1/AP/HOTA/IDF1 claims.  
Candidate counts are **not** called recall.

## Summary table

| Component | Upstream method | Existing project method | Decision | Reason | Risk | Target |
|---|---|---|---|---|---|---|
| `train/weights/best.pt` | YOLO26m detect `football/player/referee` @1024 | YOLO11n COCO person + separate football-ball YOLO | ADAPT_AND_USE | Adds native referee class; strong player/ball candidates at 1280/0.25 without claiming accuracy | AGPL weights; no GK class; multi-ball FP risk | `integrations/match_node_tracker/detector_adapter.py` |
| `custom_model.pt` | YOLO26s `football/player` | same | BENCHMARK_ONLY | High player/ball candidates, but no referee; superseded by best.pt for multi-class candidate | Confuse dual-weight ops | keep under third_party; optional weights path |
| `train/weights/last.pt` | YOLO26m same classes | same | REFERENCE_ONLY | Near-duplicate of best; prefer best.pt SHA | Waste VRAM/ops | none |
| `index.py` inference loop | Ultralytics predict + draw pipeline | `DetectionStage` / `PipelineRunner` | REFERENCE_ONLY | Glue only; hard-coded paths / interactive prompt | Path pollution | none |
| `id_tracker.py` | Frame IoU tracker max_age=20 | ByteTrack + BoT-SORT + ReID + global identity | BENCHMARK_ONLY | Simple IoU useful as offline baseline; must not become production | ID fragmentation under occlusion | `tracker_adapter.py` (flag off) |
| `custom_markers.TeamTracker` / `jersey_color` | Frame KMeans + EMA jersey BGR | SigLIP + kit descriptor + track lock | ADAPT_AND_USE | Useful lightweight fallback if adapted to **track-level** voting | Frame flicker if used raw | `team_color_adapter.py` |
| `speed_tracker._cam` | Mask players, LK flow, RANSAC affine | `CameraMotionEstimator` (border features, FB flow, scene-cut) | ADAPT_AND_USE | Similar idea; keep as optional complementary experiment | Weaker scene-cut / border logic than production | `camera_motion_adapter.py` |
| `speed_tracker` speed formula | bbox-height m/px, EMA, **max_kmh=40** | Homography `player_metrics` | REJECT | Hard-cap hides errors; no calibrated meters | Systematic speed bias | none |
| `possession.py` / markers possession | 80px bbox-edge proximity counts | Opta touch + possession timeline | REJECT | Pixel threshold; no field metric / state machine | False possession | none |
| Marker drawing (triangle/ring/ball/bar) | OpenCV overlays | `analytics_renderer` / annotated videos | ADAPT_AND_USE | Renderer-only UX improvement | Overlay clutter | `marker_renderer.py` |
| Goalkeeper class | — | Heuristic `GoalkeeperClassifier` | REJECT (N/A upstream) | **No goalkeeper class in any upstream checkpoint** | Silent GK→player merge | keep existing roles |

## Per-component decisions

### 1. `train/weights/best.pt`

- **Path:** `third_party/authorized/match-node-tracker/train/weights/best.pt`
- **What it does:** YOLO26m detector; names `{0:football,1:player,2:referee}`; train imgsz 1024; SHA256 `2caf0b0cac1a09600c7144edf91568c1848003d8e55bd9e3b3906622d7d2205a`
- **Existing counterpart:** `yolo11n.pt` person + `yolo-sn-ball-opt.pt`
- **Strength:** Football-specific classes including referee; at imgsz=1280 conf=0.25 on `football.mp4` (stride-5): 3407 player / 201 referee / 98 ball candidates; ~55 FPS; ~351 MB VRAM
- **Weakness:** No goalkeeper; multi-ball frames and stand/bench FP heuristics non-zero; AGPL
- **Integration risk:** Medium (schema mapping + dual-detector policy)
- **Decision:** ADAPT_AND_USE (candidate flag)
- **Target:** `detector_adapter.py`
- **Tests:** model SHA, names auto-read, feature flag off by default, no production yaml change

### 2. `custom_model.pt`

- **Path:** `…/custom_model.pt` SHA256 `d97464e7…`
- **What:** YOLO26s `{football,player}` only
- **Existing:** same detectors
- **Strength:** Lower VRAM; high player/ball candidates; stand_fp≈0 at conf≥0.2
- **Weakness:** No referee/GK; overlaps best.pt role
- **Risk:** Low
- **Decision:** BENCHMARK_ONLY
- **Target:** optional weights override only
- **Tests:** class map without referee

### 3. `train/weights/last.pt`

- **Decision:** REFERENCE_ONLY — prefer `best.pt`

### 4. `id_tracker.py`

- **What:** Greedy IoU association
- **Existing:** Ultralytics ByteTrack (production), BoT-SORT (MVP-1), ReID global merge
- **Dense-150 same-det compare:** upstream IoU 151 IDs / mean 1.02s vs proxy ByteTrack-like 839 IDs — **fewer IDs ≠ better**; identity GT incomplete; proxy lacks Kalman
- **Decision:** BENCHMARK_ONLY
- **Target:** `tracker_adapter.py` (default disabled)

### 5. Team color (`jersey_color` + `TeamTracker`)

- **What:** Upper-crop HSV green reject, mean BGR, frame KMeans+EMA
- **Existing:** SigLIP + kit families + temporal lock + role gating
- **Strength:** Cheap color prior
- **Weakness:** Frame-level flicker; no blur/hist/voting; confuses refs/GK
- **Decision:** ADAPT_AND_USE at **track level** only
- **Target:** `team_color_adapter.py`

### 6. Camera motion (`SpeedTracker._cam`)

- **What:** Player-masked features + LK + RANSAC affine
- **Existing:** Stronger estimator with scene-cut histogram reset and border bands
- **Decision:** ADAPT_AND_USE as optional complementary adapter (not replacement)
- **Target:** `camera_motion_adapter.py`

### 7. Speed formula

- **Decision:** REJECT — bbox-height scale + 40 km/h cap forbidden as “fix”

### 8. Possession

- **Decision:** REJECT — fixed pixel proximity inferior to touch/possession stack

### 9. Custom markers

- **Decision:** ADAPT_AND_USE — draw-only

### 10. `index.py` / `requirements.txt` / `media/` / train plots

- **Decision:** REFERENCE_ONLY (docs/plots) or unused glue

## Decision counts

| Decision | Count |
|---|---:|
| DIRECT_USE | 0 |
| ADAPT_AND_USE | 4 |
| BENCHMARK_ONLY | 2 |
| REFERENCE_ONLY | 4 |
| REJECT | 3 |

ADAPT_AND_USE: detector(best.pt), team_color(adapted), camera_motion(adapted), markers  
BENCHMARK_ONLY: custom_model.pt, id_tracker  
REFERENCE_ONLY: last.pt, index.py, train plots/args, requirements  
REJECT: speed formula, possession proximity, goalkeeper-from-upstream (absent)

## Production policy

- Production defaults **unchanged** (`configs/pipeline/opta_analytics.yaml` untouched).
- Candidate config: `configs/integrations/match_node_tracker_candidate.yaml` — all flags **false**.
- Overall status: **PROVISIONAL_CANDIDATE**
