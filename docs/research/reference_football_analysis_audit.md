# Reference Football Analysis Audit

## Scope and provenance

- Repository: https://github.com/abdullahtarek/football_analysis
- Local read-only reference: `/home/ahmet/projects/third-party/reference-football-analysis`
- Commit: `e4799632cdf271cf57f4eb0c4d872cf1b7ab0f17`
- Branch at clone: `main`
- Working tree: clean
- Size at audit: 13,491,870 bytes
- `git fsck --full`: PASS
- Training video: https://www.youtube.com/watch?v=neBZ6huolkg
- License status: **LICENSE_NOT_VERIFIED** (treat as all-rights-reserved)

No `LICENSE`, `COPYING`, package metadata license field, or README license grant
was found. Therefore no source Python or notebook code is copied into this
project. The repository and video are used only as algorithmic references; all
accepted ideas are independently reimplemented against canonical contracts.

## Verified global issues

| Finding | Evidence | Consequence |
|---|---|---|
| Fixed input and model paths | `main.py:14,17`; `yolo_inference.py:3,5` | Not reusable or testable |
| Pickle stubs | `main.py:20-21,28-29`; `trackers/tracker.py:48-53,100-102`; camera estimator | Unsafe/unversioned cache contract |
| Entire video in RAM | `utils/video_utils.py:3-11`; main passes frame lists through all modules | Does not scale to full matches |
| Detection batch/confidence constants | `trackers/tracker.py:41-45` (`batch_size=20`, `conf=0.1`) | High RAM and excessive false positives |
| Unlimited ball fill | `trackers/tracker.py:28-36` (`interpolate`, then `bfill`) | Invents ball locations across long gaps and at video start |
| Ball ID forced to 1 | `trackers/tracker.py:95-98` | No association or confidence contract |
| Fixed possession distance | `player_ball_assigner.py:7` (70 px) | Resolution/perspective dependent |
| Indefinite previous possession | `main.py:67-69` | No unknown/timeout state |
| Track-specific exception | `team_assigner.py:68-69` (`player_id == 91`) | Video-specific and invalid generally |
| First-frame team model | `main.py:45-47` | Fails after cuts, lighting changes, or absent teams |
| Goalkeeper merged into player | `trackers/tracker.py:70-73` | Destroys role identity |
| Fixed FPS/window | `speed_and_distance_estimator.py:8-9` (24 FPS, 5 frames) | Wrong timing on 25/30/VFR video |
| Fixed homography | `view_transformer.py:5-18` (four pixels and 23.32 m) | Valid only for one view |
| Resolution-specific UI | `trackers/tracker.py:170-181` | Assumes approximately 1920×1080 |
| Fixed optical-flow mask | `camera_movement_estimator.py:21-22` | Uses columns `0:20` and `900:1050` regardless of resolution |
| Weak motion model | `camera_movement_estimator.py:57-73` | Selects the maximum-moving single feature; no RANSAC/inlier test |
| No cut/replay handling | all processing modules | State leaks across shots and replays |
| Missing uncertainty | track/team/ball/possession/speed dictionaries | No canonical confidence, validity, reason, or unknown values |

## Component decisions

### `main.py`

- Source function: sequential demo orchestration.
- Algorithm: read frames → YOLO/ByteTrack → optical flow → fixed perspective
  transform → KMeans teams → nearest-player possession → overlays.
- Strength: clear educational decomposition and useful end-to-end ordering.
- Hard-coded/video-specific behavior: paths, pickle files, first-frame team
  fitting, previous-team carry-forward.
- Generalization/performance/contract issues: all frames retained; untyped nested
  dictionaries; no manifests, resume, schemas, validity gates, or stage errors.
- Pipeline counterpart: config-driven resumable `PipelineRunner` and independent
  canonical stages.
- Decision: **REIMPLEMENT**.

### `trackers/tracker.py`

- Source function: YOLO inference, ByteTrack, ball extraction, interpolation,
  annotation.
- Algorithm: batched YOLO at confidence 0.1; supervision ByteTrack; pandas
  interpolation; ellipse/triangle rendering.
- Strength: foot point for players and center for ball are correct initial
  geometric choices; tracker IDs and simple visual markers are useful.
- Hard-coded behavior: batch 20, confidence 0.1, ball ID 1, fixed overlay pixels.
- Generalization/performance/contract issues: frame-list input, goalkeeper role
  loss, no confidence in outputs, unlimited fill and pickle cache.
- Pipeline counterpart: existing streaming Ultralytics adapter plus canonical
  detections/tracks; new bounded ball trajectory and analytics renderer.
- Decision: foot point and marker ideas **ADAPT**; implementation **REIMPLEMENT**;
  unlimited interpolation and fixed-ID logic **REJECT**.

### `team_assigner/team_assigner.py`

- Source function: shirt-color team clustering.
- Algorithm: per-crop two-cluster RGB KMeans, corner voting for background, then
  two-team KMeans.
- Strength: unsupervised color is practical when no team classifier exists.
- Hard-coded behavior: first-frame fitting and player 91 override.
- Generalization issues: RGB, full upper-half contamination, no pitch/dark/light
  masks, temporal samples, outlier rejection, referee/goalkeeper separation,
  confidence, cut reset, or similar-kit handling.
- Pipeline counterpart: track-level upper-torso LAB features, quality masks,
  temporal robust pooling, explicit role/unknown output and confidence.
- Decision: KMeans concept **ADAPT**; code **REIMPLEMENT**; ID exception **REJECT**.

### `player_ball_assigner/player_ball_assigner.py`

- Source function: assign ball to nearest player foot.
- Algorithm: minimum distance from ball center to either bottom bbox corner,
  thresholded at 70 pixels.
- Strength: proximity to player feet is a useful possession observation.
- Hard-coded behavior: 70 pixels.
- Generalization issues: no perspective normalization, field distance, temporal
  state, contest, pass, timeout, ball confidence, or unknown.
- Pipeline counterpart: state machine using field metres when valid, otherwise
  bbox-height-normalized distance, debounce and timeout.
- Decision: proximity signal **ADAPT**; implementation **REIMPLEMENT**.

### `camera_movement_estimator/camera_movement_estimator.py`

- Source function: estimate and draw camera translation.
- Algorithm: Shi–Tomasi + LK flow.
- Strength: sparse optical flow is lightweight and streamable in principle.
- Hard-coded behavior: pixel-column mask and 5-pixel threshold.
- Generalization issues: takes every frame in memory, picks the single largest
  feature displacement, no forward/backward validation, RANSAC, rotation/scale,
  confidence, player exclusion, or scene reset.
- Pipeline counterpart: normalized border regions, optional dynamic bbox mask,
  forward/backward consistency, robust affine RANSAC and inlier metrics.
- Decision: optical-flow concept **ADAPT**; code **REIMPLEMENT**.

### `view_transformer/view_transformer.py`

- Source function: pixel-to-pitch perspective transform.
- Algorithm: four-point OpenCV homography.
- Strength: correct mathematical primitive after correspondences are validated.
- Hard-coded behavior: four image points, width 68 m, partial length 23.32 m.
- Generalization issues: one camera/view only; no reprojection error, orientation,
  visible coverage, per-segment validity, or calibration provenance.
- Pipeline counterpart: adapter priority sn-calibration → PnLCalib → verified
  manual JSON → explicitly demo-only fallback. Current run emits invalid/null
  calibration if none is available.
- Decision: homography primitive **ADAPT**; fixed points/default fallback **REJECT**.

### `speed_and_distance_estimator/speed_and_distance_estimator.py`

- Source function: calculate and draw speed/distance.
- Algorithm: displacement over a five-frame window at 24 FPS.
- Strength: windowed displacement reduces some frame noise.
- Hard-coded behavior: 5-frame window and 24 FPS.
- Generalization issues: no timestamp use, smoothing model, acceleration/speed
  plausibility gates, coverage, calibration quality, shot validity,
  fragmentation, confidence, or null output.
- Pipeline counterpart: timestamp-driven smoothed field displacement gated by
  calibration, shot type, and track quality.
- Decision: **REIMPLEMENT**.

### `utils/bbox_utils.py`

- Source function: bbox center, width, Euclidean distance, foot point.
- Strength: player foot point `(x1+x2)/2, y2` matches this project's canonical
  convention; ball center is appropriate.
- Missing behavior: clipping, validity, area, visibility/truncation, crop quality,
  and foot-point confidence.
- Pipeline counterpart: validated `BBox` geometry utilities.
- Decision: formulas **ADAPT**; module **REIMPLEMENT**.

### `utils/video_utils.py`

- Source function: read and save video.
- Strength: simple OpenCV example.
- Hard-coded behavior: XVID and 24 FPS.
- Performance issue: reads and returns every frame, then writes a frame list.
- Pipeline counterpart: streaming capture/writer with metadata FPS and bounded
  chunk state.
- Decision: **REJECT** and **REIMPLEMENT** streaming I/O.

### `yolo_inference.py`

- Source function: one-off YOLO demo.
- Issues: fixed model/input paths and non-canonical native result.
- Pipeline counterpart: existing Ultralytics adapter and canonical Parquet.
- Decision: **REJECT** (already superseded).

### Training notebook and model notes

- `training/football_training_yolo_v5.ipynb`: educational custom YOLO workflow;
  not a reproducible package contract and references external data/runtime.
- `development_and_analysis/color_assignement.ipynb`: exploratory color
  clustering; useful only as algorithmic background.
- README model: external Google Drive `best.pt`; no checksum, model card,
  dataset/license record, or verified local compatibility.
- Pipeline counterpart: model registry with hashes and current verified
  `yolo11n.pt`; custom football model is deferred until licensed data/model
  provenance is verified.
- Decision: notebooks **DEFER**; unverified model **DEFER**.

## Integration boundary

Accepted ideas are limited to: player foot-point geometry, ball center, temporal
shirt-color evidence, sparse optical flow, homography as a calibrated primitive,
nearest-player proximity as one possession signal, and resolution-independent
ellipse/triangle visualization. Every accepted idea is reimplemented with
streaming, confidence, unknown/null states, config thresholds, canonical Parquet,
stage manifests, and conservative validity gates.
