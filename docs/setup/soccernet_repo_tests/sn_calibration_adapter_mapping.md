# sn-calibration → football-analytics canonical calibration mapping

Status: **DESIGN-ONLY** (no production wiring). The neural stage of sn-calibration is
currently **BLOCKED** (missing `resources/soccer_pitch_segmentation.pth`), so this document
only describes how its output *would* map if the checkpoint were available.

## 1. What sn-calibration produces

Two-stage pipeline:

1. `src/detect_extremities.py` — DeepLabV3-ResNet50 (29 classes) segments pitch-line
   pixels, post-processed into per-class **line extremities** (normalized `{x,y}` in `[0,1]`).
   Output: `extremities_<frame>.json`. **Requires the missing checkpoint.**
2. `src/baseline_cameras.py` — reads extremities, estimates a plane homography from line
   correspondences, initializes a `Camera` (`from_homography`), refines it
   (`solvePnPRefineLM`), and writes camera parameters via `Camera.to_json_parameters()`.
   Output: `camera_<frame>.json`.

`Camera.to_json_parameters()` fields:

```
pan_degrees, tilt_degrees, roll_degrees,
position_meters [X,Y,Z],
x_focal_length, y_focal_length,
principal_point [cx,cy],
radial_distortion [6], tangential_distortion [2], thin_prism_distortion [4]
```

Note: sn-calibration outputs a **full camera model** (intrinsics + extrinsics + distortion),
**not** a raw 3×3 homography. The pitch coordinate system is **centered** (origin at
center mark; X∈[-52.5,52.5], Y∈[-34,34], Z up), image default 960×540.

## 2. Canonical calibration schema (football-analytics `calibration.parquet`)

Observed columns (from `run_20260718_033654_77a8a7/calibration.parquet`, schema_version 2.0.0):

```
schema_version, run_id, match_id, frame_id, timestamp_ms,
source_method, confidence, valid, segment_id, provider,
homography_json (3x3), orientation, pitch_length_m, pitch_width_m,
reprojection_error, visible_pitch_coverage, invalid_reason
```

The canonical contract is **homography-centric** (`homography_json`), corner-origin pitch
coords (`[0..105]×[0..68]`), as produced by the PnLCalib provider.

## 3. Field-by-field mapping

| Canonical field          | From sn-calibration                                                                 | Adapter needed |
|--------------------------|-------------------------------------------------------------------------------------|----------------|
| `frame_id`               | frame index of processed image                                                      | trivial |
| `timestamp_ms`           | `frame_id / fps * 1000` (sn-calibration has no time; video-derived)                 | compute |
| `homography_json`        | **derive** H = `K · [r1 r2 (R·(-C))]` from camera params (Z=0 plane)                | **yes** (build H from cam model) |
| `orientation`            | infer from pan sign / pitch layout                                                   | heuristic |
| `pitch_length_m`         | 105.0 (constant in `SoccerPitch`)                                                    | constant |
| `pitch_width_m`          | 68.0 (constant in `SoccerPitch`)                                                     | constant |
| `reprojection_error`     | from `evaluate_extremities` / recompute vs detected extremities                     | compute (px) |
| `visible_pitch_coverage` | fraction of pitch points projected inside image bounds                              | compute |
| `confidence`             | no native score; derive from #line matches / inlier ratio / reproj                 | **yes** (define proxy) |
| `valid`                  | `from_homography` success AND enough line matches (≥4) AND reproj below threshold    | rule |
| `invalid_reason`         | e.g. "insufficient line correspondences", "homography failed"                       | rule |
| `source_method`/`provider` | constant `"sn_calibration"`                                                        | constant |

## 4. Coordinate-convention differences (must handle in adapter)

- **Origin**: sn-calibration = centered; canonical/PnLCalib = corner-origin. A fixed
  translation (`+52.5`, `+34`) + possible axis flip is required to align homographies.
- **Output type**: sn-calibration = camera model → must be converted to a plane homography
  to fit `homography_json`. Conversion: `H = K · [R[:,0], R[:,1], R·(-C)]`, then normalize
  by `H[2,2]`. (Validated to be invertible / finite in `geometry_validation.json`.)
- **Resolution**: sn-calibration assumes 960×540 (calibration) / 640×360 (segmentation);
  football.mp4 is 1920×1080 → scale intrinsics/homography accordingly (`Camera.scale_resolution`).
- **Error units**: canonical `reprojection_error` here is ~meters (PnLCalib); sn-calibration
  challenge uses normalized/pixel reprojection with an Accuracy@threshold metric. Not directly comparable.

## 5. Blockers for real integration

1. **Missing segmentation checkpoint** `resources/soccer_pitch_segmentation.pth` (Git-LFS,
   not present, not downloadable per rules). Without it, no extremities → no calibration.
2. Inference entrypoints only accept a **SoccerNet dataset folder** (`per_match_info.json` +
   images), not single frames/video. A wrapper calling `SegmentationNetwork.analyse_image`
   directly would be needed for per-frame use (feasible only once the checkpoint exists).
3. The existing pipeline already records `sn_calibration` as **BLOCKED** and uses PnLCalib.

## 6. Recommendation

Keep **PnLCalib** as the calibration provider. Revisit sn-calibration only if the official
`soccer_pitch_segmentation.pth` becomes available; at that point a thin adapter
(single-frame wrapper + camera-model→homography conversion + coordinate re-origin) is
straightforward and its geometry math (`camera.py`) is already validated.
