# sn-gamestate → football-analytics integration mapping

Status: **DESIGN-ONLY** (no production wiring). sn-gamestate is the SoccerNet **Game State
Reconstruction** baseline on **TrackLab**. It could not be executed here (no functional env,
no checkpoints, no official dataset — see report), so this maps the **documented output schema**
(extracted read-only from `examples_predictions/SoccerNetGS-test.zip`) against the existing
football-analytics MVP-2 parquet outputs.

## 1. sn-gamestate output schema (per video `tracklab/SNGS-<id>.json`)

```json
{ "predictions": [
  {
    "bbox_pitch": {"x_bottom_left","y_bottom_left","x_bottom_right","y_bottom_right","x_bottom_middle","y_bottom_middle"},
    "bbox_image": {"x","y","w","h"},
    "category_id": 1.0,
    "image_id": "3116000001",
    "video_id": "116",
    "track_id": 1,
    "supercategory": "object",
    "attributes": {"role": "player", "jersey": "33", "team": "left"},
    "id": "0"
  } ]
}
```

- Pitch coords in **meters**; image bbox as **x,y,w,h** (top-left + size).
- `attributes`: role ∈ {player, goalkeeper, referee, ball, ...}, team ∈ {left, right, null}, jersey = string number.

## 2. football-analytics MVP-2 outputs (read-only)

| File | Key columns |
|------|-------------|
| `detections.parquet` | frame_id, bbox_x1..y2 (xyxy), detection_confidence, object_type |
| `tracks.parquet` | frame_id, track_id, bbox_x1..y2, foot_x/y_pixel, tracking_confidence |
| `track_identities.parquet` | track_id, role, team_id, role/team_confidence, valid |
| `game_state.parquet` | track_id, role, team_id, foot_x/y_pixel, **x_field, y_field** (m), calibration_confidence |
| `calibration.parquet` | homography_json (3×3), orientation, pitch_length_m=105, pitch_width_m=68, reprojection_error |

## 3. Field-by-field mapping

| sn-gamestate field | football-analytics source | Adapter |
|--------------------|---------------------------|---------|
| `bbox_image {x,y,w,h}` | tracks `bbox_x1,bbox_y1,(x2-x1),(y2-y1)` | xyxy→xywh (trivial) |
| `bbox_pitch.x/y_bottom_middle` | game_state `x_field,y_field` (foot point, m) | **origin/orientation align** |
| `bbox_pitch.*_bottom_left/right` | not directly produced | derive by projecting bbox foot corners via homography |
| `track_id` | tracks/game_state `track_id` | direct |
| `image_id` / `video_id` | frame_id / match_id | compose |
| `attributes.role` | track_identities/game_state `role` | value map (currently mostly `unknown` in MVP-2) |
| `attributes.team` | `team_id` | map to `left`/`right` (MVP-2 team_id often null) |
| `attributes.jersey` | **none** (no jersey OCR stage in MVP-2) | **gap** — requires a jersey stage (cf. sn-jersey) |
| `category_id`/`supercategory` | object_type | constant/lookup |

## 4. Coordinate-convention notes

- Both use meters on a 105×68 pitch. sn-gamestate `bbox_pitch` sample values (~x 30, y 2) vs
  MVP-2 `x_field/y_field` (~59, 59) imply **different origin/orientation** (corner vs centered,
  and axis direction). A fixed affine (origin shift + axis flip) is needed; verify against
  `calibration.parquet.orientation = left_to_right`.
- MVP-2 calibration is a **3×3 homography** (PnLCalib); sn-gamestate calibration is HRNet
  keypoint-based but ultimately yields pitch coords — comparable at the coordinate level only.

## 5. Gaps / blockers for real integration

1. sn-gamestate cannot run here (no env deps, no checkpoints, no dataset) → cannot generate
   real predictions to compare.
2. MVP-2 lacks a **jersey number** field and has weak team/role (`unknown`) in this run.
3. Convention alignment (pitch origin/orientation) must be validated with real overlapping output.

## 6. Recommendation

The two systems are **schema-compatible with a thin adapter** (xyxy→xywh, pitch origin
alignment, role/team value maps) — but sn-gamestate itself is not runnable in the current
offline/checkpoint-less setup. If GSR-style output is desired, either (a) provision the
sn-gamestate env + weights + dataset in a networked setup, or (b) keep the existing MVP-2
pipeline and add a jersey/team-refinement stage. No production adapter written at this stage.
