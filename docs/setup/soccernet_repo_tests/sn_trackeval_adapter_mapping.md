# sn-trackeval ↔ football-analytics `tracks.parquet` — Format Mapping

Status: **format-mapping only (no production adapter, no evaluation performed)**

This document describes, at schema level, how our `tracks.parquet` output would map
to the MOTChallenge 2D box format consumed by `sn-trackeval`. It does **not**
implement a production adapter and does **not** produce any accuracy metric.

## Source: football-analytics `tracks.parquet`

Inspected file: `/home/ahmet/workspace/runs/run_20260718_033654_77a8a7/tracks.parquet`
(10,601 rows).

| column | dtype | notes |
|---|---|---|
| `frame_id` | int64 | **0-based** frame index |
| `timestamp_ms` | float64 | |
| `track_id` | int64 | tracker-assigned identity |
| `detection_id` | object | e.g. `f0_t1_d0` |
| `object_type` | object | e.g. `person` |
| `class_id` | int64 | `0` = person in this run |
| `bbox_x1`,`bbox_y1`,`bbox_x2`,`bbox_y2` | float64 | **corner** format (top-left, bottom-right) in pixels |
| `foot_x_pixel`,`foot_y_pixel` | float64 | |
| `tracking_confidence` | float64 | |
| `source_tracker` | object | e.g. `bytetrack` |
| `source_model` | object | e.g. `yolo11n.pt` |
| `schema_version` | object | |

## Target: sn-trackeval MOTChallenge 2D box (`{seq}.txt`)

CSV, one detection per line, 10 fields, **1-based** frames/ids:

```
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x=-1>,<y=-1>,<z=-1>
```

## Field mapping (tracker predictions)

| MOT field | derived from `tracks.parquet` |
|---|---|
| `frame` | `frame_id + 1` (0-based → 1-based) |
| `id` | `track_id` |
| `bb_left` | `bbox_x1` |
| `bb_top` | `bbox_y1` |
| `bb_width` | `bbox_x2 - bbox_x1` |
| `bb_height` | `bbox_y2 - bbox_y1` |
| `conf` | `tracking_confidence` |
| `x`,`y`,`z` | `-1` (ignored for 2D) |

Row filtering: keep `object_type == 'person'` (the ball / other classes are handled
separately by the SoccerNet GS dataset config, e.g. `IGNORE_BALL`).

## Conversion notes / gaps

- **Coordinate format:** corner (`x1,y1,x2,y2`) → `left,top,width,height`. Straightforward arithmetic.
- **Frame indexing:** our data is 0-based; MOT is 1-based. Off-by-one must be applied.
- **Sequence structure:** MOT expects one `{seq}.txt` per sequence plus a `seqinfo.ini`
  (or a `SEQ_INFO` dict) giving `seqLength`. Our parquet is a single run; the sequence
  name + length would have to be supplied.
- **SoccerNet Game-State (GS):** the official `run_soccernet_gs.py` path expects a richer
  ground-truth JSON (roles, teams, jersey numbers, pitch/image space) — considerably more
  than our current schema carries. Plain MOT (`MotChallenge2DBox`) is the realistic target
  for our current `tracks.parquet`.

## CRITICAL: evaluation is not possible without ground truth

`tracks.parquet` contains **only tracker predictions** — there is **no ground-truth
annotation** in it or alongside it. `sn-trackeval` (HOTA/CLEAR/Identity) computes metrics
by comparing predictions against ground truth. Therefore:

- No accuracy/tracking metric can be produced from `tracks.parquet` alone.
- Running the evaluator on this file against itself (or with no GT) would be meaningless
  and must not be reported as a success metric.
- A real benchmark requires the SoccerNet tracking / game-state **ground-truth dataset**
  (NDA / official), which is **not present** on this machine → that benchmark is **BLOCKED**.

## Not done here (by design)

- No production converter script was written.
- No metric was computed on `tracks.parquet`.
