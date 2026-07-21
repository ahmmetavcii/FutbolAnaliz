# sn-jersey → football-analytics integration mapping

Status: **DESIGN-ONLY** (no production wiring). sn-jersey is a **README-only** challenge/dataset
spec: it defines the jersey-number recognition task, dataset format, and accuracy metric, but
ships **no baseline code and no checkpoint**. The dataset itself (jersey-2023) is fully present
and validated locally. This document describes how the *dataset/task* would connect to the
existing pipeline and what is needed to actually produce jersey numbers.

## 1. What sn-jersey provides vs. needs

| Component | In sn-jersey? | Notes |
|-----------|---------------|-------|
| Task/metric spec | Yes (README) | Recognition; accuracy over `{player_id: number}`; `-1` = not visible |
| Dataset (jersey-2023) | Yes (local) | train 1427 + test 1211 tracklets, GT JSON per split |
| Loader code | **No** | Independent reference reader validated (`loader_test_results.json`) |
| Model architecture | **No** | Nothing to instantiate; generic backbone smoke only |
| Checkpoint | **No** | No compatible recognition weights anywhere locally |
| Training/inference/eval scripts | **No** | Metric reimplemented for smoke (`evaluation_smoke_results.json`) |

## 2. Existing pipeline outputs to bridge from

- `run_20260718_033654_77a8a7/tracks.parquet` — per-frame player boxes with `track_id`.
- `run_20260718_033654_77a8a7/track_identities.parquet` — track identity / team info.

## 3. How a jersey-number stage would attach

1. **Crop generation** — for each `track_id`, crop the player bbox from each frame of
   `tracks.parquet` → build a *tracklet* (folder of thumbnails), mirroring the jersey-2023
   layout (`images/<track_id>/<track_id>_<frame>.jpg`).
2. **Frame selection / min quality** — jersey-2023 tracklets range widely in length; ~99% have
   ≥20 frames (`dataset_schema.json`). Apply a **min-frame filter (≈20)** and drop tiny/blurred
   crops (observed thumbnail sizes as small as 3–8 px → enforce a min bbox size before recognition).
3. **Temporal aggregation** — the number is visible in only a subset of frames, so a per-frame
   classifier must be aggregated over the tracklet (majority vote / confidence-weighted / a
   temporal model) to a single number per track, exactly as the task defines.
4. **Confidence & unknown handling** — emit `-1` when no frame is confident (matches GT `-1`
   semantics). Track fragmentation means one player may span several `track_id`s → numbers should
   be reconciled after re-ID / across scene cuts (reset on scene cut).
5. **Join key** — output `{track_id: jersey_number, confidence}` and join back onto
   `track_identities.parquet` (combine with team identity for full player labelling).

## 4. Canonical fields (proposed jersey output)

| Field | Source |
|-------|--------|
| `track_id` | from tracks/track_identities |
| `jersey_number` | recognition output (int, `-1` if not visible/low-conf) |
| `confidence` | aggregation score |
| `n_frames_used` | frames passing quality/min filter |
| `source_method` | e.g. `"sn_jersey_baseline"` once a model exists |

## 5. Blockers for real integration

1. **No model + no checkpoint** in sn-jersey → cannot produce real jersey numbers today
   (`inference_results.json` → `BLOCKED_CHECKPOINT_MISSING`). A recognition model must be
   trained on jersey-2023 (data is available) or a compatible checkpoint obtained.
2. sn-jersey provides no loader; a reader like the validated reference one would be needed.

## 6. Recommendation

The **dataset is production-usable now** (validated: valid GT, 0 corrupt in sampled decode,
loadable into batches). To add a jersey-number stage, **train a recognition model on jersey-2023**
(or integrate an external one) and wrap crop-generation + temporal aggregation as above. sn-jersey
itself contributes the data + task/metric definition, not runnable code.
