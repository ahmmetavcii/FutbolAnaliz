# ReID resolution report

**Date:** 2026-07-21  
**Smoke run:** `/home/ahmet/workspace/opta_analytics_smoke/run_20260720_154807_15747b`

## Verdict

`reid_status = SOLVED` on the football smoke clip after the ReID stack rewrite.

| Metric | Before | After |
|---|---:|---:|
| Prototype tracks (embedding coverage) | 23 / 224 (10%) | 167 / 224 (74.5%) |
| Merged fragments | 5 | 44 |
| Validated team_1 | 37 | 11 |
| Validated team_0 | 11 | 7 |
| `stats_publishable` (identity) | false | true |
| `reid_solved` | false | true |

## What was wrong

1. ReID only ran on `usable_for_metrics` tracks → almost no embeddings for stitchable fragments.
2. Absolute cosine thresholds (0.42 / 0.55) were unsafe: Market1501 OSNet gives ~0.83 mean similarity even for simultaneous same-team hard negatives.
3. Fragment classifier treated successful merges as `duplicate_identity`.
4. No on-field / roster capacity handling for false team tags beyond 11 co-visible people.

## What we shipped

- `analytics/reid_matching.py` — torso crop, quality ranking, robust median prototypes, hard-negative calibration, relative gallery margin
- `stages/reid.py` — broad coverage mode, torso crops, consistency sidecar
- `opta/identity_resolve.py` — calibrated thresholds, unique/dominant position slots, second-pass gallery merge, on-field + roster surplus demotion (no false merges)
- `evaluation/reid_metrics.py` + `scripts/recompute_reid_identity.py`
- Config updates in `configs/pipeline/opta_analytics.yaml` and `mvp2_spatial_analytics.yaml`

## Honesty notes

- Market1501 OSNet is still not football-specific; discrimination is weak. Safety comes from hard-negative calibration + position/time gates, not from trusting raw cosine.
- Identities demoted as `on_field_surplus` / `roster_surplus` stay in the map as unresolved — they are not force-merged into another player.
- Labeled IDF1 still requires completed identity GT (`configs/evaluation/short_clip_gt_template/`). Until GT is annotated, numeric IDF1 remains `GT_INCOMPLETE`.
- Multi-camera ReID real-video claim remains open.

## How to recompute

```bash
conda activate ai-dev
cd /home/ahmet/projects/football-analytics
PYTHONPATH=src python scripts/recompute_reid_identity.py \
  --run-dir /path/to/run \
  --config configs/pipeline/opta_analytics.yaml
```
