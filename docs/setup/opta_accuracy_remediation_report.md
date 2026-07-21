# Accuracy evaluation remediation (GT-honest)

## Status

Ground truth prepared but **not annotated**. Therefore:

- ball precision / recall / F1 = **not computed** (`GT_INCOMPLETE`)
- IDF1 / ID switches = **not computed** (`GT_INCOMPLETE`)
- touch precision = **not computed** (`REVIEW_INCOMPLETE`)
- `overall_publishable = false`

Candidate coverage must not be called recall.

## Artifacts

Ball GT sample (165 frames, seed=42):

`configs/evaluation/short_clip_gt_template/football_ball/`

Identity GT sample (20s window, 100 frames @ stride 5):

`configs/evaluation/short_clip_gt_template/football_identity/`

Football smoke evaluation:

`/home/ahmet/workspace/opta_analytics_smoke/run_20260720_154807_15747b/evaluation/`

Key files:

- `ball_evaluation_report.json`
- `identity_evaluation_report.json`
- `identity_team_1_fragment_classes.csv`
- `calibration_frame_audit.parquet`
- `calibration_evaluation_report.json`
- `touch_review/`
- `publishability_flags.json`

## Annotation

```bash
PYTHONPATH=src python scripts/annotate_ball_gt.py
PYTHONPATH=src python scripts/annotate_player_identity_gt.py --run-dir <run>
```

## Calibration note

Frame `calibration.valid` was already ~90.9%. Propagation → **95.6%** measured+propagated.
Previously reported ~7.6% was `player_metrics.valid` row rate (misleading); renamed to
`player_metrics_row_valid_rate`. Physical metrics still gated on spikes + publishability.

## Identity team_1=37

Classified without hard-cap. See fragment report. Duplicate/unresolved classes flagged;
short fragments not auto-counted as real players.
