# Remaining External / Runtime Blockers

This file lists items that remain blocked or unverified for reasons outside the
clean-room football-analytics capability path. They do **not** automatically
downgrade a component's technical PASS when a real local capability exists.

## BLOCKED_EXTERNAL_ACCESS / NOT_RUN

| Component | Blocker | Classification | Impact |
|---|---|---|---|
| sn-trackeval | Official SoccerNet Game State challenge ground truth not present locally | `NOT_RUN_NO_OFFICIAL_GT` | Official leaderboard comparison not executed; synthetic + adapter evaluation PASS |
| sn-gamestate | Original TrackLab pipeline OOM (exit 137) after ~8 GB VRAM use on short clip | `BLOCKED_RUNTIME_OOM` | Original baseline not PASS; compatible MVP-2 export PASS |
| sn-gamestate | Challenge/private NDA assets intentionally unused | Policy | No attempt to bypass access controls |
| sn-echoes | Upstream LICENSE file not verified for redistribution | `LICENSE_REVIEW_REQUIRED` | Technical dataset/devkit PASS retained; redistribution gated |
| sn-calibration | Upstream LICENSE for repo/checkpoint not explicit | `LICENSE_REVIEW_REQUIRED` | Original + compatible capability PASS retained |
| sn-jersey | Upstream repo is README-only; license review still required for dataset redistribution | `LICENSE_REVIEW_REQUIRED` | Clean-room model PASS; SoccerNet data terms still apply |

## Non-blockers (explicitly out of scope)

- sn-echoes does not generate ASR for new videos; this is expected for a dataset/devkit.
- sn-jersey original code does not exist upstream; replacement is the intended path.
- CUDA Toolkit / nvcc were not installed by design.

## Recommended next actions (optional)

1. Acquire official SoccerNet tracking/Game State GT under approved terms, then run TrackEval leaderboard-style evaluation.
2. Retry sn-gamestate original baseline with reduced module set / smaller batch / lower-resolution clip if original PASS is required.
3. Complete legal review for LICENSE_REVIEW_REQUIRED repos before any redistribution.
