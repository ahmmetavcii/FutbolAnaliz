# Third-Party Notices

Exact remotes and commits are recorded in `external_repos.lock.yaml` and
`third_party/manifest.json`. License texts that could be verified are copied
into `third_party/licenses/`.

## Distribution policy

- MIT-licensed components may be used with attribution.
- GPL-licensed components remain in isolated environments or external processes.
  No GPL source is copied into `src/football_analytics`.
- Where an upstream LICENSE file is missing, the technical capability may still
  be used locally, but redistribution requires license review.

## SoccerNet components

| Component | License | Integration |
|---|---|---|
| SoccerNet SDK | MIT | External package |
| sn-trackeval | MIT | External evaluator + clean-room adapter |
| sn-echoes | LICENSE_REVIEW_REQUIRED | Clean-room data reader only |
| sn-calibration | LICENSE_REVIEW_REQUIRED | External original model + clean-room planar adapter |
| sn-jersey | LICENSE_REVIEW_REQUIRED | Clean-room implementation; upstream is README-only |
| sn-reid | MIT | External OSNet FeatureExtractor → global identity |
| sn-gamestate | GPL-3.0 | Isolated process + clean-room JSON exporter |

## Third-party dependencies

| Component | License | Integration |
|---|---|---|
| TrackLab | MIT | Isolated `sn-gamestate-env` |
| PnLCalib | GPL-2.0 | External provider; canonical parquet only |
| No-Bells-Just-Whistles | GPL-2.0 | External sn-gamestate calibration module |
| Ultralytics YOLO | AGPL-3.0 / commercial | External detection weights for Game State baseline |
| Match-node-tracker (ajwise9) | Owner-authorized local use; Ultralytics AGPL weights | Immutable under `third_party/authorized/match-node-tracker` @ `2777aa3f1e9cc563eba07a675cebdf4bfd9306bf`; feature-flagged adapters only — not production default |

## Data terms

SoccerNet datasets remain subject to SoccerNet data terms. Broadcast and NDA
splits are not redistributed by this project.
