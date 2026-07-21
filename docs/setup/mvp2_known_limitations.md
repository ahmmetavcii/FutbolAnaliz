# MVP-2 Known Limitations

## Current conservative limitations

- The verified detector is COCO `yolo11n.pt`, not a football-specific detector.
  It detects people and sports balls but has no referee or goalkeeper class.
  Role identity therefore remains `unknown` unless an upstream role observation
  is available; no player-ID exception is used.
- Team identity is unsupervised temporal shirt-colour clustering. Similar kits,
  goalkeeper kits, referee clothing, compression, shadows, and very short
  tracks can remain unknown. Unknown identities are excluded from team metrics.
- The shot classifier is an explainable heuristic baseline, not a trained
  broadcast-shot model. Replay detection requires explicit replay evidence in a
  future model; visual heuristics can conservatively return `unknown`.
- PnLCalib is executed out-of-process via
  `scripts/pnlcalib_worker.py` in the isolated `sn-banner-mmseg` env
  (shapely + torch/CUDA). `ai-dev` pins are untouched. Homography fitting and
  every validity gate still run in-process through `calibration_from_mapping`.
  Sampled frames that fail coverage/confidence gates stay invalid; hold is
  bounded by `hold_max_frames`. `sn_calibration` remains **BLOCKED** (baseline
  weights not retrievable; no placeholder). A reviewed
  `DEMO_MANUAL_FALLBACK` JSON for `football.mp4` frame 100 is available at
  `configs/calibration/manual_football_frame100.json` and is only used if
  PnLCalib fails. The pipeline never invents the reference repository's four
  fixed pixels or 23.32 m.
- COCO ball recall is low on broadcast footage. Ball prediction is bounded and
  becomes null after the configured gap; it is not backfilled.
- Possession is a conservative state-machine estimate, not ground truth. It uses
  field metres only when calibration is valid and otherwise bbox-height
  normalized pixel distance. Unknown/loose/contested states are intentional.
- Camera motion estimates a global partial affine transform. Strong parallax,
  zoom plus perspective changes, motion blur, and cuts can invalidate frames.
- Chunk size and stage resume are implemented, and video processing is
  streaming. Canonical tabular artifacts are currently read per stage; a
  full-match dataset can fit typical RAM, but partitioned Parquet/chunk-level
  restart inside a stage remains future work. Stage-level restart does not
  rerun completed stages.
- Re-identification does not stitch fragmented track IDs. Metrics remain
  separate unless a future canonical identity stage provides stitch confidence.
- The current real-video fixture is 6.12 seconds, shorter than the requested
  20–60 second secondary test. No longer suitable main-camera clip was assumed
  or fabricated.

## Deferred improvements

1. Add a licensed football-specific detector with player/referee/goalkeeper/ball
   classes and a model card/checksum.
2. Add a trained shot/replay classifier and segment-level evaluation.
3. Improve PnLCalib coverage (lower-stride sampling, temporal smoothing of
   homographies) and evaluate against labelled pitch landmarks.
4. Evaluate team/role identity on manually labelled tracklets.
5. Add partitioned Parquet and per-chunk completion manifests for full matches.
6. Add HOTA/IDF1, calibration and ball/possession benchmark datasets.
