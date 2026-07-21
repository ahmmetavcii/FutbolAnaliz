# Unresolved External Blockers

_Generated: 2026-07-18T17:00:38.674666+00:00_

## EXTERNAL_BLOCKER_OFFICIAL_GT
- **sn-trackeval**: official SoccerNet tracking leaderboard GT not available locally. Synthetic evaluator PASS remains; leaderboard NOT claimed.

## EXTERNAL_BLOCKER_DATA_ACCESS
- **sn-mvfoul**: multi-view foul videos require NDA/special access. Devkit PASS remains; model inference NOT claimed.
- **sn-nvs**: SN-NVS scene dataset (HF) required to train/evaluate full NVS; no published ready-to-run scene checkpoint. CUDA build now works (smoke).
- **SoccerNet-v3**: Frames-v3 dataset access; repo is a dataloader/devkit, no model checkpoint published.

## CHECKPOINT_NOT_PUBLISHED
- **sn-caption**: no official caption model checkpoint found in README/releases/HF/Drive.
- **ActiveSpotting**: NetVLAD smoke only (previous audit); no published full-pipeline checkpoint.

## Not dependency problems
- **sn-jersey**: original is N/A_README_ONLY; clean-room COMPATIBLE_CLEANROOM_PASS unchanged.
- **sn-echoes**: dataset/devkit, not an ASR inference model.
