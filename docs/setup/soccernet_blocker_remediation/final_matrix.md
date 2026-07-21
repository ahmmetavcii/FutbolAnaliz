# SoccerNet Blocker Remediation — Final Matrix

_Generated: 2026-07-18T17:00:38.674666+00:00_

| Repo | Prev original | New original | Compatible | Env | Dep fixed | Build fixed | Checkpoint | Real inference | Evaluator | Peak RAM | Peak VRAM | License | Remaining blocker | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sn-banner | BLOCKED_DEPENDENCY_CONFLICT | ORIGINAL_INFERENCE_PASS | NOT_IMPLEMENTED | sn-banner-runtime | True | N/A | official HF best_mIoU_iter_10935.pth (83 | YES | N/A | - | 3.33 | GPL-3.0 | - | DO_NOT_USE |
| sn-caption | BLOCKED_DEPENDENCY_CONFLICT | PASS_DEVKIT | NOT_IMPLEMENTED | sn-caption-eval | True | N/A | CHECKPOINT_NOT_PUBLISHED | no | PASS (BLEU/METEOR/ROUGE_L/CIDE | - | 0 | LICENSE_NOT_FOUND (roo | CHECKPOINT_NOT_PUBLISHED | DO_NOT_USE |
| sn-nvs | BLOCKED_BUILD_NO_NVCC | ORIGINAL_SMOKE_PASS | N/A | sn-nvs-build | True | True | CHECKPOINT_NOT_PUBLISHED (requires train | no | N/A | - | 0.018 | LICENSE_NOT_FOUND (roo | CHECKPOINT_NOT_PUBLISHED + EXTER | RESEARCH_ONLY |
| sn-gamestate | BLOCKED_RUNTIME_OOM | ORIGINAL_INFERENCE_PASS | COMPATIBLE_IMPLEMENTATION_PASS | sn-gamestate-env | N/A | N/A | existing open TrackLab assets (YOLO/PRTR | YES | N/A (no local GT) | 4.5 | 4.5 | GPL-3.0 | - | OPTIONAL |
| PTS-baseline | SOURCE_ONLY_NO_MODEL | ORIGINAL_INFERENCE_PASS | NOT_IMPLEMENTED | sn-pts-baseline | True | N/A | official e2e-spot-models soccer_rny002gs | YES | shared ActionSpotting (prev PA | - | - | BSD-3-Clause | - | OPTIONAL |
| sn-teamspotting | SOURCE_ONLY_NO_MODEL | ORIGINAL_INFERENCE_PASS | NOT_IMPLEMENTED | sn-teamspotting | True | N/A | official Drive SoccerNetBall_baseline/ch | YES | shared ActionSpotting | - | 0 | GPL-3.0 | - | OPTIONAL |
| sn-depth | SOURCE_ONLY_NO_MODEL | ORIGINAL_INFERENCE_PASS | NOT_IMPLEMENTED | sn-depth-runtime | True | N/A | official Drive ZoeDepthN_football.pt (3. | YES | N/A | - | 2.23 | LICENSE_NOT_FOUND (roo | - | RESEARCH_ONLY |
| SoccerNet-v3 | SOURCE_ONLY_NO_MODEL | CHECKPOINT_NOT_PUBLISHED | N/A | ai-dev (static) | False | N/A | CHECKPOINT_NOT_PUBLISHED (dataset/devkit | no | N/A | - | 0 | MIT | CHECKPOINT_NOT_PUBLISHED + EXTER | DO_NOT_USE |
| sn-mvfoul | PASS_DEVKIT | PASS_DEVKIT | NOT_IMPLEMENTED | sn-eval-env | N/A | N/A | Drive folder exists but multi-view video | no | PASS_DEVKIT (prev) | - | 0 | GPL-3.0 | EXTERNAL_BLOCKER_DATA_ACCESS | DO_NOT_USE |
| sn-trackeval | PASS (evaluator) / NOT_RUN_NO_OFFICIAL_GT | PASS_DEVKIT | EXISTING_COMPATIBLE_PASS | sn-trackeval-env / ai-dev | N/A | N/A | N/A | no | PASS (synthetic perfect/imperf | - | 0 | MIT | EXTERNAL_BLOCKER_OFFICIAL_GT | USE (via existing adapter) |
| sn-jersey | N/A_README_ONLY | N/A_README_ONLY | COMPATIBLE_CLEANROOM_PASS | sn-jersey-env / ai-dev | N/A | N/A | clean-room /home/ahmet/models/jersey_rec | no | N/A | - | 0 | README-only (no source | - | USE (clean-room) |
| sn-echoes | PASS (dataset/devkit) | PASS_DEVKIT | EXISTING_COMPATIBLE_PASS | ai-dev | N/A | N/A | N/A (no ASR model) | no | N/A | - | 0 | LICENSE_REVIEW_REQUIRE | - | USE (reader) |
