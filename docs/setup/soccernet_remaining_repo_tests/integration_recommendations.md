# Integration Recommendations for football-analytics

_Generated: 2026-07-18T15:47:42.171691+00:00_

## sn-banner — DO_NOT_USE
- Adds: Broadcast overlay/banner handling — not core to tactical analysis
- Original status: BLOCKED_DEPENDENCY_CONFLICT; Compatible: NOT_IMPLEMENTED
- GPU/RAM fit: model not run in constraints
- License risk: GPL-3.0
- Recommendation: **DO_NOT_USE**

## sn-nvs — RESEARCH_ONLY
- Adds: 3D scene NVS — research; heavy CUDA build; not core
- Original status: BLOCKED_BUILD_NO_NVCC; Compatible: N/A
- GPU/RAM fit: model not run in constraints
- License risk: LICENSE_NOT_FOUND (root)
- Recommendation: **RESEARCH_ONLY**

## sn-teamspotting — OPTIONAL
- Adds: Ball-action spotting could complement event detection
- Original status: SOURCE_ONLY_NO_MODEL; Compatible: NOT_IMPLEMENTED
- GPU/RAM fit: model not run in constraints
- License risk: GPL-3.0
- Recommendation: **OPTIONAL**

## sn-depth — RESEARCH_ONLY
- Adds: Depth could aid 3D/scale but PnLCalib already gives pitch geometry
- Original status: SOURCE_ONLY_NO_MODEL; Compatible: NOT_IMPLEMENTED
- GPU/RAM fit: model not run in constraints
- License risk: LICENSE_NOT_FOUND (root)
- Recommendation: **RESEARCH_ONLY**

## sn-mvfoul — DO_NOT_USE
- Adds: Foul/VAR classification — niche; needs multi-view feeds we lack
- Original status: PASS_DEVKIT; Compatible: NOT_IMPLEMENTED
- GPU/RAM fit: model not run in constraints
- License risk: GPL-3.0
- Recommendation: **DO_NOT_USE**

## sn-caption — DO_NOT_USE
- Adds: Commentary already covered by sn-echoes reader; overlaps
- Original status: BLOCKED_DEPENDENCY_CONFLICT; Compatible: NOT_IMPLEMENTED
- GPU/RAM fit: model not run in constraints
- License risk: LICENSE_NOT_FOUND (root)
- Recommendation: **DO_NOT_USE**

## sn-spotting — OPTIONAL
- Adds: Action spotting is directly relevant to event detection
- Original status: PASS_DEVKIT; Compatible: NOT_IMPLEMENTED
- GPU/RAM fit: model not run in constraints
- License risk: MIT
- Recommendation: **OPTIONAL**

## sn-reid — USE
- Adds: ReID could strengthen long-term track identity
- Original status: PASS_DEVKIT; Compatible: NOT_IMPLEMENTED
- GPU/RAM fit: model not run in constraints
- License risk: MIT
- Recommendation: **USE**

## sn-tracking — OPTIONAL
- Adds: We already have tracking + trackeval adapter; overlaps
- Original status: PASS_DEVKIT; Compatible: EXISTING_COMPATIBLE_PASS
- GPU/RAM fit: model not run in constraints
- License risk: LICENSE_NOT_FOUND (root)
- Recommendation: **OPTIONAL**

## ActiveSpotting — RESEARCH_ONLY
- Adds: Active-learning sampling could reduce labeling cost
- Original status: ORIGINAL_SMOKE_PASS; Compatible: NOT_IMPLEMENTED
- GPU/RAM fit: OK (light)
- License risk: MIT
- Recommendation: **RESEARCH_ONLY**

## PTS-baseline — OPTIONAL
- Adds: Precise temporal spotting relevant to event timestamps
- Original status: SOURCE_ONLY_NO_MODEL; Compatible: NOT_IMPLEMENTED
- GPU/RAM fit: model not run in constraints
- License risk: BSD-3-Clause
- Recommendation: **OPTIONAL**

## sn-grounding — DO_NOT_USE
- Adds: Replay grounding is broadcast-specific; low relevance
- Original status: PASS_DEVKIT; Compatible: NOT_IMPLEMENTED
- GPU/RAM fit: model not run in constraints
- License risk: MIT
- Recommendation: **DO_NOT_USE**

## SoccerNet-v3 — DO_NOT_USE
- Adds: Overlaps SoccerNet SDK dataset access
- Original status: SOURCE_ONLY_NO_MODEL; Compatible: N/A
- GPU/RAM fit: model not run in constraints
- License risk: MIT
- Recommendation: **DO_NOT_USE**

