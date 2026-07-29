# Tracking Refinement Research

Date: 2026-07-29  
Scope: spectator/staff FP reduction, ID continuity, bbox stability for broadcast football.

## Methods reviewed

| Method | Official source | License / notes | Checkpoint | RTX 4060 8GB | Relation to our problems | Decision |
|---|---|---|---|---|---|---|
| **SportsMOT + MixSort** | [arXiv:2304.05170](https://arxiv.org/abs/2304.05170), [MCG-NJU/SportsMOT](https://github.com/MCG-NJU/SportsMOT) | Dataset + paper; MixSort research code | Dataset-only for ROI policy; MixSort needs custom assoc model | N/A for dataset policy | Explicit objective: track **only players on playground**; exclude spectators, referees, coaches | **ADAPT** (annotation/policy: on-pitch-only tracking + specialized filtering). MixSort itself **REFERENCE_ONLY** (heavy custom assoc). |
| **Deep-EIoU** | [WACV 2024 RWS](https://openaccess.thecvf.com/content/WACV2024W/RWS/papers/Huang_Iterative_Scale-Up_ExpansionIoU_and_Deep_Features_Association_for_Multi-Object_Tracking_WACVW_2024_paper.pdf), [hsiangwei0903/Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU) (also MCG-NJU fork refs) | Research code; check repo LICENSE before prod | Needs ReID features + detector dets in MOT format | Feasible offline if isolated env; risk of dep conflict with ai-dev | Motion-agnostic expansion IoU helps fast sports motion / ID switches | **ISOLATED_EXPERIMENT** — not pip-installed into ai-dev; fallback if clone fails |
| **GTATrack / GTA-Link** | [arXiv:2602.00484](https://arxiv.org/abs/2602.00484), [ron941/GTATrack-STC2025](https://github.com/ron941/GTATrack-STC2025), GTA-Link [sjc042/gta-link](https://github.com/sjc042/gta-link) | Research; reproduction scripts | Deep-EIoU + offline GTA-Link; SoccerTrack fisheye-oriented | Heavy full stack; fisheye-specific pieces less relevant | Two-stage local → global tracklet association matches our fragmentation problem | **ADAPT** offline constrained global association ideas (veto-safe). Full GTATrack stack **ISOLATED_EXPERIMENT / REFERENCE_ONLY** |
| **BoT-SORT GMC + ReID** | [arXiv:2206.14651](https://arxiv.org/abs/2206.14651), [Ultralytics track docs](https://docs.ultralytics.com/modes/track), `botsort.yaml` | AGPL via Ultralytics already in ai-dev | Native YOLO features or external ReID; GMC=`sparseOptFlow` | Already proven ~778MB peak on our clip | Camera pan/zoom ID breakage; appearance for rebind | **USE** (primary online tracker; tune buffer, appearance_thresh, gmc_method) |
| **ByteTrack** | Ultralytics + original ByteTrack paper | Already integrated | None beyond detector | Excellent | Low-conf recovery; no appearance | **USE** as benchmark / fallback |
| **SoccerNet Tracking / GSR** | SoccerNet challenge trackers & Game State | Existing third_party / sn-* checkouts | Challenge-specific weights | Mixed | Role / game-state concepts for referee vs player | **REFERENCE_ONLY** for role taxonomy; do not retrain in this task |
| **Hard-negative mining / pseudo-label** | Standard CV practice; GTATrack also uses pseudo-label for small targets | N/A | Would produce candidate `.pt` only | Training YOLO26m on 8GB possible but slow/risky mid-task | Directly attacks stands/bench FP | **ADAPT** mining package now; **ISOLATED_EXPERIMENT** for retrain — do **not** overwrite `best.pt` |
| **Selective mask / VOS propagation** | XMem / SAM-Track family (general) | Separate heavy deps | Large VOS weights | 8GB borderline for full-video | Occlusion ID rescue | **REJECT** full-video; **ISOLATED_EXPERIMENT** only short high-uncertainty windows if deps already present — default skip |
| **Temporal bbox smoothing (EMA/Kalman)** | Classical; BoT-SORT already Kalman for assoc | N/A | None | Free | Renderer jitter | **USE** (render/smoothed coords separate from raw association) |
| **Tracklet-level role classification** | SportsMOT policy + SoccerNet roles | N/A | Heuristic + existing detector votes | Free | Coach→referee, stands→player | **USE / ADAPT** |

## Integration plan for this refinement

1. **Pitch / zone filter** (SportsMOT-inspired on-pitch policy) using foot bottom-center + homography + green + image region priors.  
2. **Track-level roles**: PLAYER / REFEREE / STAFF / SPECTATOR / UNRESOLVED_PERSON.  
3. **BoT-SORT** with GMC + ReID, longer buffer, stricter appearance for fewer ID flips.  
4. **Offline global association** (GTA-style constraints already in our veto graph) — keep non-transitive merges.  
5. **Deep-EIoU**: attempt isolated clone under `/home/ahmet/projects/tracker-research/`; if broken, continue with BoT-SORT.  
6. **Hard-negatives**: export crops + CSV; no overwrite of production weights.

## Honesty

No HOTA/IDF1 claimed. Proxy metrics only until GT completed.
