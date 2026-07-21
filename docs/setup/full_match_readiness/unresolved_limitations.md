# Unresolved limitations (2026-07-18)

These are real blockers; none can be closed by more code alone.

1. **No 45/90-minute video.** `FULL_MATCH_90MIN_PASS` cannot be claimed.
   The chunk scheduler, resume, and disk guard are exercised, but only on a
   30 s clip (single chunk). Multi-chunk behaviour over hours of footage,
   long-run drift, and disk pressure remain unproven.
2. **No same-match multi-camera recording.** Audio/visual sync, cross-camera
   ReID, fusion, and multi-camera global identity are infrastructure-tested
   with unit fixtures only (`MULTICAMERA_INFRASTRUCTURE_PASS`).
   `MULTICAMERA_REAL_VIDEO_PASS` and multi-camera global-ID claims stay open.
3. **No labeled events in the clip.** The 30 s broadcast segment contains no
   goal/shot/substitution that our conservative detectors could support, so
   confirmed=0/candidate=0 with reason `no_supported_event_detected`.
   `MATCH_EVENTS_REAL_VIDEO_PASS` additionally requires manual confirmation of
   goals/assists on real footage.
4. **Role separation is partial on this clip.** No officials-kit or
   goalkeeper-kit reference could be measured, so referee/assistant/goalkeeper
   roles were honestly not assigned (tracks stay `unknown_person` unless team
   kit evidence supports `outfield_player`). A longer clip with clear officials
   and penalty-box coverage is needed.
5. **Jersey numbers unresolved.** The trained recognizer scored 10 real
   tracklets; all predictions fell below the 0.6 confidence threshold on
   far-field broadcast crops. Higher-resolution or closer footage is required.
6. **Calibrated-metrics coverage is low (7.3 %).** The camera-motion-based
   calibration is valid on 94.7 % of frames, but per-track field coordinates
   pass the validity gates on only ~7 % of track rows in this fast panning
   clip; physical totals are computed only from valid samples (never zeroed).
7. **Single-camera global identity only.** 224 person tracks map to 224
   global identities; no cross-gap merge cleared the conservative threshold
   because the MVP-2 pipeline produces no re-identification embeddings, so
   ambiguous candidates were kept separate and flagged `unresolved` (223 of
   224) instead of risking wrong merges. Reported strictly as
   `GLOBAL_ID_SINGLE_CAMERA_PASS`.
