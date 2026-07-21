# sn-echoes ↔ football-analytics — Integration Mapping

Status: **mapping documentation only** (no production integration written).

## What sn-echoes is (and is not)

- **Is:** a **dataset** repository — Whisper ASR transcriptions + Google-Translate English
  translations of SoccerNet broadcast **audio commentary**, stored as JSON.
- **Is not:** an ASR/inference model or pipeline. The repo contains **no** Whisper code,
  no model weights, and no audio→text runner. It only ships the transcribed JSON plus a
  tiny `stats.py` counter.

## Capability assessment for football-analytics

| Capability | Result | Rationale |
|---|---|---|
| Transcription of new video (e.g. `football.mp4`) | **NOT_APPLICABLE** | Repo provides no inference code; cannot generate transcripts. `football.mp4` also not present on disk. |
| Reading / using the shipped commentary dataset | **PASS** | 4,752 JSON files parsed, schema validated, `stats.py` cross-check matched (3,566,675 segments). |
| Direct hook into detection/tracking pipeline | **No (not directly)** | Our pipeline consumes video frames → boxes/tracks; sn-echoes is text-on-a-time-axis. No shared key beyond a time axis, and only for SoccerNet broadcast matches (not our `test_clip.mp4`). |
| Future commentary↔event alignment | **Possible (future work)** | Segments carry `[start_s, end_s, text]`; these can be aligned to a match event timeline once a common clock/offset is established. |

## Data model

sn-echoes segment JSON (per half, `1_asr.json` / `2_asr.json`):

```json
{ "segments": { "0": [start_seconds, end_seconds, "text"], "1": [ ... ] } }
```

football-analytics `tracks.parquet` (per run): frame/track rows with
`frame_id`, `timestamp_ms`, `track_id`, bbox, etc. (video-derived, no audio/text).

## Alignment sketch (future, NOT implemented here)

To align commentary with our video timeline one would:

1. Establish a shared clock: map sn-echoes `start_seconds`/`end_seconds` (per half) to the
   video timeline using `tracks.parquet.timestamp_ms` (and a per-half start offset).
2. For a target event time `t`, select commentary segments whose `[start,end]` interval
   overlaps `[t-Δ, t+Δ]` to attach spoken context to detected/tracked events.
3. Use the `_en` variant for a language-normalized text channel (keys/times align with the
   base variant within floating-point epsilon; see report).

## Hard constraints / blockers

- **No transcription pipeline** → cannot produce transcripts for arbitrary video.
- **Match scope mismatch:** sn-echoes covers SoccerNet broadcast matches; our current runs
  use `test_clip.mp4`. There is no commentary in sn-echoes for our clip, so end-to-end
  alignment cannot be demonstrated on current local footage.
- Alignment additionally needs the corresponding SoccerNet **video/event** timeline
  (matching match), which is outside this dataset.

## Not done here (by design)

- No converter/loader integrated into the pipeline.
- No transcription attempted on any local video.
