# sn-echoes — Repo Verification Report (03)

- **Repo:** `/home/ahmet/projects/soccernet/sn-echoes`
- **Remote:** https://github.com/SoccerNet/sn-echoes.git
- **Branch:** detached HEAD
- **Commit:** `7105a85b7a8c1c000a31a30d0c29c388105c3de5`
- **Environment:** `ai-dev` (Python 3.10.20, torch 2.11.0+cu128, CUDA available: True, pandas 2.3.3)
- **Repo type:** **dataset** (SoccerNet-Echoes commentary transcriptions) — **not** an inference/ASR model
- **License status:** **LICENSE_NOT_VERIFIED**
- **Date:** 2026-07-18
- **Overall result:** PARTIAL
- **Passed test groups:** 11 / 11
- **Failed test groups:** 0
- **Log:** `/home/ahmet/projects/football-analytics/logs/soccernet_repo_tests/03_sn_echoes.log`
- **Artifacts:** `/home/ahmet/projects/football-analytics/artifacts/soccernet_repo_tests/sn-echoes`

## Summary

The SoccerNet-Echoes **dataset is fully present and valid**: all 4,752 JSON files parse,
3,566,675 segments were schema-validated with no structural corruption, and the repo's
`stats.py` runs and its total exactly matches an independent inventory. The overall status
is **PARTIAL** (not PASS) only because the **license could not be verified**
(no `LICENSE`/`COPYING` file, no license text in `README`/`CITATION.cff`), which the status
criteria classify as PARTIAL. This repo is a dataset, not a model, so it not producing
boxes/tracks/transcripts for video is **NOT_APPLICABLE**, not a failure.

## Test Results

### 1. Repo integrity — PASS
- Folder exists: YES; Remote: `https://github.com/SoccerNet/sn-echoes.git`
- Branch: detached HEAD; Commit: `7105a85b7a8c1c000a31a30d0c29c388105c3de5`
- `git fsck --full`: clean; Submodules: none
- Disk size: 599M (incl `.git`), 449M `Dataset/`; tracked files: 4,755
- **License: LICENSE_NOT_VERIFIED** (no LICENSE/COPYING; no license text in README/CITATION.cff)
- **Dirty state (pre-existing, not caused by this test):** `stats.py` is modified in the
  working tree and `__pycache__/` is untracked. The working-tree change removes a stray
  trailing `s` (`")s`) that exists in the committed HEAD version (see Test 4).

### 2. Environment — PASS
- python: `/home/ahmet/miniconda3/envs/ai-dev/bin/python`, Python 3.10.20
- torch 2.11.0+cu128, CUDA True; pandas 2.3.3; `json` stdlib OK; numpy 2.2.6; opencv 5.0.0
- `pip check`: the only warnings (`scikit-image`, `tabulate` missing) belong to **sn-trackeval**,
  not sn-echoes (which uses only the stdlib `json`). Env packages **not modified**.

### 3. Repo structure & purpose — PASS
- **Dataset repo** (per README/CITATION.cff/arXiv 2405.07354). No Whisper/ASR/model code.
- Whisper variants present (verified on disk, matches README):
  `whisper_v1`, `whisper_v1_en`, `whisper_v2`, `whisper_v2_en`, `whisper_v3` (no `whisper_v3_en`).
- Hierarchy: `Dataset/<variant>/<league>/<season>/<match>/{1_asr.json,2_asr.json}`; 6 leagues per variant.
- Actual JSON structure (matches README): top-level `segments` dict; keys are string ints;
  each value is an **array** `[start(float), end(float), text(str)]` (README says index is int;
  JSON keys are strings — minor note).
- `stats.py`: counts `len(segments)` over `Dataset/**/*.json` via `os.getcwd()`; **no** argparse/`--help`;
  read-only; must be run from the repo root.
- External audio/video needed to read the dataset: **none**. Inference/transcription code: **none**.

### 4. Syntax & script — PASS
- `python -m compileall -q stats.py` → exit 0.
- **Committed** `stats.py` (HEAD) ends with `...tot_count}")s` — a stray `s` that is a
  **SyntaxError**; the working tree already fixes this (pre-existing modification). Tests
  used the working-tree version; the repo was **not** modified by this test.
- No repo Python unit tests present.

### 5. Dataset inventory (streaming, read-only) — PASS
- Total JSON files: **4,752**; total segments: **3,566,675**
- Zero-byte files: **0**; unparseable JSON: **0**; encoding-problem files: **0**
- Empty-segment (valid but 0 segments) files: **155**
- League dirs: 30 (6 unique leagues × 5 variants); seasons total: 93; matches total: 2,384
- Per variant:

| variant | files | segments | leagues | seasons | matches | empty-seg |
|---|---|---|---|---|---|---|
| whisper_v1 | 1100 | 780,160 | 6 | 19 | 550 | 50 |
| whisper_v1_en | 734 | 563,064 | 6 | 18 | 367 | 4 |
| whisper_v2 | 1100 | 761,240 | 6 | 19 | 550 | 45 |
| whisper_v2_en | 718 | 538,990 | 6 | 18 | 367 | 0 |
| whisper_v3 | 1100 | 923,221 | 6 | 19 | 550 | 56 |

### 6. JSON schema validation — PASS
- Files with structural issues: **0** / 4,752; segments checked: 3,566,675.
- No `end < start`, no negative time, no NaN/Inf, no non-numeric time, no non-string text,
  no malformed segment values, no reverse-ordered segments, no duplicate identical segments.
- Benign data-quality notes (normal for ASR, not corruption):
  - empty-text segments: **530**
  - adjacent overlaps (`seg.start < prev.end`): **2,107**

### 7. Content quality smoke — PASS
- All 5 variants: 10/10 sampled files have non-empty segments, unicode text reads correctly,
  times are sensible, and per-match speech duration is computable.
- Original ↔ `_en` matching: same match path and identical segment keys; start/end times are
  equal within floating-point epsilon (~1e-14 JSON round-trip; e.g. `62.8` vs
  `62.800000000000004`). `_en` alignment is **viable**. (`_en` has fewer matches — 367 vs 550 —
  so not every base match has an `_en` counterpart.)

### 8. stats.py real run + cross-check — PASS
- `python stats.py` (from repo root) → exit 0; processed 4,752 files;
  "Total number of annotations across all files: **3,566,675**".
- **Cross-check MATCH** vs independent inventory (3,566,675). Full output saved to
  `03_sn_echoes_statspy_fulloutput.log`.

### 9. Synthetic schema fixtures (A–F) — PASS
Temporary fixtures created in staging and validated by a reference validator, then removed:
- A (valid 3 segments) → accepted; B (end<start) → rejected; C (missing text) → rejected;
  D (broken JSON) → rejected; E (empty segments) → VALID_EMPTY; F (Turkish/unicode) → accepted
  (round-trip verified). Result: **PASS**.

### 10. Artifacts — PASS
All present, non-empty, parseable:
- `dataset_inventory.json` (per-variant + totals)
- `schema_validation_summary.json` (issue counters + examples)
- `sample_segments.csv` (100 rows + header, 9 columns: dataset_version, league, season, match,
  half_file, segment_id, start_seconds, end_seconds, text_length; no long transcript text copied)

### 11. football-analytics compatibility — PASS (dataset reading) / NOT_APPLICABLE (transcription)
- Repo ships **no** ASR/inference code, and `football.mp4` is **not present** on disk →
  **transcription capability: NOT_APPLICABLE** (not a FAIL).
- **Dataset reading capability: PASS.**
- Cannot hook directly into the detection/tracking pipeline (text-on-time-axis vs video frames);
  future commentary↔event alignment is possible via a shared clock. See
  `sn_echoes_integration_mapping.md`. No production integration written.

## Overall: PARTIAL

- Repo integrity good; dataset fully present; all JSON parsed; schema validated; `stats.py`
  ran and cross-checked; artifacts produced and validated — all technical PASS criteria met.
- Classified **PARTIAL** because the **license is unverified** (LICENSE_NOT_VERIFIED), which the
  status criteria list explicitly as a PARTIAL condition.

## Open Blocker
- **LICENSE_NOT_VERIFIED** — no `LICENSE`/`COPYING` file and no license text in
  `README.md`/`CITATION.cff`. This did not block any data-validation test.
- (Informational, not a blocker) Transcription of new video is **NOT_APPLICABLE** — this is a
  dataset repo without inference code.

## Cleanup
- Temporary scan/validator scripts and synthetic fixtures were created under
  `/home/ahmet/workspace/staging/sn_echoes_smoke`; the A–F fixtures were removed by the test.
  The staging helper scripts are removed at the end of the run.
- No repo files were modified by this test; ai-dev packages unchanged.
