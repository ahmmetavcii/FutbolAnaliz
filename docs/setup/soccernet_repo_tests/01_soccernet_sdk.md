# SoccerNet SDK — Repo Verification Report (01)

- **Repo:** `/home/ahmet/projects/soccernet/SoccerNet`
- **Remote:** https://github.com/SoccerNet/SoccerNet.git
- **Branch:** detached HEAD
- **Commit:** `74461027ac2095ce2f8d4ee991eccb5dd5f42459`
- **Environment:** `ai-dev` (Python 3.10.20, torch 2.11.0+cu128, CUDA available: True)
- **Date:** 2026-07-18
- **Overall result:** PASS
- **Passed test groups:** 7 / 7
- **Failed test groups:** 0
- **Log:** `/home/ahmet/projects/football-analytics/logs/soccernet_repo_tests/01_soccernet_sdk.log`

## Test Results

### 1. Repo integrity — PASS
- Folder exists: YES
- Remote: `https://github.com/SoccerNet/SoccerNet.git`
- Branch: detached HEAD
- Commit SHA: `74461027ac2095ce2f8d4ee991eccb5dd5f42459`
- Dirty status: clean (no changes)
- `git fsck --full`: clean (no errors/dangling)
- Disk size: 120M total (104M excluding `.git`)

### 2. ai-dev environment — PASS
- python: `/home/ahmet/miniconda3/envs/ai-dev/bin/python`
- Python 3.10.20
- torch 2.11.0+cu128, `cuda.is_available()` = True
- numpy 2.2.6, opencv 5.0.0 (read-only, not modified)

### 3. SoccerNet package — PASS
- `import SoccerNet` → IMPORT PASS
- `from SoccerNet.Downloader import SoccerNetDownloader` → DOWNLOADER PASS
- `pip show SoccerNet` → version 0.1.62 (installed in site-packages)
- `pip check` → No broken requirements found
- Note: imported package is the installed site-packages build (v0.1.62). Repo source `SoccerNet/__init__.py` is also v0.1.62 — versions match.

### 4. Source syntax (compileall) — PASS
- `python -m compileall -q <repo>` → exit code 0 (no syntax errors)

### 5. Repo tests / CLI — PASS
- No `pytest`/`unittest` suite present in the repo.
- Detected demo/visualization scripts:
  - `visualization_tools/test.py` — requires `config.yaml` + `tracking_data` + tracking dataset. **Skipped** (needs data; not run).
  - `visualization_tools/test_fifa2022.py` — hardcoded path `/home/karkid/...` + NDA tracking data. **Skipped** (missing data/path; not run).
- Safe CLI test: `python SoccerNet/Downloader.py --help` → exit code 0, usage printed (no network).

### 6. Downloader init (no network) — PASS
- Instantiated `SoccerNetDownloader(LocalDirectory=<temp empty dir>)`.
- `LocalDirectory` attribute set correctly; `downloadGames` method present.
- No network call made; temp dir remained empty and was removed.

### 7. Jersey dataset read-only — PASS
- Dataset: `/mnt/c/football_data/datasets/SoccerNet/jersey-2023`
- `train/` and `test/` dirs exist; `train/images` and `test/images` exist.
- JSON readable: `train/train_gt.json` (1427 entries), `test/test_gt.json` (1211 entries).
- Total image files: 1,297,548.
- Zero-byte image files: 0.
- Random 10 images opened with OpenCV: 10/10 successful.
- No data downloaded, modified, or deleted.

## Caveats / Notes
- Repo is in a detached HEAD state (not on a named branch).
- The two demo scripts under `visualization_tools/` could not be executed because they require external tracking datasets (NDA / not present). This does not affect core SDK functionality (import, Downloader class, CLI, syntax, local dataset read all pass).
- The active `SoccerNet` import resolves to the installed site-packages build (v0.1.62), which matches the repo source version.

## Open Blockers
- None blocking core SDK usage. (Demo/visualization scripts remain unverifiable without their tracking datasets.)
