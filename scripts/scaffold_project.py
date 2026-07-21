#!/usr/bin/env python3
"""Create the idempotent football-analytics project scaffold."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/ahmet/projects/football-analytics")

DIRECTORIES = [
    "src/football_analytics/contracts",
    "src/football_analytics/adapters",
    "src/football_analytics/stages",
    "src/football_analytics/orchestration",
    "src/football_analytics/analytics",
    "src/football_analytics/datasets",
    "src/football_analytics/evaluation",
    "src/football_analytics/visualization",
    "src/football_analytics/video",
    "src/football_analytics/api",
    "src/football_analytics/utils",
    "configs/system",
    "configs/pipeline",
    "configs/soccernet_install",
    "schemas",
    "workflows",
    "scripts/soccernet_setup",
    "tests",
    "notebooks",
    "docs/setup",
    "patches",
    "requirements",
    ".vscode",
]

FILES = {
    "pyproject.toml": """[build-system]
requires = ["setuptools>=75,<82", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "football-analytics"
version = "0.1.0"
description = "Reproducible football computer-vision pipelines"
requires-python = ">=3.10,<3.11"
dependencies = ["PyYAML>=6", "pyarrow>=15", "pydantic>=2"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.black]
line-length = 100

[tool.ruff]
line-length = 100
target-version = "py310"
""",
    "environment.yml": """name: ai-dev
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip=26.1.2
  - setuptools=81.0.0
  - pip:
      - torch==2.11.0
      - torchvision==0.26.0
      - torchaudio==2.11.0
      - numpy==2.2.6
      - pandas==2.3.3
      - scipy==1.15.3
      - scikit-learn==1.7.2
      - ultralytics==8.4.91
      - SoccerNet==0.1.62
""",
    ".gitignore": """__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/
.env
.venv/
dist/
build/
*.egg-info/
runs/
models/
*.mp4
*.mkv
*.pth
*.pt
""",
    "README.md": """# Football Analytics

Reproducible, contract-first football computer-vision pipelines for WSL2.

## Start

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-dev
python scripts/check_project.py
python ~/dev-check/check_env.py
```

Active code and environments stay under `/home/ahmet`; large datasets and
archives stay under `/mnt/c/football_data`.
""",
    "LICENSE": """MIT License

Copyright (c) 2026 Ahmet Avcı

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
""",
    "THIRD_PARTY_NOTICES.md": """# Third-Party Notices

Each external repository remains subject to its own upstream license.
Exact remotes and commits are recorded in `external_repos.lock.yaml`.

- SoccerNet repositories: see each repository's LICENSE and SoccerNet data terms.
- TrackLab: see `/home/ahmet/projects/third-party/tracklab/LICENSE`.
- PnLCalib: see `/home/ahmet/projects/third-party/pnlcalib/LICENSE`.
- No-Bells-Just-Whistles: see its upstream LICENSE.

No SoccerNet NDA dataset is redistributed by this repository.
""",
    "configs/system/paths.yaml": """system:
  project_root: /home/ahmet/projects/football-analytics
  soccernet_root: /home/ahmet/projects/soccernet
  third_party_root: /home/ahmet/projects/third-party
  active_models: /home/ahmet/models
  workspace: /home/ahmet/workspace
  runs: /home/ahmet/workspace/runs
  staging: /home/ahmet/workspace/staging
  cache: /home/ahmet/workspace/cache

storage:
  ssd_root: /mnt/c/football_data
  raw_matches: /mnt/c/football_data/videos/raw_matches
  test_clips: /mnt/c/football_data/videos/test_clips
  datasets: /mnt/c/football_data/datasets
  results: /mnt/c/football_data/results
  rendered_outputs: /mnt/c/football_data/rendered_outputs
  reports: /mnt/c/football_data/reports
  model_archive: /mnt/c/football_data/model_archive
  experiments_archive: /mnt/c/football_data/experiments_archive
  backups: /mnt/c/football_data/backups
""",
    "external_repos.lock.yaml": """schema_version: 1
generated_at: null
repositories: {}
""",
    "model_registry.yaml": """schema_version: 1
models: {}
""",
    "dataset_registry.yaml": """schema_version: 1
datasets: {}
""",
    ".vscode/extensions.json": json.dumps(
        {
            "recommendations": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "ms-toolsai.jupyter",
                "charliermarsh.ruff",
                "ms-python.black-formatter",
                "ms-python.isort",
                "redhat.vscode-yaml",
                "tamasfe.even-better-toml",
                "usernamehw.errorlens",
                "eamodio.gitlens",
                "yzhang.markdown-all-in-one",
            ]
        },
        indent=2,
    )
    + "\n",
    ".vscode/settings.json": json.dumps(
        {
            "python.defaultInterpreterPath": "/home/ahmet/miniconda3/envs/ai-dev/bin/python",
            "python.testing.pytestEnabled": True,
            "python.testing.unittestEnabled": False,
            "ruff.enable": True,
            "[python]": {
                "editor.defaultFormatter": "ms-python.black-formatter",
                "editor.formatOnSave": True,
                "editor.codeActionsOnSave": {
                    "source.organizeImports": "explicit",
                    "source.fixAll.ruff": "explicit",
                },
            },
            "isort.args": ["--profile", "black"],
        },
        indent=2,
    )
    + "\n",
    "src/football_analytics/__init__.py": '"""Football analytics package."""\n\n__version__ = "0.1.0"\n',
    "scripts/check_project.py": """#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "pyproject.toml",
    ROOT / "configs/system/paths.yaml",
    ROOT / "external_repos.lock.yaml",
    ROOT / "model_registry.yaml",
    ROOT / "dataset_registry.yaml",
]


def main() -> int:
    failures = []
    for path in REQUIRED:
        if not path.is_file():
            failures.append(f"missing file: {path}")
    paths_file = ROOT / "configs/system/paths.yaml"
    if paths_file.is_file():
        data = yaml.safe_load(paths_file.read_text())
        for section in ("system", "storage"):
            for name, value in data.get(section, {}).items():
                if not Path(value).exists():
                    failures.append(f"missing path: {section}.{name}={value}")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("PASS project configuration and paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
""",
    "scripts/check_all_envs.py": """#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys

ENVS = [
    "ai-dev", "sn-trackeval", "sn-calibration", "sn-teamspotting",
    "sn-mvfoul", "sn-pts-baseline", "sn-reid", "sn-echoes",
    "sn-active-spotting", "sn-banner-mmseg", "sn-banner-replacement", "sn-nvs",
]


def main() -> int:
    raw = subprocess.check_output(["conda", "env", "list", "--json"], text=True)
    paths = json.loads(raw)["envs"]
    names = {path.rsplit("/", 1)[-1] for path in paths}
    for env in ENVS:
        print(("PASS" if env in names else "MISSING"), env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
""",
    "scripts/setup_full_environment.sh": """#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${HOME}/projects/football-analytics"
LOG_ROOT="${HOME}/logs"
mkdir -p "$LOG_ROOT"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate ai-dev
python "$ROOT/scripts/scaffold_project.py"
python "$ROOT/scripts/check_project.py"
python "${HOME}/dev-check/check_env.py"
""",
    "scripts/bootstrap_external_repos.sh": """#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${HOME}/projects/football-analytics"
python "$ROOT/scripts/soccernet_setup/run_install_orchestrator.py" --clone-only
""",
    "scripts/soccernet_setup/install_repo.sh": """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 REPOSITORY" >&2
  exit 2
fi
exec python "${HOME}/projects/football-analytics/scripts/soccernet_setup/run_install_orchestrator.py" --repo "$1"
""",
    "scripts/soccernet_setup/run_install_orchestrator.py": """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path.home() / "projects" / "soccernet"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone-only", action="store_true")
    parser.add_argument("--repo")
    args = parser.parse_args()
    cmd = [str(Path.home() / "projects/football-analytics/scripts/bootstrap_external_repos.py")]
    if args.repo:
        cmd.extend(["--repo", args.repo])
    return subprocess.call([sys.executable, *cmd])


if __name__ == "__main__":
    sys.exit(main())
""",
    "scripts/generate_setup_report.py": """#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    status = {"generated_at": dt.datetime.now().astimezone().isoformat(), "status": "IN_PROGRESS"}
    status_path = ROOT / f"configs/soccernet_install/status_{stamp}.json"
    status_path.write_text(json.dumps(status, indent=2) + "\\n")
    (ROOT / "configs/soccernet_install/status_latest.json").write_text(
        json.dumps(status, indent=2) + "\\n"
    )
    report = f"# Ahmet Football Setup Report\\n\\nGenerated: {status['generated_at']}\\n"
    report_path = ROOT / f"docs/setup/ahmet_full_install_report_{stamp}.md"
    report_path.write_text(report)
    (ROOT / "docs/setup/ahmet_full_install_report_latest.md").write_text(report)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
    "docs/setup/manual_actions_required.md": """# Manual Actions Required

No action is currently required. SoccerNet NDA credentials or unavailable
weights will be listed here when an installation reaches that boundary.
""",
}


def main() -> int:
    for relative in DIRECTORIES:
        (ROOT / relative).mkdir(parents=True, exist_ok=True)
    for relative, content in FILES.items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
    for package_dir in (ROOT / "src/football_analytics").iterdir():
        if package_dir.is_dir():
            init = package_dir / "__init__.py"
            init.touch(exist_ok=True)
    for script in (ROOT / "scripts").rglob("*"):
        if script.is_file() and (script.suffix in {".py", ".sh"}):
            script.chmod(script.stat().st_mode | 0o111)
    print(f"Scaffold ready: {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
