#!/usr/bin/env python3
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
