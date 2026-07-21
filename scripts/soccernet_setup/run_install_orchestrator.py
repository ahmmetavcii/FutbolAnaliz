#!/usr/bin/env python3
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
