#!/usr/bin/env python3
"""Clone and lock approved third-party repositories."""

from __future__ import annotations

import datetime as dt
import os
import subprocess
from pathlib import Path

import yaml


PROJECT = Path("/home/ahmet/projects/football-analytics")
ROOT = Path("/home/ahmet/projects/third-party")
LOG = Path("/home/ahmet/logs/football_setup_20260717_234122/third_party.log")
REPOSITORIES = {
    "TrackLab": {
        "directory": "tracklab",
        "remote": "https://github.com/TrackingLaboratory/tracklab.git",
        "ref": "v1.3.24",
    },
    "PnLCalib": {
        "directory": "pnlcalib",
        "remote": "https://github.com/mguti97/PnLCalib.git",
        "ref": None,
    },
    "No-Bells-Just-Whistles": {
        "directory": "no-bells-just-whistles",
        "remote": "https://github.com/mguti97/No-Bells-Just-Whistles.git",
        "ref": None,
    },
}


def run(command: list[str], cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with LOG.open("a") as stream:
        stream.write(f"$ {' '.join(command)}\n{result.stdout}[exit={result.returncode}]\n")
    if check and result.returncode:
        raise RuntimeError(result.stdout[-1000:])
    return result.stdout.strip()


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = PROJECT / "external_repos.lock.yaml"
    lock = yaml.safe_load(lock_path.read_text())
    for name, spec in REPOSITORIES.items():
        target = ROOT / spec["directory"]
        remote = spec["remote"]
        run(["git", "ls-remote", remote, "HEAD"])
        if not target.exists():
            run(["git", "clone", remote, str(target)])
        origin = run(["git", "remote", "get-url", "origin"], target)
        if origin.rstrip("/") != remote.rstrip("/"):
            raise RuntimeError(f"{name} remote mismatch: {origin}")
        ref = spec["ref"]
        if ref:
            run(["git", "fetch", "--tags", "origin"], target)
            commit = run(["git", "rev-parse", f"{ref}^{{commit}}"], target)
        else:
            run(["git", "fetch", "origin"], target)
            commit = run(["git", "rev-parse", "origin/HEAD^{commit}"], target)
        run(["git", "checkout", "--detach", commit], target)
        dirty = bool(run(["git", "status", "--porcelain"], target))
        fsck = run(["git", "fsck", "--no-progress"], target, check=False)
        size = int(run(["du", "-sb", str(target)]).split()[0])
        lock["repositories"][name] = {
            "path": str(target),
            "remote": remote,
            "commit": commit,
            "requested_ref": ref or "remote default at install time",
            "branch": "DETACHED",
            "dirty": dirty,
            "disk_bytes": size,
            "git_fsck": "PASS" if not fsck else "PASS_WITH_OUTPUT",
            "integration": "third_party",
        }
        print(f"[ok] {name}: {commit}")
    lock["generated_at"] = dt.datetime.now().astimezone().isoformat()
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False, allow_unicode=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
