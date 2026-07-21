#!/usr/bin/env python3
"""Idempotently clone and pin the official SoccerNet repositories."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT = Path("/home/ahmet/projects/football-analytics")
ROOT = Path("/home/ahmet/projects/soccernet")
LOG_ROOT = Path("/home/ahmet/logs/football_setup_20260717_234122/repos")

REPOSITORIES = {
    "sn-tracking": "b0bbba35e07ff58010b6313ef8aa59ef663ad392",
    "sn-trackeval": "9c25232f6f2b56c9f203f1eb55784ff1e97df683",
    "sn-spotting": "9842826",
    "sn-calibration": "ab38f46",
    "sn-reid": "621e2b0",
    "sn-grounding": "910bf85",
    "SoccerNet-v3": "7d483a8",
    "sn-jersey": "2f43b48",
    "sn-caption": "c05973d",
    "SoccerNet": "7446102",
    "PTS-baseline": "af2ea82",
    "sn-mvfoul": "502fb44",
    "ActiveSpotting": "33a81cb",
    "sn-gamestate": "1c95834",
    "sn-depth": "9f6636f",
    "sn-echoes": "7105a85",
    "sn-teamspotting": "091fed2",
    "sn-banner": "f6d50b2",
    "sn-nvs": "1655ab1",
}


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    log=None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if log:
        log.write(f"$ {' '.join(command)}\n{result.stdout}[exit={result.returncode}]\n")
        log.flush()
    if check and result.returncode:
        raise RuntimeError(f"{command[0]} failed ({result.returncode}): {result.stdout[-1000:]}")
    return result


def install(name: str) -> dict:
    requested = REPOSITORIES[name]
    remote = f"https://github.com/SoccerNet/{name}.git"
    target = ROOT / name
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with (LOG_ROOT / f"{name}.log").open("a") as log:
        log.write(f"\n=== {dt.datetime.now().astimezone().isoformat()} ===\n")
        run(["git", "ls-remote", remote, "HEAD"], log=log)
        if not target.exists():
            run(["git", "clone", remote, str(target)], log=log)
        if not (target / ".git").is_dir():
            raise RuntimeError(f"Existing path is not a Git repository: {target}")
        origin = run(["git", "remote", "get-url", "origin"], cwd=target, log=log).stdout.strip()
        if origin.rstrip("/") != remote.rstrip("/"):
            raise RuntimeError(f"Remote mismatch for {name}: {origin}")
        exists = run(
            ["git", "cat-file", "-e", f"{requested}^{{commit}}"],
            cwd=target,
            check=False,
            log=log,
        )
        if exists.returncode:
            run(["git", "fetch", "--tags", "origin"], cwd=target, log=log)
        full = run(
            ["git", "rev-parse", f"{requested}^{{commit}}"], cwd=target, log=log
        ).stdout.strip()
        run(["git", "checkout", "--detach", full], cwd=target, log=log)
        dirty = bool(run(["git", "status", "--porcelain"], cwd=target, log=log).stdout.strip())
        fsck = run(["git", "fsck", "--no-progress"], cwd=target, check=False, log=log)
        size = run(["du", "-sb", str(target)], log=log).stdout.split()[0]
        branch = run(
            ["git", "symbolic-ref", "--short", "-q", "HEAD"],
            cwd=target,
            check=False,
            log=log,
        ).stdout.strip() or "DETACHED"
        if fsck.returncode:
            raise RuntimeError(f"git fsck failed for {name}")
        return {
            "path": str(target),
            "remote": remote,
            "commit": full,
            "requested_commit": requested,
            "branch": branch,
            "dirty": dirty,
            "disk_bytes": int(size),
            "git_fsck": "PASS",
            "integration": "external_official_repository",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", choices=sorted(REPOSITORIES))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    selected = [args.repo] if args.repo else list(REPOSITORIES)
    lock_path = PROJECT / "external_repos.lock.yaml"
    lock = yaml.safe_load(lock_path.read_text()) if lock_path.exists() else {}
    lock.setdefault("schema_version", 1)
    lock.setdefault("repositories", {})
    failures: dict[str, str] = {}
    for name in selected:
        print(f"[clone] {name}", flush=True)
        try:
            lock["repositories"][name] = install(name)
            print(f"[ok] {name}: {lock['repositories'][name]['commit']}", flush=True)
        except Exception as exc:
            failures[name] = str(exc)
            print(f"[fail] {name}: {exc}", file=sys.stderr, flush=True)
        lock["generated_at"] = dt.datetime.now().astimezone().isoformat()
        lock_path.write_text(yaml.safe_dump(lock, sort_keys=False, allow_unicode=True))
    if failures:
        print(yaml.safe_dump({"failures": failures}, allow_unicode=True), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
