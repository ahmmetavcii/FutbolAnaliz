#!/usr/bin/env bash
set -Eeuo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 REPOSITORY" >&2
  exit 2
fi
exec python "${HOME}/projects/football-analytics/scripts/soccernet_setup/run_install_orchestrator.py" --repo "$1"
