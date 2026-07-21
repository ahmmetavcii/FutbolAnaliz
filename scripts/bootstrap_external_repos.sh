#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${HOME}/projects/football-analytics"
python "$ROOT/scripts/soccernet_setup/run_install_orchestrator.py" --clone-only
