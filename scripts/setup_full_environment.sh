#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${HOME}/projects/football-analytics"
LOG_ROOT="${HOME}/logs"
mkdir -p "$LOG_ROOT"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate ai-dev
python "$ROOT/scripts/scaffold_project.py"
python "$ROOT/scripts/check_project.py"
python "${HOME}/dev-check/check_env.py"
