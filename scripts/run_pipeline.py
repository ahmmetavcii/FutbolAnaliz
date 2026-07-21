#!/usr/bin/env python3
"""Run MVP-1 detection + tracking pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_analytics.orchestration.runner import PipelineRunner  # noqa: E402
from football_analytics.utils.io import read_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run football-analytics MVP-1 pipeline")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "pipeline" / "mvp1_tracking.yaml"),
    )
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Override runs directory (default: paths.yaml system.runs)",
    )
    parser.add_argument(
        "--resume-run-dir",
        default=None,
        help="Resume an existing run; completed stage manifests are validated and skipped",
    )
    parser.add_argument(
        "--rerun-from",
        default=None,
        help=(
            "With --resume-run-dir: force rerun from this stage onwards; "
            "earlier stages are skipped after checksum + output validation"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = read_yaml(ROOT / "configs" / "system" / "paths.yaml")
    runs_root = Path(args.runs_root) if args.runs_root else Path(paths["system"]["runs"])
    runner = PipelineRunner(
        config_path=Path(args.config),
        input_video=Path(args.input),
        runs_root=runs_root,
        resume_run_dir=Path(args.resume_run_dir) if args.resume_run_dir else None,
        rerun_from=args.rerun_from,
    )
    report = runner.run()
    print(json.dumps(report, indent=2))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
