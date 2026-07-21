#!/usr/bin/env python3
"""Recompute ReID embeddings and/or global identity on an existing run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_analytics.evaluation.identity_fragments import classify_team_identities  # noqa: E402
from football_analytics.evaluation.reid_metrics import evaluate_reid_run  # noqa: E402
from football_analytics.stages.global_identity import GlobalIdentityStage  # noqa: E402
from football_analytics.stages.reid import ReidStage  # noqa: E402
from football_analytics.utils.io import read_yaml, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "pipeline" / "opta_analytics.yaml",
    )
    parser.add_argument(
        "--skip-reid",
        action="store_true",
        help="Only recompute global_identity from existing prototypes",
    )
    parser.add_argument(
        "--skip-identity",
        action="store_true",
        help="Only recompute reid embeddings/prototypes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run dir missing: {run_dir}")
    config = read_yaml(args.config)

    summary: dict = {"run_dir": str(run_dir)}

    if not args.skip_reid:
        stage = ReidStage(run_dir, config)
        stage.validate_inputs()
        stage.prepare()
        artifacts = stage.run()
        stage.validate_outputs(artifacts)
        summary["reid"] = json.loads((stage.stage_dir / "metrics.json").read_text(encoding="utf-8"))

    if not args.skip_identity:
        stage = GlobalIdentityStage(run_dir, config)
        stage.validate_inputs()
        stage.prepare()
        artifacts = stage.run()
        stage.validate_outputs(artifacts)
        summary["global_identity"] = json.loads(
            (run_dir / "identity_quality.json").read_text(encoding="utf-8")
        )
        for team in ("team_0", "team_1"):
            summary[f"fragments_{team}"] = classify_team_identities(run_dir, team_id=team)

    summary["reid_evaluation"] = evaluate_reid_run(run_dir)
    write_json(run_dir / "evaluation" / "reid_recompute_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
