#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    status = {"generated_at": dt.datetime.now().astimezone().isoformat(), "status": "IN_PROGRESS"}
    status_path = ROOT / f"configs/soccernet_install/status_{stamp}.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    (ROOT / "configs/soccernet_install/status_latest.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    report = f"# Ahmet Football Setup Report\n\nGenerated: {status['generated_at']}\n"
    report_path = ROOT / f"docs/setup/ahmet_full_install_report_{stamp}.md"
    report_path.write_text(report)
    (ROOT / "docs/setup/ahmet_full_install_report_latest.md").write_text(report)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
