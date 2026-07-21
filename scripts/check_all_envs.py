#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys

ENVS = [
    "ai-dev", "sn-trackeval", "sn-calibration", "sn-teamspotting",
    "sn-mvfoul", "sn-pts-baseline", "sn-reid", "sn-echoes",
    "sn-active-spotting", "sn-banner-mmseg", "sn-banner-replacement", "sn-nvs",
]


def main() -> int:
    raw = subprocess.check_output(["conda", "env", "list", "--json"], text=True)
    paths = json.loads(raw)["envs"]
    names = {path.rsplit("/", 1)[-1] for path in paths}
    for env in ENVS:
        print(("PASS" if env in names else "MISSING"), env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
