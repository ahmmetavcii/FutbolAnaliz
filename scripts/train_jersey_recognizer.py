#!/usr/bin/env python3
"""Train the temporal jersey recognizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_analytics.jersey.train import (  # noqa: E402
    load_config,
    train_jersey_recognizer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jersey/jersey_recognition_v1.yaml")
    parser.add_argument("--resume", nargs="?", const=True, default=None)
    parser.add_argument("--train-subset", type=int)
    parser.add_argument("--val-subset", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.resume is not None:
        config["training"]["resume"] = args.resume
    if args.train_subset is not None:
        config["dataset"]["train_subset"] = args.train_subset
    if args.val_subset is not None:
        config["dataset"]["val_subset"] = args.val_subset
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    print(json.dumps(train_jersey_recognizer(config), indent=2))


if __name__ == "__main__":
    main()
