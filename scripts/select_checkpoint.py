#!/usr/bin/env python3
"""Select a checkpoint from aggregate validation metrics only."""

from __future__ import annotations

import argparse
import json

from surfacelora.checkpoint_selection import load_metric_records, select_checkpoint
from surfacelora.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--leakage-budget", type=float, default=0.025)
    parser.add_argument("--chance-accuracy", type=float, default=0.20)
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = select_checkpoint(
        load_metric_records(args.metrics_dir),
        leakage_budget=args.leakage_budget,
        chance_accuracy=args.chance_accuracy,
    )
    if args.output:
        write_json(selected, args.output)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
