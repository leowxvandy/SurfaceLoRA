#!/usr/bin/env python3
"""Train the standard LoRA summarization baseline."""

from __future__ import annotations

import argparse
import json

from surfacelora.training import train_baseline
from surfacelora.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device:
        config.setdefault("runtime", {})["device"] = args.device
    print(json.dumps(train_baseline(config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
