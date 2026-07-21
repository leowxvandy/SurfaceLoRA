#!/usr/bin/env python3
"""Train SurfaceLoRA with paired utility and prompt-only adversarial branches."""

from __future__ import annotations

import argparse
import json

from surfacelora.training import train_surfacelora
from surfacelora.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--lambda-adv", type=float, help="Override training.lambda_adv")
    parser.add_argument("--output-dir", default="outputs/surfacelora")
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
    lambda_adv = float(args.lambda_adv if args.lambda_adv is not None else config["training"]["lambda_adv"])
    report = train_surfacelora(config, lambda_adv=lambda_adv, output_dir=args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
