#!/usr/bin/env python3
"""Equal-budget PPO continuation with zero-initialised tariff intercepts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import torch

from train_flex_coordination import EXPERIMENTS, run_arm
from energy_net.coordination.flex_market import FlexMarketConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--platform-steps", type=int, default=16384)
    parser.add_argument("--joint-cycles", type=int, default=4)
    args = parser.parse_args()
    if min(args.platform_steps, args.joint_cycles) < 1:
        parser.error("training budgets must be positive")
    torch.set_num_threads(1)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    initial = output / "initial_users"; initial.mkdir(exist_ok=True)
    for user in range(3):
        shutil.copy2(args.source / "initial_users" / f"user_{user}.zip", initial / f"user_{user}.zip")
    started = time.monotonic()
    results = {}
    for name, fields in EXPERIMENTS.items():
        source = args.source / name
        report = json.loads((source / "summary.json").read_text())
        results[name] = run_arm(output / name, FlexMarketConfig(**fields), initial,
            args.platform_steps, args.joint_cycles, args.seed,
            resume_platform=source / "best_fixed" / "platform.zip",
            prior_platform_steps_spent=report["trained_platform_steps"])
    summary = dict(seed=args.seed, source=str(args.source.resolve()),
        elapsed_seconds=time.monotonic() - started,
        architecture="zero-initialised learned menu intercept plus retained shared conditional scorer",
        initial_users="byte-identical to pilot; no additional adaptation between mechanism arms",
        selection="validation cash gain only; no test selection",
        requested={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        experiments=results)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"completed_continuation": {"elapsed_seconds": summary["elapsed_seconds"]}}), flush=True)


if __name__ == "__main__":
    main()
