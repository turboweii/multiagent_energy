#!/usr/bin/env python3
"""Train one pricing PPO and three independent task-owner PPOs.

All inputs are synthetic. Each cycle updates one actor at a time, with periodic
validation on separate scenarios and final held-out evaluation.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

# Also support direct execution without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from energy_net.coordination.learning import (
    ActorEnv, PolicyBundle, alternating_train, cached_episode_factory, evaluate_bundle,
)
from energy_net.coordination.scenario import MarketConfig


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--platform-days", type=int, default=16)
    parser.add_argument("--user-steps", type=int, default=64)
    parser.add_argument("--eval-days", type=int, default=5)
    parser.add_argument("--eval-every-cycles", type=int, default=0,
                        help="0 disables intermediate validation/checkpoints")
    parser.add_argument("--validation-days", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.cycles, args.platform_days, args.user_steps, args.eval_days,
           args.validation_days) < 1 or args.eval_every_cycles < 0:
        parser.error("cycles and all sample/day counts must be positive")
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    config = MarketConfig()
    train_seeds = list(range(64))
    eval_seeds = list(range(10000, 10000 + args.eval_days))
    factory = cached_episode_factory(config, maxsize=len(train_seeds))
    prototype = factory(train_seeds[0])
    bundle = PolicyBundle()
    envs = {actor: ActorEnv(factory, bundle, actor, train_seeds, prototype)
            for actor in ["platform", *range(config.n_users)]}
    log = []
    validation_seeds = list(range(9000, 9000 + args.validation_days))
    validation_factory = cached_episode_factory(config, maxsize=args.validation_days)
    progress = []

    def means(rows):
        return {key: float(np.mean([row[key] for row in rows]))
                for key, value in rows[0].items()
                if isinstance(value, (int, float, bool, np.number))}

    def phase(record, models):
        log.append(record)
        record["elapsed_seconds"] = time.monotonic() - started
        write_csv(output / "training.csv", log)
        label = "platform" if record["actor"] == "platform" else f"user_{record['actor']}"
        actor = "platform" if label == "platform" else int(record["actor"])
        models[actor].save(checkpoint / label)
        print(json.dumps(record), flush=True)
        completed_cycle = record["cycle"] + 1
        if (actor == config.n_users - 1 and args.eval_every_cycles
                and (completed_cycle % args.eval_every_cycles == 0
                     or completed_cycle == args.cycles)):
            rows = evaluate_bundle(bundle, validation_factory, validation_seeds)
            report = dict(cycle=completed_cycle,
                          elapsed_seconds=time.monotonic() - started, **means(rows))
            progress.append(report)
            write_csv(output / "validation_progress.csv", progress)
            directory = output / "snapshots" / f"cycle_{completed_cycle:04d}"
            directory.mkdir(parents=True, exist_ok=True)
            for key, model in models.items():
                label = "platform" if key == "platform" else f"user_{key}"
                model.save(directory / label)
            write_csv(directory / "validation.csv", rows)
            print(json.dumps({"validation": report}), flush=True)

    started = time.monotonic()
    models, records = alternating_train(
        bundle, envs, cycles=args.cycles, platform_days=args.platform_days,
        user_steps=args.user_steps, seed=args.seed, on_phase=phase,
    )
    evaluation_factory = cached_episode_factory(config, maxsize=min(64, args.eval_days))
    rows = evaluate_bundle(bundle, evaluation_factory, eval_seeds)
    write_csv(output / "eval.csv", rows)
    numeric = means(rows)
    summary = {
        "status": "completed", "seed": args.seed,
        "elapsed_seconds": time.monotonic() - started,
        "config": asdict(config),
        "requested": {"cycles": args.cycles, "platform_days": args.platform_days,
                      "user_steps": args.user_steps, "eval_days": args.eval_days},
        "training_scenario_seeds": train_seeds, "evaluation_scenario_seeds": eval_seeds,
        "validation_scenario_seeds": validation_seeds if args.eval_every_cycles else [],
        "validation_progress": progress,
        "training": records, "evaluation_mean": numeric,
        "agents": ["platform", "user_0", "user_1", "user_2"],
        "opponents_during_each_training_phase": "frozen deterministic policies",
        "rewards": "platform profit change and each owner's own utility gain",
        "market_time_scale": "synthetic deterministic day-ahead scenarios",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for env in envs.values():
        env.close()
    print(json.dumps({"summary": str(output / "summary.json"),
                      "eval": str(output / "eval.csv")}), flush=True)


if __name__ == "__main__":
    main()
