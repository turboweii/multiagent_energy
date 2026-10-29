#!/usr/bin/env python3
"""Train contract comparison, then four independent PPOs with compact tariffs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from energy_net.coordination.curriculum import pretrain_users, evaluate_user_responses
from energy_net.coordination.learning import (
    ActorEnv, PolicyBundle, alternating_train, cached_episode_factory, evaluate_bundle,
)
from energy_net.coordination.scenario import MarketConfig
from energy_net.coordination.tariffs import TariffCatalogue
from train_coordination import write_csv


def means(rows):
    return {key: float(np.mean([row[key] for row in rows]))
            for key, value in rows[0].items()
            if isinstance(value, (int, float, bool, np.number))}


def save_response(output, result):
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(
        {key: result[key] for key in ("summary", "per_user")}, indent=2), encoding="utf-8")
    write_csv(output / "tasks.csv", result["tasks"])
    write_csv(output / "episodes.csv", result["episodes"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrain-steps", type=int, default=16384)
    parser.add_argument("--pretrain-max-steps", type=int, default=65536)
    parser.add_argument("--cycles", type=int, default=32)
    parser.add_argument("--platform-days", type=int, default=512)
    parser.add_argument("--user-steps", type=int, default=1024)
    parser.add_argument("--eval-every-cycles", type=int, default=4)
    parser.add_argument("--eval-days", type=int, default=32)
    parser.add_argument("--validation-days", type=int, default=16)
    parser.add_argument("--pretrained-users", type=Path)
    parser.add_argument("--joint-user-gamma", type=float, default=.99)
    parser.add_argument("--opponent-sampling-probability", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.pretrain_steps, args.pretrain_max_steps, args.cycles,
           args.platform_days, args.user_steps, args.eval_every_cycles,
           args.eval_days, args.validation_days) < 1:
        parser.error("All budgets and day counts must be positive.")
    if not 0 <= args.joint_user_gamma <= 1 or not 0 <= args.opponent_sampling_probability <= 1:
        parser.error("Gamma and sampling probability must be in [0, 1].")
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = MarketConfig()
    catalogue = TariffCatalogue(config)
    train_seeds = list(range(64))
    validation_seeds = list(range(9000, 9000 + args.validation_days))
    test_seeds = list(range(10000, 10000 + args.eval_days))
    factory = cached_episode_factory(config, 64)
    validation_factory = cached_episode_factory(config, args.validation_days)
    pretrain_log, pretrain_progress = [], []
    started = time.monotonic()
    checkpoint = output / "pretrained_users"
    checkpoint.mkdir(exist_ok=True)

    def pretrain_phase(record, models):
        record = dict(record, elapsed_total_seconds=time.monotonic() - started)
        pretrain_log.append(record)
        write_csv(output / "pretraining.csv", pretrain_log)
        models[int(record["actor"])].save(checkpoint / f"user_{record['actor']}")
        if "validation" in record:
            row = dict(steps_per_user=record["cumulative_actor_steps"],
                       **record["validation"]["summary"])
            for owner, stats in record["validation"]["per_user"].items():
                row[f"user_{owner}_regret_yuan"] = stats["regret_yuan_per_task"]
                row[f"user_{owner}_ir_rate"] = stats["user_ir_rate"]
            pretrain_progress.append(row)
            write_csv(output / "pretraining_progress.csv", pretrain_progress)
        print(json.dumps({"pretraining": record}), flush=True)

    bundle, pretrain_passed, used = None, False, 0
    if args.pretrained_users:
        from stable_baselines3 import PPO
        bundle = PolicyBundle(users={owner: PPO.load(
            args.pretrained_users / f"user_{owner}.zip", device="cpu") for owner in range(3)})
        used = min(model.num_timesteps for model in bundle.users.values())
    while True:
        if bundle is None or not args.pretrained_users:
            steps = min(args.pretrain_steps, args.pretrain_max_steps - used)
            if steps <= 0:
                break
            bundle, _ = pretrain_users(
                factory, train_seeds, steps_per_user=steps, round_steps=2048,
                seed=args.seed, users=None if bundle is None else bundle.users,
                on_phase=pretrain_phase, validation_seeds=validation_seeds[:4], quotes_per_day=2)
            used = min(model.num_timesteps for model in bundle.users.values())
        response = evaluate_user_responses(bundle, validation_factory, validation_seeds,
                                           quotes_per_day=4, seed=args.seed + 7919)
        save_response(output / "pretraining_validation", response)
        pretrain_passed = all(stats["regret_yuan_per_task"] <= 5.0
                             and stats["user_ir_rate"] >= .99
                             for stats in response["per_user"].values())
        print(json.dumps({"pretraining_check": {"steps_per_user": used,
                         "passed": pretrain_passed, "summary": response["summary"],
                         "per_user": response["per_user"]}}), flush=True)
        if pretrain_passed:
            break
        if args.pretrained_users or used >= args.pretrain_max_steps:
            break

    if not pretrain_passed:
        (output / "summary.json").write_text(json.dumps({
            "status": "user_comparison_needs_more_training", "seed": args.seed,
            "pretrain_steps_per_user": used,
            "validation": {"summary": response["summary"], "per_user": response["per_user"]},
        }, indent=2), encoding="utf-8")
        raise RuntimeError("User contract comparison has not reached the validation target.")

    # Preserve the learned user weights and optimizers for joint learning.
    for model in bundle.users.values():
        model.gamma = args.joint_user_gamma
        model.rollout_buffer.gamma = args.joint_user_gamma
    bundle.platform_codec = catalogue
    bundle.training = True
    bundle.sample_opponents = True
    bundle.opponent_sampling_probability = args.opponent_sampling_probability
    bundle.price_exploration_probability = .4
    bundle.exploration_rng = np.random.default_rng(args.seed + 507)
    prototype = factory(0)
    envs = {actor: ActorEnv(factory, bundle, actor, train_seeds, prototype)
            for actor in ["platform", 0, 1, 2]}
    joint_log, validation_progress = [], []
    stage_steps = {str(actor): 0 for actor in envs}
    checkpoint = output / "checkpoints"
    checkpoint.mkdir(exist_ok=True)

    def before_phase(cycle, actor, policies):
        policies.training = True
        policies.sample_opponents = True
        policies.opponent_sampling_probability = args.opponent_sampling_probability
        policies.price_exploration_probability = .4 - .2 * cycle / max(1, args.cycles - 1)

    def phase(record, models):
        record["elapsed_total_seconds"] = time.monotonic() - started
        stage_steps[record["actor"]] += record["actual_actor_steps"]
        record["joint_actor_steps"] = stage_steps[record["actor"]]
        record["quote_exploration_probability"] = bundle.price_exploration_probability
        record["opponent_sampling_probability"] = bundle.opponent_sampling_probability
        record["user_gamma"] = args.joint_user_gamma
        joint_log.append(record)
        write_csv(output / "joint_training.csv", joint_log)
        actor = "platform" if record["actor"] == "platform" else int(record["actor"])
        label = "platform" if actor == "platform" else f"user_{actor}"
        models[actor].save(checkpoint / label)
        print(json.dumps({"joint": record}), flush=True)
        cycle = record["cycle"] + 1
        if actor == 2 and (cycle % args.eval_every_cycles == 0 or cycle == args.cycles):
            rows = evaluate_bundle(bundle, validation_factory, validation_seeds)
            row = dict(cycle=cycle, elapsed_seconds=time.monotonic() - started, **means(rows))
            row["cooperation_success_rate"] = float(np.mean([
                r["delta_system_cost_yuan"] < -1e-6 and r["user_ir_rate"] >= 1 - 1e-9
                and r["platform_participates"] for r in rows]))
            selection = evaluate_user_responses(
                bundle, validation_factory, validation_seeds[:4], quotes_per_day=2,
                seed=args.seed + 7919)
            row["diverse_quote_regret_yuan_per_task"] = selection["summary"]["regret_yuan_per_task"]
            row["diverse_quote_user_ir_rate"] = selection["summary"]["user_ir_rate"]
            validation_progress.append(row)
            write_csv(output / "validation_progress.csv", validation_progress)
            directory = output / "snapshots" / f"cycle_{cycle:04d}"
            directory.mkdir(parents=True, exist_ok=True)
            for key, model in models.items():
                label = "platform" if key == "platform" else f"user_{key}"
                model.save(directory / label)
            write_csv(directory / "validation.csv", rows)
            print(json.dumps({"validation": row}), flush=True)

    models, _ = alternating_train(
        bundle, envs, cycles=args.cycles, platform_days=args.platform_days,
        user_steps=args.user_steps, seed=args.seed, on_phase=phase,
        initial_models=dict(bundle.users), before_phase=before_phase,
        ppo_kwargs=dict(n_steps=128, batch_size=128, n_epochs=6, ent_coef=.02))
    bundle.training = False
    test_factory = cached_episode_factory(config, args.eval_days)
    rows = evaluate_bundle(bundle, test_factory, test_seeds)
    write_csv(output / "test.csv", rows)
    selection = evaluate_user_responses(bundle, test_factory, test_seeds,
                                       quotes_per_day=4, seed=args.seed + 7919)
    save_response(output / "test_user_responses", selection)
    result = {
        "status": "completed", "seed": args.seed,
        "elapsed_seconds": time.monotonic() - started,
        "config": asdict(config), "requested": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "training_scenario_seeds": train_seeds, "validation_scenario_seeds": validation_seeds,
        "test_scenario_seeds": test_seeds,
        "pretraining_passed": pretrain_passed, "pretrain_steps_per_user": used,
        "joint_actor_steps": stage_steps, "test_mean": means(rows),
        "test_diverse_quote_response": selection["summary"],
        "test_diverse_quote_per_user": selection["per_user"],
        "platform_actions": catalogue.n_actions, "platform_labels": catalogue.labels,
        "training_opponents": "parameters frozen, epsilon mixture of greedy and sampled actions",
        "opponent_sampling_probability": args.opponent_sampling_probability,
        "joint_user_gamma": args.joint_user_gamma,
        "joint_user_quote_coverage": "0.4 to 0.2 random diverse menu probability",
        "evaluation": "all four learned policies deterministic, no injected menus",
        "user_policy": "learned shared option scorer, own utility reward",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for env in envs.values():
        env.close()
    print(json.dumps({"completed": result}), flush=True)


if __name__ == "__main__":
    main()
