#!/usr/bin/env python3
"""Learn conditional platform tariffs, then verify the four-agent interaction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO

from energy_net.coordination.learning import (
    ActorEnv, PolicyBundle, OptimizerStepCounter, alternating_train,
    cached_episode_factory, evaluate_bundle,
)
from energy_net.coordination.platform_learning import (
    CandidateTariffObserver, PlatformProfitEnv, apply_private_participation,
    platform_policy_statistics,
)
from energy_net.coordination.platform_policy import CandidateTariffActorCriticPolicy
from energy_net.coordination.scenario import MarketConfig
from energy_net.coordination.tariffs import TariffCatalogue
from train_coordination import write_csv


def numeric_means(rows):
    return {k: float(np.mean([row[k] for row in rows])) for k, value in rows[0].items()
            if isinstance(value, (int, float, bool, np.number))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--users", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16384)
    parser.add_argument("--eval-interval", type=int, default=2048)
    parser.add_argument("--joint-cycles", type=int, default=8)
    parser.add_argument("--joint-platform-days", type=int, default=256)
    parser.add_argument("--joint-user-steps", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.steps, args.eval_interval, args.joint_platform_days, args.joint_user_steps) < 1:
        parser.error("Step budgets must be positive.")
    torch.set_num_threads(1)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    config, codec = MarketConfig(), TariffCatalogue()
    factory = cached_episode_factory(config, 64)
    validation_factory = cached_episode_factory(config, 16)
    test_factory = cached_episode_factory(config, 32)
    train_seeds, validation_seeds, test_seeds = list(range(64)), list(range(9000, 9016)), list(range(10000, 10032))
    users = {u: apply_private_participation(PPO.load(args.users / f"user_{u}.zip", device="cpu"))
             for u in range(3)}
    observer = CandidateTariffObserver(codec)
    bundle = PolicyBundle(users=users, platform_codec=codec, platform_observer=observer)
    started = time.monotonic()
    mean, scale = observer.fit_normalization(factory, train_seeds)
    np.savez(output / "normalization.npz", mean=mean, scale=scale)
    # The zero-offer original contract must remain an actual outside option.
    zero_rows = []
    for seed in validation_seeds:
        ep = validation_factory(seed)
        ep.set_platform_action(codec.decode(ep, 0))
        zero_rows.append(ep.play(users).metrics)
    write_csv(output / "zero_quote_validation.csv", zero_rows)
    if any(r["user_ir_rate"] < 1 - 1e-9 or r["migrated_jobs"]
           or abs(r["delta_platform_profit_yuan"]) > 1e-6 for r in zero_rows):
        raise RuntimeError("The zero quote did not reproduce the original outside option.")
    for u, model in users.items():
        model.save(output / f"fixed_user_{u}")
    bundle.training = True
    observer.training_max_discount = .05
    env = PlatformProfitEnv(factory, bundle, "platform", train_seeds, factory(0))
    gym.Env.reset(env, seed=args.seed)
    platform = PPO(CandidateTariffActorCriticPolicy, env, seed=args.seed,
                   n_steps=256, batch_size=256, n_epochs=8, learning_rate=3e-4,
                   gamma=0.0, ent_coef=.02, target_kl=.03, device="cpu", verbose=0,
                   policy_kwargs=dict(feature_mean=mean, feature_scale=scale))
    bundle.platform = platform
    counter = OptimizerStepCounter(platform)
    records, progress = [], []
    best_profit = -float("inf")
    selected_step = 0
    best_dir = output / "best_platform_validation"; best_dir.mkdir(exist_ok=True)

    def validate(label):
        nonlocal best_profit, selected_step
        rows = evaluate_bundle(bundle, validation_factory, validation_seeds)
        policy_stats, choices = platform_policy_statistics(bundle, validation_factory, validation_seeds)
        report = dict(stage=label, platform_steps=platform.num_timesteps,
                      elapsed_seconds=time.monotonic() - started,
                      **numeric_means(rows), **policy_stats)
        report["cooperation_rate"] = float(np.mean([
            row["delta_system_cost_yuan"] < -1e-6 and row["user_ir_rate"] >= 1 - 1e-9
            and row["platform_participates"] for row in rows]))
        progress.append(report); write_csv(output / "validation_progress.csv", progress)
        snap = output / "snapshots" / f"steps_{platform.num_timesteps:06d}"; snap.mkdir(parents=True, exist_ok=True)
        platform.save(snap / "platform")
        write_csv(snap / "validation.csv", rows); write_csv(snap / "choices.csv", choices)
        eligible = all(row["user_ir_rate"] >= 1 - 1e-9 for row in rows)
        # Equal greedy validation return: keep the more trained checkpoint,
        # rather than restarting joint learning from a nearly uniform early policy.
        if eligible and report["delta_platform_profit_yuan"] >= best_profit - 1e-9:
            best_profit = report["delta_platform_profit_yuan"]
            selected_step = platform.num_timesteps
            platform.save(best_dir / "platform")
        print(json.dumps({"validation": report}), flush=True)

    validate("initial")
    try:
        while platform.num_timesteps < args.steps:
            fraction = platform.num_timesteps / args.steps
            observer.training_max_discount = .05 if fraction < .25 else .10 if fraction < .5 else .30
            requested = min(args.eval_interval, args.steps - platform.num_timesteps)
            before, updates = platform.num_timesteps, counter.count
            bundle.training = True
            platform.set_env(env, force_reset=True)
            platform.learn(total_timesteps=requested, reset_num_timesteps=False, progress_bar=False)
            record = dict(stage="fixed_users", actual_steps=platform.num_timesteps - before,
                          cumulative_steps=platform.num_timesteps,
                          optimizer_steps=counter.count - updates,
                          maximum_training_discount=observer.training_max_discount)
            record.update({k: float(v) for k, v in platform.logger.name_to_value.items()
                           if k.startswith("train/") and isinstance(v, (float, int, np.number))})
            records.append(record); write_csv(output / "platform_training.csv", records)
            print(json.dumps({"training": record}), flush=True)
            validate("fixed_users")
    finally:
        counter.close()
    platform.save(output / "last_fixed_platform")
    bundle.platform = PPO.load(best_dir / "platform.zip", device="cpu")
    observer.training_max_discount = .30
    fixed = evaluate_bundle(bundle, test_factory, test_seeds)
    write_csv(output / "fixed_users_test.csv", fixed)
    # Every scenario action remains chosen by the learned policy. No profit
    # lookup or rollback is used at validation, test, or during joint learning.
    joint_records, joint_progress = [], []
    for model in users.values():
        model.gamma = 0.0; model.rollout_buffer.gamma = 0.0
    bundle.training = True; bundle.sample_opponents = True
    bundle.opponent_sampling_probability = .10
    bundle.price_exploration_probability = .20
    bundle.exploration_rng = np.random.default_rng(args.seed + 701)
    envs = {actor: (PlatformProfitEnv if actor == "platform" else ActorEnv)(
        factory, bundle, actor, train_seeds, factory(0)) for actor in ("platform", 0, 1, 2)}
    checkpoint = output / "checkpoints"; checkpoint.mkdir(exist_ok=True)

    def phase(record, models):
        joint_records.append(record); write_csv(output / "joint_training.csv", joint_records)
        key = "platform" if record["actor"] == "platform" else int(record["actor"])
        models[key].save(checkpoint / ("platform" if key == "platform" else f"user_{key}"))
        if key == 2:
            rows = evaluate_bundle(bundle, validation_factory, validation_seeds)
            policy_stats, choices = platform_policy_statistics(bundle, validation_factory, validation_seeds)
            report = dict(cycle=record["cycle"] + 1, **numeric_means(rows), **policy_stats)
            joint_progress.append(report); write_csv(output / "joint_validation.csv", joint_progress)
            print(json.dumps({"joint_validation": report}), flush=True)

    if args.joint_cycles:
        models, _ = alternating_train(bundle, envs, cycles=args.joint_cycles,
            platform_days=args.joint_platform_days, user_steps=args.joint_user_steps,
            seed=args.seed, initial_models={"platform": bundle.platform, **users}, on_phase=phase)
    else:
        for key, model in {"platform": bundle.platform, **users}.items():
            model.save(checkpoint / ("platform" if key == "platform" else f"user_{key}"))
    final = evaluate_bundle(bundle, test_factory, test_seeds)
    write_csv(output / "test.csv", final)
    policy_stats, choices = platform_policy_statistics(bundle, test_factory, test_seeds)
    write_csv(output / "test_choices.csv", choices)
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    summary = dict(status="completed", seed=args.seed, requested=arguments,
        elapsed_seconds=time.monotonic() - started, selected_fixed_platform_step=selected_step,
        fixed_users_test_mean=numeric_means(fixed), final_test_mean=numeric_means(final),
        final_policy_statistics=policy_stats, platform_training_records=records,
        joint_training_records=joint_records, training_scenario_seeds=train_seeds,
        validation_scenario_seeds=validation_seeds, test_scenario_seeds=test_seeds,
        user_choice_constraint="private bill plus inconvenience no worse than original contract",
        platform_features="public reference clearing and proposal attributes; no preference/response/profit labels",
        selection="highest mean platform profit on validation; test not used for selection",
        reward="realized platform profit change / fixed 1000 yuan")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"completed": summary}), flush=True)


if __name__ == "__main__":
    main()
