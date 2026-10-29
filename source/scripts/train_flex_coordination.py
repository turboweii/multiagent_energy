#!/usr/bin/env python3
"""Learn conditional GPU contracts and budget-balanced resource sharing."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from energy_net.coordination.curriculum import DiverseQuoteBundle
from energy_net.coordination.flex_market import FlexMarketConfig, FlexPlatformEnv, flex_episode_factory
from energy_net.coordination.flex_tariffs import FlexTariffObserver
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue
from energy_net.coordination.learning import ActorEnv, PolicyBundle, OptimizerStepCounter, alternating_train, evaluate_bundle
from energy_net.coordination.platform_policy import CandidateTariffActorCriticPolicy
from train_coordination import write_csv
from train_gpu_coordination import means, response_check


EXPERIMENTS = {
    "published_no_share": dict(conditional_rebates=False, sharing_enabled=False, carbon_eligibility=False),
    "conditional_no_share": dict(conditional_rebates=True, sharing_enabled=False, carbon_eligibility=False),
    "conditional_share": dict(conditional_rebates=True, sharing_enabled=True, carbon_eligibility=False),
    "conditional_green_share": dict(conditional_rebates=True, sharing_enabled=True, carbon_eligibility=True),
}


def adapt_users(output, source, steps, seed):
    """Warm-start the same own-utility users, never supply optimal-choice labels."""
    config = FlexMarketConfig()
    factory = flex_episode_factory(config)
    catalogue = FixedGpuTariffCatalogue(config)
    users = {u: PPO.load(source / f"user_{u}.zip", device="cpu") for u in range(3)}
    bundle = DiverseQuoteBundle(catalogue, seed=seed + 301, users=users)
    envs = {u: ActorEnv(factory, bundle, u, range(64), factory(0)) for u in range(3)}
    records = []
    for u, env in envs.items():
        gym.Env.reset(env, seed=seed + u + 1)
        model = users[u]
        model.gamma = 0.; model.rollout_buffer.gamma = 0.
        model.set_env(env, force_reset=True)
        before = model.num_timesteps
        counter = OptimizerStepCounter(model)
        try:
            model.learn(total_timesteps=steps, reset_num_timesteps=False)
        finally:
            counter.close()
        records.append(dict(user=u, initial_steps=before, added_steps=model.num_timesteps - before,
                            actual_optimizer_steps=counter.count, cumulative_steps=model.num_timesteps))
    directory = output / "initial_users"; directory.mkdir(exist_ok=True)
    for u, model in users.items():
        model.save(directory / f"user_{u}")
    write_csv(output / "user_adaptation.csv", records)
    summaries = {}
    for conditional in (False, True):
        test_factory = flex_episode_factory(replace(config, conditional_rebates=conditional), maxsize=16)
        stats, rows = response_check(users, test_factory, catalogue, range(1000, 1016))
        summaries[str(conditional)] = stats
        write_csv(output / f"user_validation_conditional_{conditional}.csv", rows)
    print(json.dumps({"user_validation": summaries}), flush=True)
    return summaries


def run_arm(output, config, initial_users, steps, cycles, seed, *,
            resume_platform=None, prior_platform_steps_spent=0):
    output.mkdir(parents=True, exist_ok=True)
    factory = flex_episode_factory(config)
    validation_factory = flex_episode_factory(config, maxsize=16)
    test_factory = flex_episode_factory(config, maxsize=32)
    catalogue = FixedGpuTariffCatalogue(config)
    observer = FlexTariffObserver(catalogue)
    feature_mean, feature_scale = observer.fit_normalization(factory, range(64))
    users = {u: PPO.load(initial_users / f"user_{u}.zip", device="cpu") for u in range(3)}
    bundle = PolicyBundle(users=users, platform_codec=catalogue, platform_observer=observer, training=True)
    env = FlexPlatformEnv(factory, bundle, "platform", range(64), factory(0))
    gym.Env.reset(env, seed=seed)
    if resume_platform is None:
        model = PPO(CandidateTariffActorCriticPolicy, env, seed=seed,
            n_steps=128, batch_size=128, n_epochs=6, learning_rate=3e-4,
            gamma=0., ent_coef=.02, policy_kwargs=dict(feature_mean=feature_mean, feature_scale=feature_scale),
            device="cpu", verbose=0)
    else:
        from energy_net.coordination.flex_policy import warm_start_with_action_bias
        source = PPO.load(resume_platform, device="cpu")
        model = warm_start_with_action_bias(source, env)
    starting_steps = model.num_timesteps
    target_steps = starting_steps + steps
    bundle.platform = model
    best_dir = output / "best_fixed"; best_dir.mkdir(exist_ok=True)
    validations, training_records, best_profit, selected = [], [], -np.inf, 0

    def fixed_validation(stage):
        nonlocal best_profit, selected
        rows = evaluate_bundle(bundle, validation_factory, range(1000, 1016))
        stats = dict(stage=stage, steps=model.num_timesteps, **means(rows))
        validations.append(stats); write_csv(output / "validation.csv", validations)
        if stats["delta_platform_profit_yuan"] >= best_profit - 1e-6:
            best_profit = stats["delta_platform_profit_yuan"]
            selected = model.num_timesteps
            model.save(best_dir / "platform")
        print(json.dumps({"arm": output.name, "validation": {key: stats[key] for key in (
            "steps", "delta_platform_profit_yuan", "delta_system_cost_yuan",
            "grid_emissions_reduction_t", "buyer_post_settlement_benefit_yuan", "platform_participates")}}), flush=True)

    fixed_validation("initial")
    counter = OptimizerStepCounter(model)
    try:
        while model.num_timesteps < target_steps:
            added = model.num_timesteps - starting_steps
            observer.training_max_discount = (.03 if resume_platform is not None else
                .005 if added < steps / 4 else .01 if added < steps / 2 else .03)
            bundle.training = True
            before, updates = model.num_timesteps, counter.count
            model.set_env(env, force_reset=True)
            model.learn(total_timesteps=min(2048, target_steps - model.num_timesteps), reset_num_timesteps=False)
            record = dict(actual_steps=model.num_timesteps - before, cumulative_steps=model.num_timesteps,
                          actual_optimizer_steps=counter.count - updates,
                          maximum_discount=observer.training_max_discount)
            record.update({k: float(v) for k, v in model.logger.name_to_value.items()
                           if k.startswith("train/") and isinstance(v, (int, float, np.number))})
            training_records.append(record); write_csv(output / "platform_training.csv", training_records)
            fixed_validation("fixed_users")
    finally:
        counter.close()
    observer.training_max_discount = .03
    bundle.platform = PPO.load(best_dir / "platform.zip", device="cpu")
    fixed = evaluate_bundle(bundle, test_factory, range(40000, 40032))
    write_csv(output / "fixed_users_test.csv", fixed)

    # Joint checkpoints contain all four actors from the same validation cycle.
    joint_dir = output / "best_joint"; joint_dir.mkdir(exist_ok=True)
    def save_bundle(directory):
        bundle.platform.save(directory / "platform")
        for u, policy in bundle.users.items():
            policy.save(directory / f"user_{u}")
    save_bundle(joint_dir)
    initial_rows = evaluate_bundle(bundle, validation_factory, range(1000, 1016))
    joint_best = means(initial_rows)["delta_platform_profit_yuan"]
    selected_cycle = 0
    bundle.training = True; bundle.sample_opponents = True
    bundle.opponent_sampling_probability = .10; bundle.price_exploration_probability = .20
    bundle.exploration_rng = np.random.default_rng(seed + 701)
    envs = {actor: (FlexPlatformEnv if actor == "platform" else ActorEnv)(
        factory, bundle, actor, range(64), factory(0)) for actor in ("platform", 0, 1, 2)}
    records, joint_validation = [], []
    def phase(record, models):
        nonlocal joint_best, selected_cycle
        records.append(record); write_csv(output / "joint_training.csv", records)
        if record["actor"] == "2":
            rows = evaluate_bundle(bundle, validation_factory, range(1000, 1016))
            stats = dict(cycle=record["cycle"] + 1, **means(rows))
            joint_validation.append(stats); write_csv(output / "joint_validation.csv", joint_validation)
            if stats["delta_platform_profit_yuan"] >= joint_best - 1e-6:
                joint_best = stats["delta_platform_profit_yuan"]
                selected_cycle = stats["cycle"]
                save_bundle(joint_dir)
    models, _ = alternating_train(bundle, envs, cycles=cycles, platform_days=256,
        user_steps=512, seed=seed, initial_models={"platform": bundle.platform, **users}, on_phase=phase)
    last = output / "last_joint"; last.mkdir(exist_ok=True)
    for actor, policy in models.items():
        policy.save(last / ("platform" if actor == "platform" else f"user_{actor}"))
    bundle.platform = PPO.load(joint_dir / "platform.zip", device="cpu")
    bundle.users = {u: PPO.load(joint_dir / f"user_{u}.zip", device="cpu") for u in range(3)}
    checkpoint = output / "checkpoints"; checkpoint.mkdir(exist_ok=True)
    save_bundle(checkpoint)
    final = evaluate_bundle(bundle, test_factory, range(40000, 40032))
    write_csv(output / "joint_test.csv", final)
    summary = dict(config=asdict(config), seed=seed, selected_platform_steps=selected,
        trained_platform_steps=prior_platform_steps_spent + steps,
        continuation_initial_policy_steps=starting_steps,
        continuation_added_steps=steps if resume_platform is not None else 0,
        source_platform=None if resume_platform is None else str(resume_platform),
        selected_joint_cycle=selected_cycle,
        platform_training_records=training_records, joint_records=records,
        fixed_users_test=means(fixed), joint_test=means(final),
        reward="actual settled platform cash-profit change / fixed 1000 yuan",
        selection="maximum validation-day mean settled cash-profit gain; no test selection",
        train_scenario_seeds=list(range(64)), validation_scenario_seeds=list(range(1000, 1016)),
        test_scenario_seeds=list(range(40000, 40032)))
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"finished_arm": output.name, "selected_steps": selected,
                      "selected_joint_cycle": selected_cycle}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--user-steps", type=int, default=4096)
    parser.add_argument("--platform-steps", type=int, default=16384)
    parser.add_argument("--joint-cycles", type=int, default=4)
    args = parser.parse_args()
    if min(args.user_steps, args.platform_steps, args.joint_cycles) < 1:
        parser.error("training budgets and joint cycles must be positive")
    torch.set_num_threads(1)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    user_validation = adapt_users(output, args.users.resolve(), args.user_steps, args.seed)
    results = {}
    for name, fields in EXPERIMENTS.items():
        results[name] = run_arm(output / name, FlexMarketConfig(**fields), output / "initial_users",
                                args.platform_steps, args.joint_cycles, args.seed)
    summary = dict(seed=args.seed, elapsed_seconds=time.monotonic() - started,
        requested={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        user_validation=user_validation, experiments=results)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"completed": {"elapsed_seconds": summary["elapsed_seconds"]}}), flush=True)


if __name__ == "__main__":
    main()
