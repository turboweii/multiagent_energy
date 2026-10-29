#!/usr/bin/env python3
"""One-seed, four-agent GPU-hour electricity/carbon information experiment."""
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

from energy_net.coordination.gpu_scenario import GpuMarketConfig
from energy_net.coordination.gpu_market import gpu_episode_factory, GpuPlatformEnv
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue, GpuTariffObserver
from energy_net.coordination.learning import ActorEnv, PolicyBundle, alternating_train, evaluate_bundle
from energy_net.coordination.policies import ContractActorCriticPolicy
from energy_net.coordination.platform_policy import CandidateTariffActorCriticPolicy
from energy_net.coordination.curriculum import DiverseQuoteBundle
from train_coordination import write_csv


def means(rows):
    return {key: float(np.mean([row[key] for row in rows]))
            for key, value in rows[0].items()
            if isinstance(value, (int, float, bool, np.number))}


def response_check(users, factory, catalogue, seeds):
    rows = []
    for seed in seeds:
        for quote in (0, 1, 10, 24, 40, 60, 76):
            episode = factory(seed)
            episode.set_platform_action(catalogue.decode(episode, quote))
            while not episode.done:
                job = episode.jobs[episode.position]
                best = episode.best_response_action()
                best_cost = episode.option_bill(job, job.options[best]) + job.options[best].user_loss_yuan
                action = users[job.user_id].predict(episode.current_user_observation(), deterministic=True)[0]
                _, _, info = episode.step_user(action)
                rows.append(dict(seed=seed, quote=quote, user=job.user_id,
                                 regret_per_gpu_hour=max(0, info["bill_yuan"] + info["loss_yuan"] - best_cost) / job.work_cu,
                                 minimizes=info["bill_yuan"] + info["loss_yuan"] <= best_cost + 1e-5,
                                 individual_rational=info["individual_rational"]))
    return means(rows), rows


def train_users(output, config, steps, seed):
    factory = gpu_episode_factory(config)
    catalogue = FixedGpuTariffCatalogue(config)
    bundle = DiverseQuoteBundle(catalogue, seed=seed + 301)
    envs = {u: ActorEnv(factory, bundle, u, range(64), factory(0)) for u in range(3)}
    for u, env in envs.items():
        gym.Env.reset(env, seed=seed + u + 1)
        bundle.users[u] = PPO(ContractActorCriticPolicy, env, seed=seed + u + 1,
            n_steps=128, batch_size=128, n_epochs=6, learning_rate=3e-4,
            gamma=0, ent_coef=.01, policy_kwargs=dict(economic_only=True, participation_constraint=True),
            device="cpu", verbose=0)
    records = []
    for round_id in range((steps + 2047) // 2048):
        for u, env in envs.items():
            model = bundle.users[u]
            requested = min(2048, steps - model.num_timesteps)
            if requested <= 0:
                continue
            model.set_env(env, force_reset=True)
            model.learn(total_timesteps=requested, reset_num_timesteps=False)
            records.append(dict(round=round_id, user=u, actual_steps=model.num_timesteps))
        stats, rows = response_check(bundle.users, factory, catalogue, range(1000, 1004))
        records[-1].update(stats)
        write_csv(output / "user_training.csv", records)
        print(json.dumps({"user_validation": stats, "steps_per_user": bundle.users[0].num_timesteps}), flush=True)
    directory = output / "initial_users"; directory.mkdir(exist_ok=True)
    for u, model in bundle.users.items():
        model.save(directory / f"user_{u}")
    stats, rows = response_check(bundle.users, factory, catalogue, range(1000, 1016))
    write_csv(output / "user_validation.csv", rows)
    return stats


def train_platform(output, config, initial_users, carbon_aware, seed, steps, joint_cycles):
    output.mkdir(parents=True, exist_ok=True)
    factory = gpu_episode_factory(config)
    validation_factory = gpu_episode_factory(config, maxsize=16)
    test_factory = gpu_episode_factory(config, maxsize=32)
    catalogue = FixedGpuTariffCatalogue(config)
    observer = GpuTariffObserver(catalogue, carbon_aware=carbon_aware)
    feature_mean, feature_scale = observer.fit_normalization(factory, range(64))
    users = {u: PPO.load(initial_users / f"user_{u}.zip", device="cpu") for u in range(3)}
    bundle = PolicyBundle(users=users, platform_codec=catalogue, platform_observer=observer, training=True)
    env = GpuPlatformEnv(factory, bundle, "platform", range(64), factory(0))
    gym.Env.reset(env, seed=seed)
    model = PPO(CandidateTariffActorCriticPolicy, env, seed=seed,
        n_steps=128, batch_size=128, n_epochs=6, learning_rate=3e-4,
        gamma=0, ent_coef=.02, policy_kwargs=dict(feature_mean=feature_mean, feature_scale=feature_scale),
        device="cpu", verbose=0)
    bundle.platform = model
    validation_rows, best_value, selected = [], -np.inf, 0
    best_dir = output / "best_fixed"; best_dir.mkdir(exist_ok=True)

    def validate(stage):
        nonlocal best_value, selected
        rows = evaluate_bundle(bundle, validation_factory, range(1000, 1016))
        stats = dict(stage=stage, steps=model.num_timesteps, **means(rows))
        validation_rows.append(stats)
        write_csv(output / "validation.csv", validation_rows)
        if stats["delta_platform_objective_yuan"] >= best_value - 1e-6:
            best_value = stats["delta_platform_objective_yuan"]
            selected = model.num_timesteps
            model.save(best_dir / "platform")
        print(json.dumps({"platform_validation": {k: stats[k] for k in (
            "stage", "steps", "delta_platform_objective_yuan", "delta_platform_profit_yuan",
            "delta_system_cost_yuan", "grid_emissions_reduction_t", "dc_footprint_reduction_t")}}), flush=True)

    validate("initial")
    while model.num_timesteps < steps:
        observer.training_max_discount = .005 if model.num_timesteps < steps / 4 else .01 if model.num_timesteps < steps / 2 else .03
        bundle.training = True
        model.set_env(env, force_reset=True)
        model.learn(total_timesteps=min(2048, steps - model.num_timesteps), reset_num_timesteps=False)
        validate("fixed_users")
    bundle.platform = PPO.load(best_dir / "platform.zip", device="cpu")
    observer.training_max_discount = .03
    frozen = evaluate_bundle(bundle, test_factory, range(30000, 30032))
    write_csv(output / "fixed_users_test.csv", frozen)
    # Primary information comparison uses these same, frozen user weights.
    # Subsequent independent updates test the four-agent interaction separately.
    bundle.training = True; bundle.sample_opponents = True
    bundle.opponent_sampling_probability = .10; bundle.price_exploration_probability = .20
    bundle.exploration_rng = np.random.default_rng(seed + 701)
    envs = {actor: (GpuPlatformEnv if actor == "platform" else ActorEnv)(
        factory, bundle, actor, range(64), factory(0)) for actor in ("platform", 0, 1, 2)}
    records = []
    def phase(record, models):
        records.append(record)
        write_csv(output / "joint_training.csv", records)
    models, _ = alternating_train(bundle, envs, cycles=joint_cycles, platform_days=256,
        user_steps=512, seed=seed, initial_models={"platform": bundle.platform, **users}, on_phase=phase)
    checkpoint = output / "checkpoints"; checkpoint.mkdir(exist_ok=True)
    for actor, policy in models.items():
        policy.save(checkpoint / ("platform" if actor == "platform" else f"user_{actor}"))
    final = evaluate_bundle(bundle, test_factory, range(30000, 30032))
    write_csv(output / "joint_test.csv", final)
    action_rows = []
    for case_seed in range(30000, 30032):
        ep = test_factory(case_seed)
        action = int(bundle.platform.predict(bundle.platform_observation(ep), deterministic=True)[0])
        action_rows.append(dict(seed=case_seed, action=action, label=catalogue.labels[action]))
    write_csv(output / "test_actions.csv", action_rows)
    summary = dict(config=asdict(config), seed=seed, carbon_aware=carbon_aware,
        selected_platform_steps=selected, trained_platform_steps=steps,
        fixed_users_test=means(frozen), joint_test=means(final), joint_records=records,
        objective="cash profit minus voluntary internal carbon valuation; valuation is not cash expense",
        train_scenario_seeds=list(range(64)), validation_scenario_seeds=list(range(1000, 1016)),
        test_scenario_seeds=list(range(30000, 30032)))
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--user-steps", type=int, default=16384)
    parser.add_argument("--platform-steps", type=int, default=16384)
    parser.add_argument("--joint-cycles", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); config = GpuMarketConfig()
    user_stats = train_users(output, config, args.user_steps, args.seed)
    results = {}
    for label, eta, aware in (("cash_only", 0., False), ("electricity_only", 100., False), ("electricity_carbon", 100., True)):
        results[label] = train_platform(output / label, replace(config, internal_carbon_value_yuan_per_t=eta),
            output / "initial_users", aware, args.seed, args.platform_steps, args.joint_cycles)
    report = dict(seed=args.seed, elapsed_seconds=time.monotonic() - started,
                  user_validation=user_stats, experiments=results)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"completed": {"elapsed_seconds": report["elapsed_seconds"], "user_validation": user_stats}}), flush=True)


if __name__ == "__main__":
    main()
