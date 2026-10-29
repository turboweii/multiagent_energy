#!/usr/bin/env python3
"""Select a frozen old-institution policy under the shared physical carbon cap."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from energy_net.coordination.capped_market import capped_episode_factory, legacy_capped_config
from energy_net.coordination.curriculum import DiverseQuoteBundle
from energy_net.coordination.flex_market import FlexPlatformEnv
from energy_net.coordination.flex_policy import FlexTariffActorCriticPolicy
from energy_net.coordination.flex_tariffs import FlexTariffObserver
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue
from energy_net.coordination.legacy_reference import FixedLegacyMenu
from energy_net.coordination.learning import ActorEnv, OptimizerStepCounter, PolicyBundle, evaluate_bundle
from train_coordination import write_csv


def means(rows):
    return {key: float(np.mean([row[key] for row in rows]))
            for key, value in rows[0].items()
            if isinstance(value, (int, float, bool, np.number))}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def adapt_legacy_users(output, source, factory, catalogue, steps, seed):
    users = {u: PPO.load(source / f"user_{u}.zip", device="cpu") for u in range(3)}
    bundle = DiverseQuoteBundle(catalogue, seed=seed + 301, users=users)
    records = []
    # Construct and seed all envs before updates, preserving a reproducible
    # independent owner RNG while opponents stay frozen during each phase.
    envs = {u: ActorEnv(factory, bundle, u, range(64), factory(0)) for u in range(3)}
    for u, env in envs.items():
        gym.Env.reset(env, seed=seed + u + 1)
    for u, env in envs.items():
        model = users[u]
        model.gamma = 0.; model.rollout_buffer.gamma = 0.
        model.set_env(env, force_reset=True)
        start = model.num_timesteps
        counter = OptimizerStepCounter(model)
        try:
            model.learn(total_timesteps=steps, reset_num_timesteps=False)
        finally:
            counter.close()
        records.append(dict(user=u, initial_steps=start,
            actual_added_steps=model.num_timesteps - start,
            cumulative_steps=model.num_timesteps, optimizer_steps=counter.count))
        print(json.dumps({"legacy_user_adaptation": records[-1]}), flush=True)
    directory = output / "users"; directory.mkdir(exist_ok=True)
    for u, model in users.items():
        model.save(directory / f"user_{u}")
    write_csv(output / "user_adaptation.csv", records)
    return users


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--user-steps", type=int, default=4096)
    parser.add_argument("--platform-steps", type=int, default=32768)
    parser.add_argument("--eval-interval", type=int, default=4096)
    args = parser.parse_args()
    if min(args.user_steps, args.platform_steps, args.eval_interval) < 1:
        parser.error("Training budgets must be positive")
    torch.set_num_threads(1)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    config = legacy_capped_config()
    train_factory = capped_episode_factory(config)
    validation_factory = capped_episode_factory(config, maxsize=16)
    catalogue = FixedGpuTariffCatalogue(config)
    users = adapt_legacy_users(output, args.users.resolve(), train_factory, catalogue, args.user_steps, args.seed)
    observer = FlexTariffObserver(catalogue)
    feature_mean, feature_scale = observer.fit_normalization(train_factory, range(64))
    bundle = PolicyBundle(users=users, platform_codec=catalogue, platform_observer=observer, training=True)
    env = FlexPlatformEnv(train_factory, bundle, "platform", range(64), train_factory(0))
    gym.Env.reset(env, seed=args.seed)
    model = PPO(FlexTariffActorCriticPolicy, env, seed=args.seed,
        n_steps=128, batch_size=128, n_epochs=6, learning_rate=3e-4,
        gamma=0., ent_coef=.02, policy_kwargs=dict(feature_mean=feature_mean, feature_scale=feature_scale),
        device="cpu", verbose=0)
    bundle.platform = model
    best_dir = output / "best_ppo"; best_dir.mkdir(exist_ok=True)
    validation, training, best_return, selected_steps = [], [], -np.inf, 0

    def validate():
        nonlocal best_return, selected_steps
        rows = evaluate_bundle(bundle, validation_factory, range(1000, 1016))
        stats = dict(steps=model.num_timesteps, **means(rows))
        validation.append(stats); write_csv(output / "platform_validation.csv", validation)
        eligible = all(row["completion_rate"] >= 1-1e-9 and row["user_ir_rate"] >= 1-1e-9 for row in rows)
        if eligible and stats["delta_platform_profit_yuan"] >= best_return - 1e-6:
            best_return = stats["delta_platform_profit_yuan"]
            selected_steps = model.num_timesteps
            model.save(best_dir / "platform")
            write_csv(best_dir / "validation_days.csv", rows)
        print(json.dumps({"legacy_platform_validation": {key: stats[key] for key in (
            "steps", "delta_platform_profit_yuan", "delta_system_cost_yuan", "grid_emissions_reduction_t")}}), flush=True)

    validate()
    counter = OptimizerStepCounter(model)
    try:
        while model.num_timesteps < args.platform_steps:
            fraction = model.num_timesteps / args.platform_steps
            observer.training_max_discount = .005 if fraction < .25 else .01 if fraction < .5 else .03
            bundle.training = True
            start, updates = model.num_timesteps, counter.count
            model.set_env(env, force_reset=True)
            model.learn(total_timesteps=min(args.eval_interval, args.platform_steps - model.num_timesteps), reset_num_timesteps=False)
            record = dict(actual_steps=model.num_timesteps-start, cumulative_steps=model.num_timesteps,
                          optimizer_steps=counter.count-updates,
                          maximum_discount=observer.training_max_discount)
            record.update({key:float(value) for key,value in model.logger.name_to_value.items()
                           if key.startswith("train/") and isinstance(value,(int,float,np.number))})
            training.append(record); write_csv(output / "platform_training.csv", training)
            validate()
    finally:
        counter.close()
    model.save(output / "last_platform")
    observer.training_max_discount = .03

    # Freeze the same users for every candidate and every validation comparison.
    frozen_users = {u: PPO.load(output / "users" / f"user_{u}.zip", device="cpu") for u in range(3)}
    all_rows = []
    for case_seed in range(64):
        episode = train_factory(case_seed)
        for index in range(catalogue.n_actions):
            episode.reset(catalogue.decode(episode, index))
            metrics = episode.play(frozen_users).metrics
            all_rows.append(dict(menu_index=index, menu_label=catalogue.labels[index], **metrics))
        print(json.dumps({"legacy_fixed_search_training_day":case_seed}), flush=True)
    fixed_means = []
    for index in range(catalogue.n_actions):
        rows = [row for row in all_rows if row["menu_index"] == index]
        valid = all(row["completion_rate"] >= 1-1e-9 and row["user_ir_rate"] >= 1-1e-9 for row in rows)
        fixed_means.append(dict(means(rows), menu_index=index, menu_label=catalogue.labels[index],
                               all_work_and_user_ir=valid))
    eligible = [row for row in fixed_means if row["all_work_and_user_ir"]]
    best_constant = max(eligible, key=lambda row:(row["delta_platform_profit_yuan"],-row["menu_index"]))
    write_csv(output / "fixed_training_days.csv", all_rows)
    write_csv(output / "fixed_training_means.csv", fixed_means)

    candidate_reports, candidate_days = [], []
    candidates = (("ppo",None,PPO.load(best_dir / "platform.zip",device="cpu")),
                  ("constant",int(best_constant["menu_index"]),FixedLegacyMenu(int(best_constant["menu_index"]))),
                  ("zero",0,FixedLegacyMenu(0)))
    for label, index, policy in candidates:
        candidate_bundle = PolicyBundle(platform=policy, users=frozen_users, platform_codec=catalogue,
            platform_observer=observer if label=="ppo" else None, training=False)
        rows = evaluate_bundle(candidate_bundle, validation_factory, range(1000,1016))
        stats = dict(candidate=label, menu_index=index, **means(rows))
        candidate_reports.append(stats)
        candidate_days.extend(dict(candidate=label, **row) for row in rows)
    write_csv(output / "candidate_validation.csv", candidate_reports)
    write_csv(output / "candidate_validation_days.csv", candidate_days)
    # Only these train/validation candidates are admissible outside references;
    # no per-test-day best menu or test outcome enters selection.
    selected = max(candidate_reports, key=lambda row:(row["delta_platform_profit_yuan"],
                   row["candidate"]=="constant",row["candidate"]=="zero"))
    destination = output / "selected"; destination.mkdir(exist_ok=True)
    for u in range(3):
        shutil.copy2(output / "users" / f"user_{u}.zip", destination / f"user_{u}.zip")
    kind = "ppo" if selected["candidate"]=="ppo" else "constant"
    if kind=="ppo":
        shutil.copy2(best_dir / "platform.zip", destination / "platform.zip")
    selection = dict(policy_kind=kind, menu_index=None if kind=="ppo" else int(selected["menu_index"]),
        platform_model="selected/platform.zip" if kind=="ppo" else None,
        selected_candidate=selected["candidate"], config=asdict(config), algorithm_seed=args.seed,
        selected_ppo_steps=selected_steps, trained_platform_steps=model.num_timesteps,
        validation_mean=selected, candidate_validation=candidate_reports,
        strongest_train_constant=best_constant,
        training_scenario_seeds=list(range(64)),validation_scenario_seeds=list(range(1000,1016)),
        test_scenario_seeds_used=[],
        selection_rule="train-select fixed menu; validation-select one best PPO checkpoint, fixed menu or zero; freeze before any test day",
        user_sha256={str(u):hashlib.sha256((destination/f'user_{u}.zip').read_bytes()).hexdigest() for u in range(3)})
    write_json(output / "selection.json",selection)
    summary = dict(status="completed",seed=args.seed,elapsed_seconds=time.monotonic()-started,
        requested={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        config=asdict(config),selection=selection,platform_training=training,
        old_users="frozen after own-utility adaptation; no joint updates during old-platform selection")
    write_json(output / "summary.json",summary)
    print(json.dumps({"legacy_selection_complete":{"policy_kind":kind,"menu_index":selection['menu_index'],
        "validation_cash_gain_yuan":selected['delta_platform_profit_yuan'],
        "elapsed_seconds":summary['elapsed_seconds']}}),flush=True)


if __name__=="__main__":
    main()
