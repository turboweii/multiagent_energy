#!/usr/bin/env python3
"""One-seed renegotiation with the same carbon cap and accepted old bookings."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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

from energy_net.coordination.amendment_market import AmendmentConfig, AmendmentPlatformEnv, amendment_episode_factory
from energy_net.coordination.amendment_features import AmendmentTariffCatalogue, AmendmentTariffObserver
from energy_net.coordination.capped_market import make_capped_scenario, capped_episode_factory
from energy_net.coordination.curriculum import DiverseQuoteBundle
from energy_net.coordination.flex_policy import FlexTariffActorCriticPolicy
from energy_net.coordination.learning import ActorEnv, PolicyBundle, OptimizerStepCounter, alternating_train, evaluate_bundle
from energy_net.coordination.legacy_reference import load_legacy_bundle
from energy_net.coordination.outside_bookings import outside_booking_factory
from train_coordination import write_csv
from train_gpu_coordination import means


def make_factories(legacy_path, config=None, maxsize=64):
    config = config or AmendmentConfig()
    legacy = load_legacy_bundle(legacy_path)
    outside = outside_booking_factory(legacy, capped_episode_factory(legacy.legacy_config), maxsize=maxsize,
                                      policy_label="train_validation_selected_legacy")
    def scenario(seed):
        return make_capped_scenario(seed, config)
    return amendment_episode_factory(scenario, outside, config=config, maxsize=maxsize)


def user_check(users, factory, catalogue, seeds):
    rows = []
    for seed in seeds:
        for quote in (0, 1, 2, 11, 25, 41, 61, 77):
            episode = factory(seed)
            episode.set_platform_action(catalogue.decode(episode, quote))
            while not episode.done:
                job = episode.jobs[episode.position]
                best = episode.best_response_action()
                best_cost = episode.option_bill(job, job.options[best]) + job.options[best].user_loss_yuan
                action = users[job.user_id].predict(episode.current_user_observation(), deterministic=True)[0]
                _, _, info = episode.step_user(action)
                actual_cost = info["bill_yuan"] + info["loss_yuan"]
                rows.append(dict(seed=seed, quote=quote, user=job.user_id,
                    regret_per_gpu_hour=max(0., actual_cost - best_cost) / job.work_cu,
                    individual_rational=info["individual_rational"],
                    minimizes=actual_cost <= best_cost + 1e-5))
    return means(rows), rows


def adapt_users(output, legacy_path, steps, seed):
    factory = make_factories(legacy_path)
    catalogue = AmendmentTariffCatalogue(AmendmentConfig())
    # All stages clone the frozen old participants; the booking factory retains
    # separate model objects and cannot be changed by this new learning round.
    users = {u: PPO.load(legacy_path / "selected" / f"user_{u}.zip", device="cpu") for u in range(3)}
    bundle = DiverseQuoteBundle(catalogue, seed=seed + 301, users=users)
    records = []
    for u in range(3):
        env = ActorEnv(factory, bundle, u, range(64), factory(0))
        gym.Env.reset(env, seed=seed + u + 1)
        model = users[u]; before = model.num_timesteps
        model.gamma = 0.; model.rollout_buffer.gamma = 0.
        model.set_env(env, force_reset=True)
        counter = OptimizerStepCounter(model)
        try:
            model.learn(total_timesteps=steps, reset_num_timesteps=False)
        finally:
            counter.close()
        records.append(dict(user=u, actual_added_steps=model.num_timesteps - before,
                            cumulative_steps=model.num_timesteps, optimizer_steps=counter.count))
    directory = output / "initial_users"; directory.mkdir(exist_ok=True)
    for u, policy in users.items(): policy.save(directory / f"user_{u}")
    write_csv(output / "user_adaptation.csv", records)
    check, rows = user_check(users, factory, catalogue, range(1000, 1016))
    write_csv(output / "user_validation.csv", rows)
    print(json.dumps({"user_validation": check}), flush=True)
    return check


def train_arm(output, legacy_path, initial_users, order, steps, cycles, seed):
    output.mkdir(parents=True, exist_ok=True)
    config = AmendmentConfig(settlement_order=order)
    factory = make_factories(legacy_path, config)
    validation_factory = make_factories(legacy_path, config, maxsize=16)
    catalogue = AmendmentTariffCatalogue(config)
    observer = AmendmentTariffObserver(catalogue)
    mean, scale = observer.fit_normalization(factory, range(64))
    users = {u: PPO.load(initial_users / f"user_{u}.zip", device="cpu") for u in range(3)}
    bundle = PolicyBundle(users=users, platform_codec=catalogue, platform_observer=observer, training=True)
    env = AmendmentPlatformEnv(factory, bundle, "platform", range(64), factory(0))
    gym.Env.reset(env, seed=seed)
    platform = PPO(FlexTariffActorCriticPolicy, env, seed=seed, n_steps=128, batch_size=128,
        n_epochs=6, learning_rate=3e-4, gamma=0., ent_coef=.02,
        policy_kwargs=dict(feature_mean=mean, feature_scale=scale), device="cpu", verbose=0)
    bundle.platform = platform
    best_dir = output / "best_fixed"; best_dir.mkdir(exist_ok=True)
    progress, records, best_profit, selected_step = [], [], -np.inf, 0
    def validate():
        nonlocal best_profit, selected_step
        rows = evaluate_bundle(bundle, validation_factory, range(1000, 1016))
        stats = dict(steps=platform.num_timesteps, **means(rows))
        progress.append(stats); write_csv(output / "validation.csv", progress)
        if stats["platform_gain_vs_outside_yuan"] >= best_profit - 1e-6:
            best_profit = stats["platform_gain_vs_outside_yuan"]
            selected_step = platform.num_timesteps; platform.save(best_dir / "platform")
        print(json.dumps({"order": order, "validation": {k: stats[k] for k in (
            "steps", "platform_gain_vs_outside_yuan", "incremental_system_cost_saving_vs_outside_yuan",
            "accepted_amendments", "buyer_post_settlement_benefit_yuan", "platform_floor_shortfall_yuan")}}), flush=True)
    validate()
    counter = OptimizerStepCounter(platform)
    try:
        while platform.num_timesteps < steps:
            observer.training_max_discount = .005 if platform.num_timesteps < steps / 4 else .01 if platform.num_timesteps < steps / 2 else .03
            bundle.training = True
            before, updates = platform.num_timesteps, counter.count
            platform.set_env(env, force_reset=True)
            platform.learn(total_timesteps=min(4096, steps - platform.num_timesteps), reset_num_timesteps=False)
            records.append(dict(actual_steps=platform.num_timesteps - before, cumulative_steps=platform.num_timesteps,
                optimizer_steps=counter.count - updates, maximum_extra_discount=observer.training_max_discount,
                **{k:float(v) for k,v in platform.logger.name_to_value.items()
                   if k.startswith("train/") and isinstance(v,(float,int,np.number))}))
            write_csv(output / "platform_training.csv", records); validate()
    finally:
        counter.close()
    observer.training_max_discount = .03
    bundle.platform = PPO.load(best_dir / "platform.zip", device="cpu")
    joint_dir = output / "best_joint"; joint_dir.mkdir(exist_ok=True)
    def save(directory):
        bundle.platform.save(directory / "platform")
        for u, policy in bundle.users.items(): policy.save(directory / f"user_{u}")
    save(joint_dir)
    joint_best, selected_cycle = best_profit, 0
    bundle.training = True; bundle.sample_opponents = True
    bundle.opponent_sampling_probability = .1; bundle.price_exploration_probability = .2
    bundle.exploration_rng = np.random.default_rng(seed + 701)
    envs = {a:(AmendmentPlatformEnv if a=="platform" else ActorEnv)(factory,bundle,a,range(64),factory(0))
            for a in ("platform",0,1,2)}
    joint, joint_progress = [], []
    def phase(record, models):
        nonlocal joint_best, selected_cycle
        joint.append(record); write_csv(output / "joint_training.csv", joint)
        if record["actor"] == "2":
            stats = dict(cycle=record["cycle"]+1, **means(evaluate_bundle(bundle,validation_factory,range(1000,1016))))
            joint_progress.append(stats); write_csv(output / "joint_validation.csv",joint_progress)
            if stats["platform_gain_vs_outside_yuan"] >= joint_best - 1e-6:
                joint_best=stats["platform_gain_vs_outside_yuan"]; selected_cycle=stats["cycle"]; save(joint_dir)
    models,_=alternating_train(bundle,envs,cycles=cycles,platform_days=256,user_steps=512,
        seed=seed,initial_models={"platform":bundle.platform,**users},on_phase=phase)
    last=output/"last_joint";last.mkdir(exist_ok=True)
    for a,p in models.items():p.save(last/("platform" if a=="platform" else f"user_{a}"))
    checkpoint=output/"checkpoints";checkpoint.mkdir(exist_ok=True)
    for p in joint_dir.glob("*.zip"):shutil.copy2(p,checkpoint/p.name)
    result=dict(seed=seed,config=asdict(config),trained_platform_steps=steps,selected_platform_steps=selected_step,
        selected_joint_cycle=selected_cycle,legacy_path=str(legacy_path),platform_training_records=records,joint_records=joint,
        training_scenario_seeds=list(range(64)),validation_scenario_seeds=list(range(1000,1016)),
        test_scenario_seeds=list(range(50000,50032)),
        selection="validation mean additional cash profit over accepted legacy contracts; test unused",
        participation="pre-binding admission protects accepted legacy bookings; not a learned profit guarantee")
    (output/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"finished_order":order,"selected_steps":selected_step,"selected_cycle":selected_cycle}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=0);parser.add_argument("--user-steps",type=int,default=4096)
    parser.add_argument("--platform-steps",type=int,default=32768);parser.add_argument("--joint-cycles",type=int,default=4)
    args=parser.parse_args();torch.set_num_threads(1)
    if min(args.user_steps,args.platform_steps,args.joint_cycles)<1:parser.error("training budgets must be positive")
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True);legacy=args.legacy.resolve();started=time.monotonic()
    user_validation=adapt_users(output,legacy,args.user_steps,args.seed)
    reports={order:train_arm(output/order,legacy,output/"initial_users",order,args.platform_steps,args.joint_cycles,args.seed)
             for order in ("direct","gross","net")}
    summary=dict(seed=args.seed,elapsed_seconds=time.monotonic()-started,legacy=str(legacy),
        requested={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},user_validation=user_validation,experiments=reports)
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"completed":summary["elapsed_seconds"]}),flush=True)


if __name__=="__main__":main()
