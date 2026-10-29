#!/usr/bin/env python3
"""One-seed tests of contextual prices, with unchanged net-surplus settlement."""
from __future__ import annotations
import argparse
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from energy_net.coordination.amendment_market import AmendmentConfig, AmendmentPlatformEnv, amendment_episode_factory
from energy_net.coordination.amendment_features import AmendmentTariffCatalogue, AmendmentTariffObserver
from energy_net.coordination.cached_grid import MemoizedGridClearing
from energy_net.coordination.capped_market import make_capped_scenario
from energy_net.coordination.curriculum import DiverseQuoteBundle
from energy_net.coordination.flex_market import FlexMarketEpisode
from energy_net.coordination.flex_policy import FlexTariffActorCriticPolicy
from energy_net.coordination.learning import ActorEnv, PolicyBundle, OptimizerStepCounter, alternating_train
from energy_net.coordination.legacy_reference import load_legacy_bundle
from energy_net.coordination.outside_bookings import outside_booking_factory
from train_amendments import make_factories, user_check

TRAIN = tuple(range(64))
VALIDATION = tuple(range(1000, 1016))
TEST = tuple(range(60000, 60032))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def means(rows):
    return pd.DataFrame(rows).mean(numeric_only=True).to_dict()


def catalogue_for(name, config):
    if name == 'old':
        return AmendmentTariffCatalogue(config)
    from energy_net.coordination.dynamic_tariffs import ExpandedAmendmentCatalogue
    return ExpandedAmendmentCatalogue(config)


def config_for(family):
    if family == 'original':
        return AmendmentConfig()
    from energy_net.coordination.varied_scenario import VariedMarketConfig
    return VariedMarketConfig()


def observer_for(name, codec):
    if name == 'roster':
        from energy_net.coordination.roster_features import RosterAmendmentObserver
        return RosterAmendmentObserver(codec)
    return AmendmentTariffObserver(codec)


def factory_for(family, legacy_path, maxsize=64):
    if family == 'original':
        return make_factories(legacy_path, maxsize=maxsize)
    from energy_net.coordination.varied_scenario import make_varied_scenario
    config = config_for(family)
    legacy = load_legacy_bundle(legacy_path)
    legacy_config = replace(config, conditional_rebates=legacy.legacy_config.conditional_rebates,
                            sharing_enabled=False, carbon_eligibility=False,
                            internal_carbon_value_yuan_per_t=0.)
    @lru_cache(maxsize=maxsize)
    def scenario(seed):
        return make_varied_scenario(int(seed), config)
    @lru_cache(maxsize=maxsize)
    def old_grid(seed):
        return MemoizedGridClearing(scenario(seed).grid_spec, maxsize=2048)
    def old_factory(seed):
        return FlexMarketEpisode(replace(scenario(seed), config=legacy_config), grid=old_grid(seed))
    outside = outside_booking_factory(legacy, old_factory, maxsize=maxsize,
                                      policy_label='frozen_accepted_legacy_contracts')
    return amendment_episode_factory(scenario, outside, config=config, maxsize=maxsize,
                                     grid_cache_size=2048)


def load_users(path):
    return {u: PPO.load(path / f'user_{u}.zip', device='cpu') for u in range(3)}


def play(episode, codec, users, menu):
    episode.reset(codec.decode(episode, menu))
    while not episode.done:
        user = episode.current_user_id
        action = users[user].predict(episode.current_user_observation(), deterministic=True)[0]
        episode.step_user(action)
    return episode.outcome().metrics


def collect(factory, codec, observer, users, seeds, *, model=None, constant=None,
            observation_factory=None):
    rows = []
    for seed in seeds:
        episode = factory(int(seed))
        if constant is not None:
            menu = int(constant)
        else:
            state_episode = episode if observation_factory is None else observation_factory(int(seed))
            menu = int(model.predict(observer(state_episode, training=False), deterministic=True)[0])
        metrics = play(episode, codec, users, menu)
        matrix = episode.discounts
        rows.append(dict(metrics, scenario_seed=int(seed), menu=menu, menu_label=codec.labels[menu],
                         quote_spatial_std=float(matrix.std(axis=0).mean()),
                         quote_temporal_std=float(matrix.std(axis=1).mean()),
                         quote_matrix=matrix.tolist()))
    return rows


def adapt(output, legacy, source_users, family, steps, seed):
    output.mkdir(parents=True, exist_ok=True)
    factory = factory_for(family, legacy)
    codec = catalogue_for('expanded', factory.config)
    users = load_users(source_users)
    bundle = DiverseQuoteBundle(codec, seed=seed+301, users=users)
    records=[]
    for u in range(3):
        env=ActorEnv(factory,bundle,u,TRAIN,factory(0));gym.Env.reset(env,seed=seed+u+1)
        model=users[u];before=model.num_timesteps
        model.set_env(env,force_reset=True);model.gamma=0.;model.rollout_buffer.gamma=0.
        counter=OptimizerStepCounter(model)
        try:model.learn(total_timesteps=steps,reset_num_timesteps=False)
        finally:counter.close()
        records.append(dict(user=u,added_steps=model.num_timesteps-before,
                            cumulative_steps=model.num_timesteps,optimizer_steps=counter.count))
        model.save(output/f'user_{u}')
    check, rows=user_check(users,factory,codec,VALIDATION)
    csv(output/'training.csv',records);csv(output/'validation.csv',rows)
    write_json(output/'summary.json',dict(seed=seed,family=family,validation=check,source=str(source_users)))
    print(json.dumps({'users_ready':str(output),'validation':check}),flush=True)


def benchmark(output, legacy, users_path, family, catalogue):
    output.mkdir(parents=True,exist_ok=True)
    factory=factory_for(family,legacy);codec=catalogue_for(catalogue,factory.config)
    users=load_users(users_path);rows=[]
    for seed in TRAIN:
        episode=factory(seed)
        for menu in range(codec.n_actions):
            rows.append(dict(scenario_seed=seed,menu=menu,menu_label=codec.labels[menu],
                             **play(episode,codec,users,menu)))
        print(json.dumps({'benchmark':family,'catalogue':catalogue,'day':seed}),flush=True)
    frame=pd.DataFrame(rows)
    grouped=frame.groupby('menu').mean(numeric_only=True)
    selected=int(grouped.platform_gain_vs_outside_yuan.idxmax())
    best=frame.loc[frame.groupby('scenario_seed').platform_gain_vs_outside_yuan.idxmax()]
    report=dict(family=family,catalogue=catalogue,menus=codec.n_actions,menu=selected,
        label=codec.labels[selected],selection='training mean real additional platform cash; no test selection',
        constant_profit_yuan=float(grouped.loc[selected,'platform_gain_vs_outside_yuan']),
        hindsight_profit_yuan=float(best.platform_gain_vs_outside_yuan.mean()),
        hindsight_menu_counts=best.menu.value_counts().to_dict(),
        dynamic_headroom_yuan=float(best.platform_gain_vs_outside_yuan.mean()-grouped.loc[selected,'platform_gain_vs_outside_yuan']))
    csv(output/'training_outcomes.csv',rows);csv(output/'menu_means.csv',grouped.reset_index())
    csv(output/'hindsight_train.csv',best.to_dict(orient='records'));write_json(output/'selection.json',report)
    print(json.dumps(report),flush=True)


def train(output, legacy, users_path, family, catalogue, mode, steps, cycles, seed, observer_name='basic'):
    from energy_net.coordination.frozen_response import FrozenResponsePlatformEnv
    output.mkdir(parents=True,exist_ok=True)
    factory=factory_for(family,legacy);vf=factory_for(family,legacy,maxsize=16)
    codec=catalogue_for(catalogue,factory.config);observer=observer_for(observer_name,codec)
    mean,scale=observer.fit_normalization(factory,TRAIN)
    users=load_users(users_path)
    bundle=PolicyBundle(users=users,platform_codec=codec,platform_observer=observer,training=True)
    env=FrozenResponsePlatformEnv(factory,bundle,'platform',TRAIN,factory(0))
    gym.Env.reset(env,seed=seed)
    platform=PPO(FlexTariffActorCriticPolicy,env,seed=seed,n_steps=128,batch_size=128,n_epochs=6,
        learning_rate=3e-4,gamma=0.,ent_coef=.02 if mode=='legacy' else .12,
        policy_kwargs=dict(feature_mean=mean,feature_scale=scale),device='cpu',verbose=0)
    bundle.platform=platform;best=-np.inf;best_step=0;progress=[];training=[]
    destination=output/'best_fixed';destination.mkdir(exist_ok=True)
    def validate():
        nonlocal best,best_step
        rows=collect(vf,codec,observer,users,VALIDATION,model=platform)
        stats=dict(steps=platform.num_timesteps,unique_menus=len({r['menu'] for r in rows}),**means(rows))
        progress.append(stats);csv(output/'validation.csv',progress)
        if stats['platform_gain_vs_outside_yuan']>best+1e-6:
            best=stats['platform_gain_vs_outside_yuan'];best_step=platform.num_timesteps
            platform.save(destination/'platform');csv(destination/'validation_days.csv',rows)
        print(json.dumps({'family':family,'catalogue':catalogue,'mode':mode,'step':platform.num_timesteps,
            'profit':stats['platform_gain_vs_outside_yuan'],'saved':stats['incremental_system_cost_saving_vs_outside_yuan'],
            'menus':stats['unique_menus']}),flush=True)
    validate();counter=OptimizerStepCounter(platform)
    try:
        while platform.num_timesteps<steps:
            if mode=='legacy':
                observer.training_max_discount=.005 if platform.num_timesteps<steps/4 else .01 if platform.num_timesteps<steps/2 else .03
            else:
                observer.training_max_discount=.03
                platform.ent_coef=.12 if platform.num_timesteps<steps/2 else .04 if platform.num_timesteps<3*steps/4 else .02
            before,updates=platform.num_timesteps,counter.count
            platform.set_env(env,force_reset=True)
            platform.learn(total_timesteps=min(4096,steps-before),reset_num_timesteps=False)
            training.append(dict(actual_steps=platform.num_timesteps-before,steps=platform.num_timesteps,
                optimizer_steps=counter.count-updates,entropy_coefficient=platform.ent_coef,
                **{k:float(v) for k,v in platform.logger.name_to_value.items()
                   if k.startswith('train/') and isinstance(v,(float,int,np.number))}))
            csv(output/'platform_training.csv',training);validate()
    finally:counter.close()
    observer.training_max_discount=.03;bundle.platform=PPO.load(destination/'platform.zip',device='cpu')
    joint_dir=output/'best_joint';joint_dir.mkdir(exist_ok=True)
    def save():
        bundle.platform.save(joint_dir/'platform')
        for u,p in bundle.users.items():p.save(joint_dir/f'user_{u}')
    save();joint=[];jp=[];joint_best=best;best_cycle=0
    if cycles:
        bundle.training=True;bundle.sample_opponents=True;bundle.opponent_sampling_probability=.1
        bundle.price_exploration_probability=.2;bundle.exploration_rng=np.random.default_rng(seed+701)
        envs={a:(AmendmentPlatformEnv if a=='platform' else ActorEnv)(factory,bundle,a,TRAIN,factory(0))
              for a in ('platform',0,1,2)}
        def phase(record,models):
            nonlocal joint_best,best_cycle
            joint.append(record);csv(output/'joint_training.csv',joint)
            if record['actor']=='2':
                rows=collect(vf,codec,observer,bundle.users,VALIDATION,model=bundle.platform)
                stats=dict(cycle=record['cycle']+1,**means(rows));jp.append(stats);csv(output/'joint_validation.csv',jp)
                if stats['platform_gain_vs_outside_yuan']>joint_best+1e-6:
                    joint_best=stats['platform_gain_vs_outside_yuan'];best_cycle=stats['cycle'];save()
        alternating_train(bundle,envs,cycles=cycles,platform_days=256,user_steps=512,seed=seed,
                          initial_models={'platform':bundle.platform,**users},on_phase=phase)
    summary=dict(seed=seed,family=family,catalogue=catalogue,mode=mode,observer=observer_name,menus=codec.n_actions,
        steps=steps,selected_step=best_step,joint_cycles=cycles,selected_joint_cycle=best_cycle,
        users_path=str(users_path),legacy_path=str(legacy),training_records=training,joint_records=joint,
        training_scenarios=TRAIN,validation_scenarios=VALIDATION,test_scenarios=TEST,
        selection='validation mean real platform cash; no preference for price variation')
    write_json(output/'summary.json',summary)
    print(json.dumps({'finished':str(output),'selected_step':best_step,'joint_cycle':best_cycle}),flush=True)


def evaluate(output, legacy, users_path, family, catalogue, model_dir, benchmark_dir, joint_benchmark_dir=None):
    report=json.loads((model_dir/'summary.json').read_text())
    fixed=json.loads((benchmark_dir/'selection.json').read_text())
    factory=factory_for(family,legacy,maxsize=32);codec=catalogue_for(catalogue,factory.config)
    observer=observer_for(report.get('observer','basic'),codec);users=load_users(users_path)
    model=PPO.load(model_dir/'best_fixed/platform.zip',device='cpu')
    rows=[]
    for label,p,constant in [('outside',None,0),('strong_constant',None,fixed['menu']),('ppo',model,None)]:
        group=collect(factory,codec,observer,users,TEST,model=p,constant=constant)
        rows.extend(dict(r,policy=label) for r in group)
    other=factory_for(family,legacy,maxsize=32)
    shifted={seed:TEST[(i+1)%len(TEST)] for i,seed in enumerate(TEST)}
    group=collect(factory,codec,observer,users,TEST,model=model,
                  observation_factory=lambda seed:other(shifted[seed]))
    rows.extend(dict(r,policy='ppo_shuffled_state') for r in group)
    jusers=load_users(model_dir/'best_joint')
    jmodel=PPO.load(model_dir/'best_joint/platform.zip',device='cpu')
    group=collect(factory,codec,observer,jusers,TEST,model=jmodel)
    rows.extend(dict(r,policy='joint') for r in group)
    jfixed=(json.loads((joint_benchmark_dir/'selection.json').read_text())
            if joint_benchmark_dir is not None else fixed)
    group=collect(factory,codec,observer,jusers,TEST,constant=jfixed['menu'])
    rows.extend(dict(r,policy='joint_strong_constant') for r in group)
    group=collect(factory,codec,observer,jusers,TEST,model=jmodel,
                  observation_factory=lambda seed:other(shifted[seed]))
    rows.extend(dict(r,policy='joint_shuffled_state') for r in group)
    csv(output/'test_days.csv',rows)
    frame=pd.DataFrame(rows);summaries=[]
    for label,g in frame.groupby('policy',sort=False):
        summaries.append(dict(policy=label,unique_menus=g.menu.nunique(),**g.mean(numeric_only=True).to_dict()))
    csv(output/'test_mean.csv',summaries)
    write_json(output/'test_report.json',dict(training=report,benchmark=fixed,joint_benchmark=jfixed,results=summaries,
        held_out_scenarios=TEST,shuffled_state='cyclic one-day shift of public observations; actual job policies and clearing stay on actual day'))
    print(json.dumps({'evaluation':str(output),'results':[{k:r[k] for k in
        ('policy','unique_menus','platform_gain_vs_outside_yuan','incremental_system_cost_saving_vs_outside_yuan')}
        for r in summaries]}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('users','bench','train','evaluate'),required=True)
    p.add_argument('--family',choices=('original','varied'),default='varied')
    p.add_argument('--catalogue',choices=('old','expanded'),default='expanded')
    p.add_argument('--mode',choices=('legacy','balanced'),default='balanced')
    p.add_argument('--observer',choices=('basic','roster'),default='basic')
    p.add_argument('--legacy',type=Path,required=True);p.add_argument('--users',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model-dir',type=Path);p.add_argument('--benchmark-dir',type=Path)
    p.add_argument('--joint-benchmark-dir',type=Path)
    p.add_argument('--steps',type=int,default=32768);p.add_argument('--joint-cycles',type=int,default=0)
    p.add_argument('--seed',type=int,default=0);args=p.parse_args();torch.set_num_threads(1)
    if args.stage=='users':adapt(args.output,args.legacy,args.users,args.family,args.steps,args.seed)
    elif args.stage=='bench':benchmark(args.output,args.legacy,args.users,args.family,args.catalogue)
    elif args.stage=='train':train(args.output,args.legacy,args.users,args.family,args.catalogue,args.mode,args.steps,args.joint_cycles,args.seed,args.observer)
    else:evaluate(args.output,args.legacy,args.users,args.family,args.catalogue,args.model_dir,args.benchmark_dir,args.joint_benchmark_dir)


if __name__=='__main__':main()
