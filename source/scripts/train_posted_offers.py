#!/usr/bin/env python3
"""Train four sequential, cash-reward PPOs for one binding offer per task."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from energy_net.coordination.learning import OptimizerStepCounter, alternating_train
from energy_net.coordination.platform_policy import CandidateTariffActorCriticPolicy
from energy_net.coordination.posted_offer_features import PostedOfferCatalogue, PostedOfferObserver
from energy_net.coordination.posted_offer_learning import (PostedOfferEnv, PostedPolicyBundle,
    posted_episode_factory, preference_cases)
from energy_net.coordination.posted_offer_policy import warm_load_binary_user

TRAIN=preference_cases(range(256),8,100000)
VALIDATION=preference_cases(range(1000,1032),4,200000)
TEST=preference_cases(range(70000,70032),4,300000)
BENCH_TRAIN=preference_cases(range(64),1,100000)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def csv(path,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(path,index=False,encoding='utf-8-sig')


def load_users(path):return {u:PPO.load(path/f'user_{u}.zip',device='cpu') for u in range(3)}


def bundle_for(users,history=True):
    codec=PostedOfferCatalogue()
    return PostedPolicyBundle(users=users,platform_codec=codec,
        platform_observer=PostedOfferObserver(codec,enable_history=history))


def rollouts(factory,bundle,cases,*,controller=None,model=None,record_offers=False):
    rows,offers=[],[];previous=bundle.training;bundle.training=False
    try:
        for case in cases:
            episode=factory(case)
            while not episode.done:
                if controller is not None:action=controller.choose(episode)
                else:action=int(model.predict(bundle.platform_observation(episode),deterministic=True)[0])
                proposal=episode.preview_offer(action)
                episode.set_offer(action)
                _,_,info=episode.step_accept(bundle.user_action(episode))
                if record_offers:
                    offers.append(dict(case_id=case,**info['public_feedback'],
                        G=proposal['G'],R=proposal['R'],M=proposal['M'],
                        platform_gain_yuan=proposal['platform_gain_yuan'] if info['chosen_accept'] else 0.))
            metrics=episode.outcome().metrics
            rows.append(dict(case_id=case,**metrics))
    finally:bundle.training=previous
    return rows,offers


def mean(rows):return pd.DataFrame(rows).mean(numeric_only=True).to_dict()


def users_stage(output,legacy,source,steps,seed):
    output.mkdir(parents=True,exist_ok=True)
    factory=posted_episode_factory(legacy)
    bundle=bundle_for({});bundle.training=True;bundle.exploration_rng=np.random.default_rng(seed+301)
    envs={u:PostedOfferEnv(factory,bundle,u,TRAIN,factory(TRAIN[0])) for u in range(3)}
    for u in range(3):
        gym.Env.reset(envs[u],seed=seed+u+1)
        old=PPO.load(source/f'user_{u}.zip',device='cpu')
        bundle.users[u]=warm_load_binary_user(old,envs[u],seed=seed+u+1,
            n_steps=128,batch_size=128,n_epochs=6,learning_rate=3e-4,ent_coef=.02,
            gamma=1.,gae_lambda=1.)
    records=[]
    for u,model in bundle.users.items():
        counter=OptimizerStepCounter(model)
        try:model.learn(total_timesteps=steps,reset_num_timesteps=False)
        finally:counter.close()
        records.append(dict(user=u,actual_steps=model.num_timesteps,optimizer_steps=counter.count,
                            warm_start=model.warm_start_metadata))
        model.save(output/f'user_{u}')
        print(json.dumps({'user_ready':u,'steps':model.num_timesteps}),flush=True)
    dump(output/'summary.json',dict(seed=seed,training=records,source=str(source),
        objective='actual daily own utility, gamma=1, reward=yuan/1000',
        public_training_days=256,independent_private_draws_per_day=8))
    csv(output/'training.csv',[{k:v for k,v in r.items() if k!='warm_start'} for r in records])


def train_stage(output,legacy,users_path,steps,cycles,seed,history):
    output.mkdir(parents=True,exist_ok=True)
    factory=posted_episode_factory(legacy)
    vf=posted_episode_factory(legacy,public_cache_size=32,case_cache_size=128)
    users=load_users(users_path);bundle=bundle_for(users,history)
    observer=bundle.platform_observer
    # Cover every physical training day; each has eight private draws in the
    # rollout pool. No future responses or outcome labels enter normalization.
    mu,sigma=observer.fit_normalization(factory,TRAIN[::8])
    env=PostedOfferEnv(factory,bundle,'platform',TRAIN,factory(TRAIN[0]))
    gym.Env.reset(env,seed=seed)
    model=PPO(CandidateTariffActorCriticPolicy,env,seed=seed,n_steps=256,batch_size=256,
        n_epochs=6,learning_rate=3e-4,gamma=1.,gae_lambda=.95,ent_coef=.02,
        policy_kwargs=dict(feature_mean=mu,feature_scale=sigma,actor_hidden=(128,64),value_hidden=(128,64)),
        device='cpu',verbose=0)
    bundle.platform=model;bundle.training=True
    best=-np.inf;selected=0;progress=[];training=[]
    bestdir=output/'best_fixed';bestdir.mkdir(exist_ok=True)
    def validate():
        nonlocal best,selected
        rows,_=rollouts(vf,bundle,VALIDATION,model=model)
        stats=dict(steps=model.num_timesteps,**mean(rows));progress.append(stats)
        csv(output/'validation.csv',progress)
        value=stats['procurement_net_saving_yuan']
        if value>best+1e-6:
            best=value;selected=model.num_timesteps;model.save(bestdir/'platform')
            csv(bestdir/'validation_days.csv',rows)
        print(json.dumps({'history':history,'step':model.num_timesteps,
            'net_saving':value,'profit':stats['platform_gain_vs_outside_yuan'],
            'resource_saving':stats['flexibility_resource_saving_yuan'],
            'social_saving':stats['social_saving_vs_outside_yuan'],
            'accepted':stats['accepted_offers']}),flush=True)
    validate();counter=OptimizerStepCounter(model)
    try:
        while model.num_timesteps<steps:
            model.ent_coef=.02 if model.num_timesteps<steps/2 else .005
            before,updates=model.num_timesteps,counter.count
            model.set_env(env,force_reset=True)
            model.learn(total_timesteps=min(8192,steps-before),reset_num_timesteps=False)
            training.append(dict(actual_steps=model.num_timesteps-before,steps=model.num_timesteps,
                optimizer_steps=counter.count-updates,
                **{k:float(v) for k,v in model.logger.name_to_value.items()
                   if k.startswith('train/') and isinstance(v,(float,int,np.number))}))
            csv(output/'platform_training.csv',training)
            # These are only already occurred public responses, not a table
            # of unquoted hypothetical user actions.
            if env.feedback_records:
                csv(output/f'feedback_step_{model.num_timesteps}.csv',env.feedback_records)
                env.feedback_records.clear()
            validate()
    finally:counter.close()
    model.save(output/'last_fixed_platform')
    bundle.platform=PPO.load(bestdir/'platform.zip',device='cpu')
    jdir=output/'best_joint';jdir.mkdir(exist_ok=True)
    def save_joint():
        bundle.platform.save(jdir/'platform')
        for u,p in bundle.users.items():p.save(jdir/f'user_{u}')
    save_joint();jbest=best;selected_cycle=0;jrecords=[];jprogress=[]
    if cycles:
        bundle.training=True;bundle.sample_opponents=True;bundle.opponent_sampling_probability=.1
        bundle.price_exploration_probability=.1;bundle.exploration_rng=np.random.default_rng(seed+701)
        envs={a:PostedOfferEnv(factory,bundle,a,TRAIN,factory(TRAIN[0])) for a in ('platform',0,1,2)}
        def phase(record,models):
            nonlocal jbest,selected_cycle
            jrecords.append(record);csv(output/'joint_training.csv',jrecords)
            if record['actor']=='2':
                rows,_=rollouts(vf,bundle,VALIDATION,model=bundle.platform)
                stats=dict(cycle=record['cycle']+1,**mean(rows));jprogress.append(stats)
                csv(output/'joint_validation.csv',jprogress)
                if stats['procurement_net_saving_yuan']>jbest+1e-6:
                    jbest=stats['procurement_net_saving_yuan'];selected_cycle=stats['cycle'];save_joint()
        models,_=alternating_train(bundle,envs,cycles=cycles,platform_days=8192,user_steps=4096,
            seed=seed,initial_models={'platform':bundle.platform,**users},on_phase=phase)
        last=output/'last_joint';last.mkdir(exist_ok=True)
        for actor,p in models.items():p.save(last/('platform' if actor=='platform' else f'user_{actor}'))
    dump(output/'summary.json',dict(seed=seed,history=history,steps=steps,selected_step=selected,
        joint_cycles=cycles,selected_joint_cycle=selected_cycle,training_records=training,joint_records=jrecords,
        train_cases=TRAIN,validation_cases=VALIDATION,test_cases=TEST,
        config=asdict(factory.config),users_path=str(users_path),legacy_path=str(legacy),
        objective='actual procurement net saving M; platform cash=.8M; social saving separately G-dD',
        selection='validation mean M; price variation is not rewarded',
        gamma=1.,user_reward='actual utility_yuan/1000',platform_reward='actual cash_yuan/1000'))
    print(json.dumps({'completed':str(output),'selected_step':selected,'selected_joint_cycle':selected_cycle}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('users','train'),required=True)
    p.add_argument('--legacy',type=Path,required=True);p.add_argument('--users',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=131072)
    p.add_argument('--joint-cycles',type=int,default=1);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--no-history',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='users':users_stage(a.output,a.legacy,a.users,a.steps,a.seed)
    else:train_stage(a.output,a.legacy,a.users,a.steps,a.joint_cycles,a.seed,not a.no_history)


if __name__=='__main__':main()
