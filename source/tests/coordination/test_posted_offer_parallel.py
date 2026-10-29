"""Spawned deterministic selection is identical to serial on real TRAIN IDs."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import evaluate_posted_offers as evaluation

from energy_net.coordination.capped_market import legacy_capped_config
from energy_net.coordination.policies import ContractActorCriticPolicy
from energy_net.coordination.posted_offer_parallel import SpawnedTemplateEvaluator
from energy_net.coordination.posted_offer_policy import warm_load_binary_user
from energy_net.coordination.varied_scenario import VariedMarketConfig


class ShapeEnv(gym.Env):
    def __init__(self,actions):
        self.action_space=spaces.Discrete(actions)
        self.observation_space=spaces.Box(-np.inf,np.inf,(92,),dtype=np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        return np.zeros(92,dtype=np.float32),{}
    def step(self,action):
        return np.zeros(92,dtype=np.float32),0.,True,False,{}


@pytest.fixture(scope='module')
def saved_models(tmp_path_factory):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    path=tmp_path_factory.mktemp('spawned-template-selection')
    legacy,users=path/'legacy',path/'users'
    selected=legacy/'selected';selected.mkdir(parents=True);users.mkdir()
    source=PPO(ContractActorCriticPolicy,ShapeEnv(10),seed=0,n_steps=2,batch_size=2,n_epochs=1,
        policy_kwargs=dict(economic_only=True,participation_constraint=True),device='cpu')
    source.save(selected/'user_0')
    for owner in (1,2):shutil.copy2(selected/'user_0.zip',selected/f'user_{owner}.zip')
    (legacy/'selection.json').write_text(json.dumps(dict(policy_kind='constant',menu_index=0,
        config=asdict(legacy_capped_config()),selected_candidate='unit_zero')),encoding='utf-8')
    binary=warm_load_binary_user(source,ShapeEnv(2),seed=0)
    binary.save(users/'user_0')
    for owner in (1,2):shutil.copy2(users/'user_0.zip',users/f'user_{owner}.zip')
    (users/'summary.json').write_text('{}',encoding='utf-8')
    yield legacy,users
    torch.set_num_threads(previous)


def plain(value):
    if isinstance(value,dict):return {key:plain(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [plain(item) for item in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


@pytest.mark.parametrize('workers',[2,4])
def test_spawned_selection_preserves_every_row_mean_winner_budget_and_random_fit(saved_models,workers):
    legacy,users_path=saved_models
    config=VariedMarketConfig(jobs_per_user=1,batch_release_steps=(13,))
    factory=evaluation.posted_episode_factory(legacy,config=config)
    users=evaluation.load_users(users_path)
    cases=evaluation.preference_cases(range(1),2,110000)
    calibration=evaluation.preference_cases(range(1),2,120000)
    templates=evaluation.matrix_templates(config)
    settings=dict(raw_actions=(0,26),templates=[templates[index] for index in (0,1,178)],
                  rhos=(.3,.6),seed=0)
    serial_order=[]
    serial=evaluation.select_controls(factory,users,cases,calibration,**settings,
        on_family=lambda name,rows,means:serial_order.append((name,plain(rows),plain(means))))
    fresh_factory=evaluation.posted_episode_factory(legacy,config=config)
    parallel_order=[]
    with SpawnedTemplateEvaluator(legacy,users_path,config=config,workers=workers) as pool:
        assert pool._pool._mp_context.get_start_method()=='spawn'
        parallel=evaluation.select_controls(fresh_factory,users,cases,calibration,**settings,
            on_family=lambda name,rows,means:parallel_order.append((name,plain(rows),plain(means))),
            candidate_evaluator=pool)
        assert pool.worker_initializations and len(pool.worker_initializations)<=workers
        assert set(pool.worker_initializations.values())=={1}
    for index in range(5):assert plain(parallel[index])==plain(serial[index])
    assert parallel[5].to_dict()==serial[5].to_dict()
    assert parallel[6].to_dict()==serial[6].to_dict()
    assert parallel_order==serial_order
    assert parallel[0]['all_selection_queries']==dict(day_case_rollouts=16,resolved_task_responses=48)
    assert pool._pool is None


def test_only_deterministic_families_use_the_optional_evaluator(saved_models):
    legacy,users_path=saved_models
    config=VariedMarketConfig(jobs_per_user=1,batch_release_steps=(13,))
    factory=evaluation.posted_episode_factory(legacy,config=config)
    users=evaluation.load_users(users_path)
    cases=evaluation.preference_cases(range(1),1,110000)
    calibration=evaluation.preference_cases(range(1),1,120000)
    called=[]
    def deterministic_only(family,parameters,case_ids):
        called.append((family,tuple(case_ids)))
        for index,values in enumerate(parameters):
            controller=({'raw':lambda:evaluation.FixedRawOffer(values['action']),
                'matrix':lambda:evaluation.matrix_controller(values),
                'value_share':lambda:evaluation.ValueShareGreedy(values['rho'])}[family])()
            rows,_=evaluation.run_policy(factory,users,case_ids,f'{family}_{index}',controller=controller)
            yield rows
    evaluation.select_controls(factory,users,cases,calibration,raw_actions=(26,),
        templates=[evaluation.matrix_templates(config)[1]],rhos=(.3,),candidate_evaluator=deterministic_only)
    assert [item[0] for item in called]==['raw','matrix','value_share']
    assert all(item[1]==tuple(cases) for item in called)
    assert not any(set(item[1]).intersection(calibration) for item in called)


@pytest.mark.parametrize('workers',[0,-1,.5,True])
def test_invalid_worker_counts_are_rejected_without_spawning(workers,tmp_path):
    with pytest.raises(ValueError):SpawnedTemplateEvaluator(tmp_path,tmp_path,workers=workers)
