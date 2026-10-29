#!/usr/bin/env python3
"""One-seed carbon/profit experiment, keeping the original IPPO snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from energy_net.coordination.learning import OptimizerStepCounter
from energy_net.coordination.posted_offer_baselines import LearnedResponseGreedy
from energy_net.coordination.posted_offer_learning import PostedOfferEnv
from evaluate_posted_offers import enrich_day, write_csv, write_json
from train_posted_offers import (TRAIN, VALIDATION, bundle_for, load_users, mean,
                                 posted_episode_factory, preference_cases, rollouts)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'experiments' / 'carbon-aware' / 'seed0'
FRESH_TEST = preference_cases(range(80000, 80032), 4, 400000)
RULE_TRAIN = tuple(case for i, case in enumerate(TRAIN) if i // 8 < 64 and i % 8 < 2)
PLATFORM_LR = 5e-5


def emit(**values):
    print(json.dumps(values, ensure_ascii=False, allow_nan=False), flush=True)


def inputs():
    return dict(legacy=ROOT / 'inputs' / 'legacy-seed0',
                users=ROOT / 'seed0' / 'full-history' / 'best_joint',
                original=ROOT / 'seed0' / 'full-history' / 'best_joint' / 'platform.zip',
                response=ROOT / 'seed0' / 'baselines-initial' / 'logit_online_map.json')


def hashes():
    paths = inputs()
    files = [paths['original'], paths['response'], *paths['users'].glob('*.zip'),
             paths['legacy'] / 'selection.json', *sorted((paths['legacy'] / 'selected').glob('*.zip'))]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def evaluate(factory, bundle, cases, *, model=None, controller=None, label='', offers=False):
    rows, feedback = [], []
    for start in range(0, len(cases), 128):
        batch, issued = rollouts(factory, bundle, cases[start:start+128],
            model=model, controller=controller, record_offers=offers)
        rows.extend(batch)
        feedback.extend(issued)
        emit(stage='evaluation', label=label, days=len(rows), total=len(cases),
             profit=mean(batch)['platform_gain_vs_outside_yuan'],
             reduction_t=mean(batch)['grid_emissions_reduction_t'])
    return rows, feedback


def prepare(output, args):
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(seed=0, algorithm='four independent actors; IPPO, not MAPPO',
        objective='max expected realised grid CO2 reduction subject to expected incremental cash profit floor',
        q=args.profit_fraction, profit_reference='frozen original platform + original users on ALL TRAIN cases',
        profit_units='yuan/day above the same accepted old bookings; not absolute net profit',
        carbon_units='tonnes/day, whole-grid actual generation, separate from allocated DC footprint',
        additional_fixed_platform_steps=args.steps, additional_joint_platform_steps=args.joint_steps,
        joint_cycles=1, joint_user_steps_per_owner=args.user_steps,
        joint_order='user0, user1, user2, platform; opponents frozen in each phase',
        cash_control='same original actor, fresh optimizer, same additional budget and target_kl',
        n_steps=256, batch_size=256, n_epochs=6, learning_rate=PLATFORM_LR,
        warm_learning_rate_reason='TRAIN-only 256-step integration check showed a first-update KL overshoot at 3e-4; both arms use 5e-5',
        gamma=1., gae_lambda=.95, ent_coef=.005, target_kl=.02,
        dual_initial=1., dual_lr=.1, dual_max=100.,
        scales=dict(profit_yuan=1000., emissions_t=1.),
        train_cases=list(TRAIN), validation_cases=list(VALIDATION), rule_train_cases=list(RULE_TRAIN),
        final_test_cases=list(FRESH_TEST), final_test_public_days=32, private_draws_per_day=4,
        rule_selection='TRAIN-only; highest CO2 reduction among profit-qualified lambda candidates; else highest profit with explicit failure flag',
        ppo_selection='VALIDATION-only; highest CO2 reduction among profit-qualified checkpoints; cash control maximises cash',
        physical_mechanism='same jobs, carbon cap, nonnegative funded contract surplus, user IR and real cash ledger',
        original_input_sha256=hashes(), final_test_opened=False)
    write_json(output / 'plan.json', plan)
    factory = posted_episode_factory(inputs()['legacy'])
    bundle = bundle_for(load_users(inputs()['users']))
    original = PPO.load(inputs()['original'], device='cpu')
    rows, _ = evaluate(factory, bundle, TRAIN, model=original, label='original_ALL_TRAIN')
    write_csv(output / 'reference_train_days.csv', rows)
    ref = mean(rows)
    write_json(output / 'reference.json', dict(mean=ref, q=args.profit_fraction,
        required_profit_yuan=args.profit_fraction * ref['platform_gain_vs_outside_yuan'],
        reference_profit_yuan=ref['platform_gain_vs_outside_yuan'], cases=len(TRAIN),
        original_input_sha256=hashes()))
    emit(stage='reference_ready', profit=ref['platform_gain_vs_outside_yuan'],
         required=args.profit_fraction * ref['platform_gain_vs_outside_yuan'])


def select_rules(output):
    from energy_net.coordination.carbon_offer_baselines import (
        CARBON_PROFIT_GRID, CarbonResponseGreedy, select_carbon_candidate)
    ref = json.loads((output / 'reference.json').read_text())
    response = LearnedResponseGreedy.from_json(inputs()['response'])
    factory = posted_episode_factory(inputs()['legacy'], public_cache_size=64, case_cache_size=128)
    bundle = bundle_for(load_users(inputs()['users']))
    candidates, days = [], []
    for index, weight in enumerate((*CARBON_PROFIT_GRID, None)):
        controller = CarbonResponseGreedy(response, lambda_profit=0. if weight is None else weight,
                                         cash_limit=weight is None)
        rows, _ = evaluate(factory, bundle, RULE_TRAIN, controller=controller,
                           label=f'carbon_rule_{index}')
        stats = mean(rows)
        candidates.append(dict(index=index, lambda_profit=weight, cash_limit=weight is None,
            mean_platform_gain_yuan=stats['platform_gain_vs_outside_yuan'],
            mean_grid_emissions_reduction_t=stats['grid_emissions_reduction_t'],
            mean_social_saving_yuan=stats['social_saving_vs_outside_yuan'],
            controller=controller.to_dict()))
        days.extend(dict(row, candidate=index) for row in rows)
        write_csv(output / 'rule_training_days.csv', days)
        write_json(output / 'rule_candidates.json', candidates)
    selected = select_carbon_candidate(candidates, ref['reference_profit_yuan'],
                                       profit_fraction=ref['q'])
    write_json(output / 'rule_selection.json', selected)
    chosen = CarbonResponseGreedy.from_dict(candidates[selected['selected_index']]['controller'])
    chosen.to_json(output / 'carbon_rule.json')
    validation, _ = evaluate(factory, bundle, VALIDATION, controller=chosen,
                             label='selected_carbon_rule_validation')
    write_csv(output / 'rule_validation_days.csv', validation)
    validation_stats = mean(validation)
    write_json(output / 'rule_validation.json', dict(**validation_stats,
        profit_qualified=validation_stats['platform_gain_vs_outside_yuan'] >= ref['required_profit_yuan'],
        required_profit_yuan=ref['required_profit_yuan'], selected_on='TRAIN only'))
    emit(stage='rule_selected', **{k:v for k,v in selected.items() if k != 'selected'})


class ValidationCallback(BaseCallback):
    def __init__(self, output, model_kind, bundle, factory, threshold, *, period=8192):
        super().__init__()
        self.output, self.kind, self.bundle, self.factory = output, model_kind, bundle, factory
        self.threshold, self.period = threshold, period
        self.next_check = period
        self.start_steps = 0
        self.records, self.training_records = [], []
        self.best_key = None

    def validate(self):
        local_step = self.model.num_timesteps - self.start_steps
        if self.records and self.records[-1]['additional_steps'] == local_step:
            return
        rows, _ = evaluate(self.factory, self.bundle, VALIDATION, model=self.model,
                           label=f'{self.kind}_validation_{local_step}')
        stats = dict(additional_steps=local_step, model_steps=self.model.num_timesteps, **mean(rows))
        cash = stats['platform_gain_vs_outside_yuan']
        carbon = stats['grid_emissions_reduction_t']
        eligible = cash >= self.threshold
        stats.update(profit_qualified=eligible, required_profit_yuan=self.threshold,
                     dual_value=float(getattr(self.model, 'dual_value', 0.)))
        key = (cash,) if self.kind.startswith('cash') else ((1, carbon, cash) if eligible else (0, cash, carbon))
        if self.best_key is None or key > self.best_key:
            self.best_key = key
            self.model.save(self.output / 'best_platform')
            write_csv(self.output / 'selected_validation_days.csv', rows)
            write_json(self.output / 'selection.json', stats)
        self.records.append(stats)
        write_csv(self.output / 'validation.csv', self.records)
        emit(stage='checkpoint', method=self.kind, additional_steps=local_step,
             profit=cash, carbon_reduction_t=carbon, eligible=eligible,
             dual=stats['dual_value'])

    def _on_training_start(self):
        self.start_steps = self.model.num_timesteps
        self.validate()

    def _on_step(self):
        return True

    def record_training(self):
        record = dict(steps=self.model.num_timesteps,
            **{key:float(value) for key,value in self.model.logger.name_to_value.items()
               if key.startswith('train/') and isinstance(value, (int,float,np.number))})
        if self.training_records and self.training_records[-1]['steps'] == record['steps']:
            self.training_records[-1] = record
        else:
            self.training_records.append(record)
        write_csv(self.output / 'training.csv', self.training_records)
        if getattr(self.model, 'dual_history', None):
            write_csv(self.output / 'dual_history.csv', self.model.dual_history)

    def _on_rollout_start(self):
        self.record_training()
        if self.model.num_timesteps - self.start_steps >= self.next_check:
            self.validate()
            self.next_check += self.period

    def _on_training_end(self):
        self.record_training()
        self.validate()
        if getattr(self.model, 'dual_history', None):
            write_csv(self.output / 'dual_history.csv', self.model.dual_history)


def train_method(output, kind, args):
    from energy_net.coordination.carbon_training import CarbonConstraintConfig, CarbonOfferEnv
    from energy_net.coordination.carbon_ppo import CarbonConstrainedPPO
    ref = json.loads((output / 'reference.json').read_text())
    destination = output / kind
    destination.mkdir(parents=True, exist_ok=True)
    users = load_users(inputs()['users'])
    bundle = bundle_for(users)
    bundle.training = True
    factory = posted_episode_factory(inputs()['legacy'])
    validation_factory = posted_episode_factory(inputs()['legacy'], public_cache_size=32, case_cache_size=128)
    cfg = CarbonConstraintConfig(q=ref['q'], profit_reference_yuan=ref['reference_profit_yuan'], lambda_init=1.)
    def platform_env():
        env = CarbonOfferEnv(factory, bundle, 'platform', TRAIN, factory(TRAIN[0]), carbon_config=cfg)
        gym.Env.reset(env, seed=0)
        return env
    env = platform_env()
    original = PPO.load(inputs()['original'], device='cpu')
    if kind == 'carbon':
        model = CarbonConstrainedPPO.from_profit_ppo(original, env,
            profit_floor_yuan_per_day=ref['required_profit_yuan'], seed=0,
            dual_initial=1., dual_lr=.1, dual_max=100., target_kl=.02,
            ent_coef=.005, learning_rate=PLATFORM_LR, carbon_reward_scale_t=cfg.emission_scale_t)
    else:
        model = PPO.load(inputs()['original'], env=env, device='cpu')
        model.num_timesteps = 0
        model._n_updates = 0
        model.target_kl, model.ent_coef, model.learning_rate = .02, .005, PLATFORM_LR
        model._setup_lr_schedule()
        model.set_random_seed(0)
        model.policy.optimizer = model.policy.optimizer_class(model.policy.parameters(),
            lr=PLATFORM_LR, **model.policy.optimizer_kwargs)
    bundle.platform = model
    callback = ValidationCallback(destination, kind, bundle, validation_factory, ref['required_profit_yuan'])
    counter = OptimizerStepCounter(model)
    started = time.monotonic()
    try:
        model.learn(total_timesteps=args.steps, reset_num_timesteps=False, callback=callback)
    finally:
        counter.close()
    model.save(destination / 'last_platform')
    write_json(destination / 'training_summary.json', dict(root_seed=0, method=kind,
        actual_additional_actor_steps=model.num_timesteps, optimizer_steps=counter.count,
        requested_actor_steps=args.steps, rollout_steps=model.n_steps, batch_size=model.batch_size,
        n_epochs=model.n_epochs, target_kl=model.target_kl, entropy_coefficient=model.ent_coef,
        wall_seconds=time.monotonic()-started, fixed_users=True,
        source_platform=str(inputs()['original']), source_platform_steps=original.num_timesteps,
        warm_start=getattr(model,'warm_start_metadata',None)))
    emit(stage='fixed_training_complete', method=kind, steps=model.num_timesteps)
    if args.joint_steps <= 0:
        return
    loader = CarbonConstrainedPPO if kind == 'carbon' else PPO
    model = loader.load(destination / 'best_platform.zip', device='cpu')
    bundle.platform = model
    bundle.training, bundle.sample_opponents = True, True
    bundle.opponent_sampling_probability = .1
    bundle.price_exploration_probability = .1
    bundle.exploration_rng = np.random.default_rng(701)
    joint = destination / 'joint'
    joint.mkdir(exist_ok=True)
    joint_started = time.monotonic()
    records = []
    for owner in range(3):
        user_env = PostedOfferEnv(factory, bundle, owner, TRAIN, factory(TRAIN[0]))
        gym.Env.reset(user_env, seed=owner+1)
        user = users[owner]
        user.set_env(user_env, force_reset=True)
        before = user.num_timesteps
        counter = OptimizerStepCounter(user)
        try:
            user.learn(total_timesteps=args.user_steps, reset_num_timesteps=False)
        finally:
            counter.close()
        user.save(joint / f'user_{owner}')
        records.append(dict(actor=str(owner), actual_steps=user.num_timesteps-before,
                            optimizer_steps=counter.count))
        write_json(joint / 'training.json', records)
        emit(stage='joint_user_complete', method=kind, owner=owner, steps=user.num_timesteps-before)
    # One platform phase after the three user phases. Same target was fixed
    # before any training; a different private response never changes the cap.
    env = platform_env()
    model.set_env(env, force_reset=True)
    callback = ValidationCallback(joint, kind+'_joint', bundle, validation_factory,
                                  ref['required_profit_yuan'], period=args.joint_steps)
    before = model.num_timesteps
    counter = OptimizerStepCounter(model)
    try:
        model.learn(total_timesteps=args.joint_steps, reset_num_timesteps=False, callback=callback)
    finally:
        counter.close()
    model.save(joint / 'last_platform')
    records.append(dict(actor='platform', actual_steps=model.num_timesteps-before,
                        optimizer_steps=counter.count))
    write_json(joint / 'training.json', records)
    write_json(joint / 'summary.json',dict(root_seed=0, cycles=1,
        requested_user_steps_per_owner=args.user_steps, requested_platform_steps=args.joint_steps,
        validation_candidates='joint phase start and joint phase end',
        wall_seconds=time.monotonic()-joint_started, actual_phase_records=records))
    emit(stage='joint_training_complete', method=kind, steps=model.num_timesteps-before)


def final_test(output):
    from energy_net.coordination.carbon_offer_baselines import CarbonResponseGreedy
    from energy_net.coordination.carbon_ppo import CarbonConstrainedPPO
    ref = json.loads((output / 'reference.json').read_text())
    plan = json.loads((output / 'plan.json').read_text())
    if hashes() != plan['original_input_sha256']:
        raise ValueError('Original snapshot inputs changed')
    variants = dict(original_ppo=PPO.load(inputs()['original'], device='cpu'),
        cash_continued=PPO.load(output / 'cash' / 'best_platform.zip', device='cpu'),
        carbon_ppo=CarbonConstrainedPPO.load(output / 'carbon' / 'best_platform.zip', device='cpu'))
    rule = CarbonResponseGreedy.from_json(output / 'carbon_rule.json')
    response = LearnedResponseGreedy.from_json(inputs()['response'])
    cohorts = [('fixed_users', inputs()['users'])]
    if (output / 'carbon' / 'joint' / 'user_0.zip').exists():
        cohorts.append(('carbon_joint_users', output / 'carbon' / 'joint'))
        variants['carbon_joint_ppo'] = CarbonConstrainedPPO.load(
            output / 'carbon' / 'joint' / 'best_platform.zip', device='cpu')
    if (output / 'cash' / 'joint' / 'best_platform.zip').exists():
        cohorts.append(('cash_joint_users', output / 'cash' / 'joint'))
        variants['cash_joint_ppo'] = PPO.load(output / 'cash' / 'joint' / 'best_platform.zip', device='cpu')
    # Freeze every choice before creating the first final-test scenario.
    frozen = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in output.rglob('*') if p.suffix in ('.zip',) or p.name in
              ('carbon_rule.json', 'reference.json', 'rule_selection.json')}
    write_json(output / 'test_freeze.json', dict(inputs=frozen, cases=list(FRESH_TEST), root_seed=0))
    plan['final_test_opened'] = True
    write_json(output / 'plan.json', plan)
    all_rows, offers = [], []
    for cohort, users_path in cohorts:
        factory = posted_episode_factory(inputs()['legacy'], public_cache_size=32, case_cache_size=128)
        bundle = bundle_for(load_users(users_path))
        methods = [(key, model, None) for key,model in variants.items()]
        methods.extend([('carbon_rule',None,rule),('cash_response_rule',None,response)])
        for scheme, model, controller in methods:
            rows, issued = evaluate(factory,bundle,FRESH_TEST,model=model,controller=controller,
                                    label=cohort+':'+scheme,offers=True)
            all_rows.extend(dict(enrich_day(row,scheme), cohort=cohort) for row in rows)
            offers.extend(dict(row,scheme=scheme,cohort=cohort) for row in issued)
            write_csv(output / 'test_days.csv',all_rows)
            write_json(output / 'test_offers.json',offers)
    data = pd.DataFrame(all_rows)
    summary = data.groupby(['cohort','scheme']).mean(numeric_only=True).reset_index()
    summary['required_profit_yuan'] = ref['required_profit_yuan']
    summary['test_profit_qualified'] = summary.platform_gain_vs_outside_yuan >= ref['required_profit_yuan']
    rule_selection = json.loads((output / 'rule_selection.json').read_text())
    rule_validation = json.loads((output / 'rule_validation.json').read_text())
    full_rule_check = json.loads((output / 'rule_all_train_check.json').read_text())
    selected_full_rule = next(row for row in full_rule_check['results']
                             if row['controller'] == rule.to_dict())
    summary['selection_split'] = summary.scheme.map(lambda scheme:
        'TRAIN' if scheme == 'carbon_rule' else 'VALIDATION' if scheme in variants and scheme != 'original_ppo'
        else 'frozen reference')
    summary['train_profit_qualified'] = summary.scheme.map(lambda scheme:
        selected_full_rule['mean_platform_gain_yuan'] >= ref['required_profit_yuan']
        if scheme == 'carbon_rule' else None)
    summary['rule_calibration_profit_qualified'] = summary.scheme.map(lambda scheme:
        rule_selection['feasible'] if scheme == 'carbon_rule' else None)
    def selected_validation_qualified(scheme):
        if scheme=='carbon_rule':
            return rule_validation['profit_qualified']
        paths = dict(cash_continued=output/'cash',carbon_ppo=output/'carbon',
                     cash_joint_ppo=output/'cash'/'joint', carbon_joint_ppo=output/'carbon'/'joint')
        return (json.loads((paths[scheme]/'selection.json').read_text())['profit_qualified']
                if scheme in paths else None)
    summary['validation_profit_qualified'] = summary.scheme.map(selected_validation_qualified)
    reference = summary[summary.scheme=='original_ppo'].set_index('cohort').platform_gain_vs_outside_yuan
    summary['retained_original_profit_fraction'] = summary.apply(
        lambda row: row.platform_gain_vs_outside_yuan/reference.loc[row.cohort],axis=1)
    summary['same_test_95pct_profit_qualified'] = summary.retained_original_profit_fraction >= ref['q']
    summary['same_test_profit_gap_yuan'] = summary.apply(
        lambda row: row.platform_gain_vs_outside_yuan-ref['q']*reference.loc[row.cohort],axis=1)
    write_csv(output / 'test_summary.csv', summary.to_dict('records'))
    emit(stage='complete', output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('prepare','rule','cash','carbon','test'))
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--steps',type=int,default=49152)
    parser.add_argument('--joint-steps',type=int,default=8192)
    parser.add_argument('--user-steps',type=int,default=4096)
    parser.add_argument('--profit-fraction',type=float,default=.95)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.stage=='prepare': prepare(args.output,args)
    elif args.stage=='rule': select_rules(args.output)
    elif args.stage in ('cash','carbon'): train_method(args.output,args.stage,args)
    else: final_test(args.output)


if __name__=='__main__':
    main()
