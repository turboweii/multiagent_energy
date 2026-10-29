#!/usr/bin/env python3
"""Predeclared sampled-policy supplement after the frozen primary test.

No test scenario is instantiated on import or by --self-check. The explicit
normal CLI requires a completed primary test and all frozen model hashes.
Every platform uses four paired categorical action draws, while the same
original private users execute deterministically. No checkpoint/decode choice
is made from these outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import t
import torch
from stable_baselines3 import PPO

from carbon_offer_experiment import FRESH_TEST, ROOT, bundle_for, hashes, inputs, load_users, posted_episode_factory
from check_carbon_decoding import execute_case
from evaluate_posted_offers import enrich_day
from energy_net.coordination.carbon_ppo import CarbonConstrainedPPO
from energy_net.coordination.preference_scenario import decode_preference_episode_id


PPO_SCHEMES = ('original_ppo','cash_continued','carbon_ppo','cash_joint_ppo','carbon_joint_ppo')
METRICS = ('actual_G_yuan','actual_R_yuan','actual_M_yuan','actual_S_yuan',
    'actual_platform_gain_yuan','actual_user_gain_yuan','actual_buyer_gain_yuan',
    'platform_gain_vs_outside_yuan','grid_emissions_reduction_t','social_saving_vs_outside_yuan',
    'carbon_emissions_t','dc_consumption_emissions_t','delta_dc_consumption_emissions_t',
    'work_gpu_hours','required_work_cu','completion_rate','deadline_compliance','user_ir_rate',
    'same_cap_compliant','old_profit_floor_satisfied','posted_offers','accepted_offers',
    'platform_profit_yuan','outside_platform_profit_yuan')
PAIR_METRICS = ('platform_gain_vs_outside_yuan','grid_emissions_reduction_t',
                'social_saving_vs_outside_yuan','carbon_emissions_t',
                'dc_consumption_emissions_t','delta_dc_consumption_emissions_t')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mid(scheme, decode):
    return scheme+'__'+decode


def aggregate(rows, required_profit_yuan):
    """Draw mean -> private-case mean -> equally weighted public weather mean."""
    frame = pd.DataFrame(rows)
    columns = [column for column in METRICS if column in frame]
    keys = ['method','scheme','decode','cohort','public_seed','case_id']
    cases = frame.groupby(keys,sort=False)[columns].mean().reset_index()
    weather_keys = ['method','scheme','decode','cohort','public_seed']
    weather = cases.groupby(weather_keys,sort=False)[columns].mean().reset_index()
    sampled_old = weather.loc[weather.method==mid('original_ppo','sampled'),
                               'platform_gain_vs_outside_yuan'].mean()
    greedy_old = weather.loc[weather.method==mid('original_ppo','greedy_reference'),
                              'platform_gain_vs_outside_yuan'].mean()
    summaries = []
    for method, group in weather.groupby('method',sort=False):
        actual = frame.loc[frame.method==method]
        record = {key:group.iloc[0][key] for key in ['method','scheme','decode','cohort']}
        record.update({column:float(group[column].mean()) for column in columns})
        cash = record['platform_gain_vs_outside_yuan']
        record.update(public_weather_days=group.public_seed.nunique(),
            private_cases=actual.case_id.nunique(),action_draws_per_case=int(actual.groupby('case_id').size().iloc[0]),
            total_outcomes=len(actual),required_profit_yuan=required_profit_yuan,
            profit_qualified_absolute_train=cash>=required_profit_yuan-1e-6,
            profit_shortfall_yuan=max(0.,required_profit_yuan-cash),
            same_test_original_greedy_profit_yuan=float(greedy_old),
            same_test_original_sampled_profit_yuan=float(sampled_old),
            retained_original_greedy_profit_fraction=cash/greedy_old,
            retained_original_sampled_profit_fraction=cash/sampled_old)
        summaries.append(record)
    method_ids = list(weather.method.unique())
    references = [mid('original_ppo','sampled'),mid('original_ppo','greedy_reference')]
    references += [method for method in method_ids if method.endswith('__deterministic_rule')]
    contrasts = {(mid(scheme,'sampled'), reference) for scheme in PPO_SCHEMES
                 for reference in references if mid(scheme,'sampled')!=reference}
    contrasts.update((mid(a,'sampled'),mid(b,'sampled')) for a,b in (
        ('carbon_ppo','cash_continued'),('carbon_joint_ppo','cash_joint_ppo'),
        ('carbon_joint_ppo','carbon_ppo'),('cash_joint_ppo','cash_continued')))
    paired = []
    for method, reference in sorted(contrasts):
        if method not in method_ids or reference not in method_ids:
            continue
        a = weather.loc[weather.method==method].set_index('public_seed')
        b = weather.loc[weather.method==reference].set_index('public_seed')
        if set(a.index)!=set(b.index):
            raise ValueError('Unpaired public-weather cases')
        case_a = set(cases.loc[cases.method==method,'case_id'])
        case_b = set(cases.loc[cases.method==reference,'case_id'])
        if case_a!=case_b:
            raise ValueError('Unpaired private cases')
        for public in sorted(a.index):
            row = dict(method=method,reference_method=reference,public_seed=int(public),
                       scheme=a.loc[public,'scheme'],decode=a.loc[public,'decode'],
                       reference_scheme=b.loc[public,'scheme'],reference_decode=b.loc[public,'decode'],
                       cohort='fixed_users')
            for column in PAIR_METRICS:
                if column in columns:
                    row['method_'+column]=float(a.loc[public,column])
                    row['reference_'+column]=float(b.loc[public,column])
                    row['delta_'+column]=float(a.loc[public,column]-b.loc[public,column])
            paired.append(row)
    return pd.DataFrame(summaries),pd.DataFrame(paired),weather


def brief(summary, paired, required):
    lines=['冻结PPO的采样执行补充结果','',
        '训练算法种子仍为0。每个PPO每个case有4次独立动作采样，不是4个训练种子。',
        '全部使用原best_joint三用户并确定性响应。仅平台采样，规则与原PPO greedy参考继承主测试结果。',
        '先平均动作draw，再平均每个天气日的4组私有偏好，最后对32天气日等权汇总。',
        f'固定TRAIN收益门槛为{required:.6f}元/日。另列相对同TEST原PPO greedy与sampled收益比例，二者不替换原门槛。','',
        '方法\t执行\tΔΠ元/日\t全网减排t/日\tS元/日\tDC足迹变化t/日\tTRAIN门槛合格\t原greedy收益比\t原sampled收益比']
    for row in summary.to_dict('records'):
        lines.append('\t'.join([row['scheme'],row['decode'],
            f"{row['platform_gain_vs_outside_yuan']:.2f}",f"{row['grid_emissions_reduction_t']:.6f}",
            f"{row['social_saving_vs_outside_yuan']:.2f}",f"{row['delta_dc_consumption_emissions_t']:.6f}",
            str(row['profit_qualified_absolute_train']),f"{row['retained_original_greedy_profit_fraction']:.4%}",
            f"{row['retained_original_sampled_profit_fraction']:.4%}"]))
    lines.extend(['','天气配对差值（本方法减参考，95%区间仅描述给定策略下天气差异及有限动作采样）：'])
    for (method,reference), group in paired.groupby(['method','reference_method'],sort=False):
        parts=[]
        for column in ('platform_gain_vs_outside_yuan','grid_emissions_reduction_t','social_saving_vs_outside_yuan'):
            values=group['delta_'+column]
            half=float(t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.
            parts.append(f'{column}={values.mean():.6f}[{values.mean()-half:.6f},{values.mean()+half:.6f}]')
        lines.append(method+' vs '+reference+'\t'+'\t'.join(parts))
    lines.extend(['','greedy主表保留；本补充不用于更换模型、重选规则、修改约束或在TEST上挑执行方式。',''])
    return '\n'.join(lines)


def run(output, draws=4, seed=0):
    started=time.monotonic()
    if draws!=4 or seed!=0:
        raise ValueError('The predeclared supplement uses four action draws and root seed 0')
    if (output/'sampled_test_summary.csv').exists():
        raise FileExistsError('Completed sampled supplement exists; do not repeat TEST')
    if not (output/'test_summary.csv').is_file():
        raise FileNotFoundError('Wait for the completed primary test before running this CLI')
    freeze=json.loads((output/'test_freeze.json').read_text())
    plan=json.loads((output/'plan.json').read_text())
    if not plan.get('secondary_sampled_evaluation'):
        raise ValueError('The sampled supplement must be declared before opening TEST')
    if hashes()!=plan['original_input_sha256']:
        raise ValueError('Frozen original inputs changed')
    reference=json.loads((output/'reference.json').read_text())
    paths=dict(original_ppo=inputs()['original'],cash_continued=output/'cash/best_platform.zip',
        carbon_ppo=output/'carbon/best_platform.zip',cash_joint_ppo=output/'cash/joint/best_platform.zip',
        carbon_joint_ppo=output/'carbon/joint/best_platform.zip')
    frozen_models={}
    for scheme,path in paths.items():
        key=str(path.relative_to(ROOT)) if scheme=='original_ppo' else str(path.relative_to(output))
        expected=(plan['original_input_sha256'] if scheme=='original_ppo' else freeze['inputs']).get(key)
        if expected is None or digest(path)!=expected:
            raise ValueError(f'Model does not match primary freeze: {scheme}')
        frozen_models[scheme]=expected
    primary_path=output/'test_days.csv'
    primary_hash=digest(primary_path)
    primary=pd.read_csv(primary_path,dtype={'case_id':'int64'})
    primary=primary.loc[primary.cohort=='fixed_users'].copy()
    for scheme in PPO_SCHEMES:
        original=primary.loc[primary.scheme==scheme]
        if len(original)!=len(FRESH_TEST) or set(original.case_id)!=set(FRESH_TEST):
            raise ValueError('Primary fixed-user test must be complete for all five frozen PPOs')
    bundle=bundle_for(load_users(inputs()['users']))
    bundle.training,bundle.sample_opponents=False,False
    factory=posted_episode_factory(inputs()['legacy'],public_cache_size=32,case_cache_size=128)
    rows=[]
    for scheme in PPO_SCHEMES:
        loader=CarbonConstrainedPPO if scheme.startswith('carbon_') else PPO
        model=loader.load(paths[scheme],device='cpu')
        for index,case in enumerate(FRESH_TEST):
            public,private=decode_preference_episode_id(case)
            rng_seed=int(np.random.SeedSequence([seed,index]).generate_state(1)[0])
            outcomes=execute_case(factory,bundle,model,case,deterministic=False,draws=draws,action_seed=rng_seed)
            for draw,metrics in enumerate(outcomes):
                rows.append(dict(enrich_day(dict(case_id=case,**metrics),scheme),
                    method=mid(scheme,'sampled'),decode='sampled',cohort='fixed_users',
                    public_seed=public,preference_seed=private,action_draw=draw,action_rng_seed=rng_seed))
            if (index+1)%16==0:
                print(json.dumps(dict(scheme=scheme,cases=index+1,total=len(FRESH_TEST),
                    elapsed_seconds=time.monotonic()-started)),flush=True)
        pd.DataFrame(rows).to_csv(output/'sampled_test_days.csv',index=False,encoding='utf-8-sig')
    # Rules are deterministic with deterministic users, so reuse one physical
    # outcome per case rather than replicating it as four invented rollouts.
    for record in primary.to_dict('records'):
        scheme=record['scheme']
        if scheme in PPO_SCHEMES and scheme!='original_ppo':
            continue
        decode='greedy_reference' if scheme=='original_ppo' else 'deterministic_rule'
        rows.append(dict(record,method=mid(scheme,decode),decode=decode,
                         action_draw=0,action_rng_seed=None))
    frame=pd.DataFrame(rows)
    sampled=frame.loc[frame.decode=='sampled']
    if not sampled.groupby(['scheme','case_id']).size().eq(draws).all():
        raise ValueError('Incomplete per-case action draw panel')
    if not sampled.groupby('case_id').action_rng_seed.nunique().eq(1).all():
        raise ValueError('PPO action sampling is not paired across models')
    summary,paired,weather=aggregate(rows,float(reference['required_profit_yuan']))
    for scheme,path in paths.items():
        if digest(path)!=frozen_models[scheme]:
            raise RuntimeError('A frozen model changed during the supplement')
    if hashes()!=plan['original_input_sha256'] or digest(primary_path)!=primary_hash:
        raise RuntimeError('Original inputs or primary outcomes changed')
    frame.to_csv(output/'sampled_test_days.csv',index=False,encoding='utf-8-sig')
    paired.to_csv(output/'sampled_weather_paired.csv',index=False,encoding='utf-8-sig')
    weather.to_csv(output/'sampled_weather_means.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(output/'sampled_test_summary.csv',index=False,encoding='utf-8-sig')
    (output/'sampled_summary.txt').write_text(brief(summary,paired,reference['required_profit_yuan']),encoding='utf8')
    (output/'sampled_evaluation_manifest.json').write_text(json.dumps(dict(
        root_action_seed=seed,algorithm_training_seed=0,action_draws_per_case=draws,
        cohort='fixed_users',cases=list(FRESH_TEST),frozen_model_sha256=frozen_models,
        primary_outcomes_sha256=primary_hash,model_selection_changed=False,
        elapsed_seconds=time.monotonic()-started),ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(status='complete',summary=summary.to_dict('records'),
                         elapsed_seconds=time.monotonic()-started),ensure_ascii=False),flush=True)


def self_check():
    """One fake case, four action rows, no real factory/model/test-file access."""
    class Episode:
        def __init__(self): self.position,self.cash,self.carbon=0,0.,0.
        @property
        def done(self): return self.position==2
        def set_offer(self,action): self.action=action
        def step_accept(self,accept):
            self.cash+=float(self.action+1); self.carbon+=.1*self.action; self.position+=1
        def outcome(self):
            return SimpleNamespace(metrics=dict(profit=self.cash,carbon=self.carbon))
    class Model:
        def predict(self,observations,deterministic):
            probabilities=torch.tensor([[.4,.6]]).repeat(len(observations),1)
            return torch.multinomial(probabilities,1).numpy().reshape(-1),None
    bundle=SimpleNamespace(platform_observation=lambda e:np.array([e.position],np.float32),
                           user_action=lambda e:1)
    rows=[]
    for scheme in PPO_SCHEMES:
        outcomes=execute_case(lambda case:Episode(),bundle,Model(),-1,
                              deterministic=False,draws=4,action_seed=0)
        for draw,result in enumerate(outcomes):
            rows.append(dict(method=mid(scheme,'sampled'),scheme=scheme,decode='sampled',
                cohort='fixed_users',public_seed=0,case_id=-1,action_draw=draw,
                platform_gain_vs_outside_yuan=result['profit'],grid_emissions_reduction_t=result['carbon'],
                social_saving_vs_outside_yuan=result['profit']+result['carbon']))
    rows.append(dict(method=mid('original_ppo','greedy_reference'),scheme='original_ppo',
        decode='greedy_reference',cohort='fixed_users',public_seed=0,case_id=-1,action_draw=0,
        platform_gain_vs_outside_yuan=4.,grid_emissions_reduction_t=.2,social_saving_vs_outside_yuan=4.2))
    summary,paired,_=aggregate(rows,2.)
    sampled=summary.loc[summary.decode=='sampled']
    assert sampled.action_draws_per_case.eq(4).all() and sampled.private_cases.eq(1).all()
    assert sampled.platform_gain_vs_outside_yuan.nunique()==1
    comparison=paired.loc[(paired.method==mid('carbon_ppo','sampled'))
                          & (paired.reference_method==mid('cash_continued','sampled'))]
    assert comparison['delta_platform_gain_vs_outside_yuan'].eq(0).all()
    print(json.dumps(dict(self_check='passed',fake_case_count=1,sampled_outcomes=20,
                         real_cases_requested=0,real_test_files_read=0)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/carbon-aware/seed0')
    parser.add_argument('--draws',type=int,default=4)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.self_check: self_check()
    else: run(args.output.resolve(),args.draws,args.seed)


if __name__=='__main__':
    main()
