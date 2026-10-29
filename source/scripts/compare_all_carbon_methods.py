#!/usr/bin/env python3
"""Evaluate archived controls on the completed carbon experiment's cases.

No policy is trained and no controller is selected here. Existing primary
outcomes are reused; additional controllers keep their TRAIN-selected settings.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from carbon_offer_experiment import (DEFAULT_OUTPUT, FRESH_TEST, ROOT, bundle_for,
    evaluate, inputs, load_users, posted_episode_factory)
from energy_net.coordination.posted_offer_baselines import (
    FixedRawOffer, LearnedResponseGreedy, ValueShareGreedy)
from evaluate_posted_offers import enrich_day, matrix_controller, write_csv, write_json


LABELS = {
    'exit': '保留旧合同',
    'fixed_raw': '固定单合同',
    'fixed_matrix': '固定分区价表＋动态路由',
    'value_share': '资源节约比例规则',
    'response_prior': '经验接受概率规则',
    'cash_response_rule': '经验概率＋在线更新',
    'original_ppo': '原现金PPO（联合后）',
    'cash_continued': '普通PPO等步数续训',
    'cash_joint_ppo': '普通PPO续训＋一轮联合',
    'carbon_rule': '碳感知规则',
    'carbon_ppo': '碳PPO（冻结用户）',
    'carbon_joint_ppo': '碳PPO＋一轮联合',
    'old_fixed_ppo': '旧现金PPO（联合前）',
    'original_zero_history': '原PPO清空显式历史',
    'trained_no_history': '独立训练的无历史PPO',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    torch.set_num_threads(1)
    output = DEFAULT_OUTPUT / 'all_methods'
    output.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(DEFAULT_OUTPUT / 'test_days.csv', dtype={'case_id':'int64'})
    primary = primary[primary.cohort == 'fixed_users'].copy()
    case_set = set(FRESH_TEST)
    if any(set(group.case_id) != case_set for _,group in primary.groupby('scheme')):
        raise ValueError('The primary test must be complete and identically paired')
    selection_path = ROOT/'seed0/baselines-joint/selection.json'
    prior_path = ROOT/'seed0/baselines-joint/logit_prior.json'
    old_fixed_path = ROOT/'seed0/full-history/best_fixed/platform.zip'
    no_history_path = ROOT/'seed0/no-history/best_fixed/platform.zip'
    selection = json.loads(selection_path.read_text())['selected']
    extra_models = dict(old_fixed_ppo=old_fixed_path, original_zero_history=inputs()['original'],
                        trained_no_history=no_history_path)
    tracked = [selection_path, prior_path, *extra_models.values(),
               DEFAULT_OUTPUT/'test_days.csv']
    before = {str(path.relative_to(ROOT)): digest(path) for path in tracked}
    write_json(output/'settings.json', dict(root_algorithm_seed=0, cases=list(FRESH_TEST),
        users=str(inputs()['users'].relative_to(ROOT)), parameters=selection,
        sources_sha256=before, controller_selection='frozen archived TRAIN selection; none on TEST',
        purpose='matched evaluation only; no training or changes to previous results',
        names=LABELS))
    controls = [
        ('exit', FixedRawOffer(0)),
        ('fixed_raw', FixedRawOffer(selection['raw']['parameters']['action'])),
        ('fixed_matrix', matrix_controller(selection['matrix']['parameters'])),
        ('value_share', ValueShareGreedy(selection['value_share']['parameters']['rho'])),
        ('response_prior', LearnedResponseGreedy.from_json(prior_path)),
    ]
    factory = posted_episode_factory(inputs()['legacy'], public_cache_size=32, case_cache_size=128)
    users = load_users(inputs()['users'])
    additional = []
    for scheme, controller in controls:
        bundle = bundle_for(users)
        rows,_ = evaluate(factory,bundle,FRESH_TEST,controller=controller,label=scheme)
        additional.extend(dict(enrich_day(row,scheme),cohort='fixed_users') for row in rows)
        write_csv(output/'additional_days.csv',additional)
    for scheme,path in extra_models.items():
        bundle = bundle_for(users, history=scheme=='old_fixed_ppo')
        model = PPO.load(path, device='cpu')
        rows,_ = evaluate(factory,bundle,FRESH_TEST,model=model,label=scheme)
        additional.extend(dict(enrich_day(row,scheme),cohort='fixed_users') for row in rows)
        write_csv(output/'additional_days.csv',additional)
    combined = pd.concat([primary,pd.DataFrame(additional)],ignore_index=True)
    write_csv(output/'all_days.csv',combined.to_dict('records'))
    reference = json.loads((DEFAULT_OUTPUT/'reference.json').read_text())
    required = float(reference['required_profit_yuan'])
    stats = []
    for scheme,label in LABELS.items():
        group = combined[combined.scheme==scheme]
        if len(group)!=128 or set(group.case_id)!=case_set:
            raise ValueError(f'Missing matched cases: {scheme}')
        row = dict(scheme=scheme,method=label,public_days=32,private_cases=128,root_seed=0)
        for name in ('social_saving_vs_outside_yuan','platform_gain_vs_outside_yuan',
            'grid_emissions_reduction_t','carbon_emissions_t','outside_carbon_emissions_t',
            'dc_consumption_emissions_t','delta_dc_consumption_emissions_t','user_gain_yuan',
            'completion_rate','deadline_compliance','user_ir_rate','work_gpu_hours',
            'same_cap_compliant','old_profit_floor_satisfied'):
            row[name] = float(group[name].mean())
        row.update(required_profit_yuan=required,
            meets_training_reference_profit_floor=row['platform_gain_vs_outside_yuan']>=required)
        stats.append(row)
    write_csv(output/'comparison.csv',stats)
    names = ('方法','系统成本节约元/日','平台利润增量元/日','全网减排吨/日',
             '机房分摊碳足迹增量吨/日','满足4178.15元收益要求')
    brief = ['同一批测试场景上的全部现有比较方法','',
        '算法种子0；32日场景×4偏好，全部采用共同的原联合版三个用户。',
        '既有7种主方法的结果直接复用；补评旧规则和历史消融，不重新训练、不重新选参数。',
        '金额相对旧合同；机房足迹增量为正表示增加；全网减排为正表示减少。',
        '收益门槛固定为原PPO ALL TRAIN增量利润的95%，4178.1514元/日。',
        '各方法满足任务完成、截止时间、用户参与条件和原合同绝对收益底线。','', '\t'.join(names)]
    for row in stats:
        brief.append('\t'.join([row['method'],f"{row['social_saving_vs_outside_yuan']:.2f}",
            f"{row['platform_gain_vs_outside_yuan']:.2f}",f"{row['grid_emissions_reduction_t']:.6f}",
            f"{row['delta_dc_consumption_emissions_t']:+.6f}",
            str(row['meets_training_reference_profit_floor'])]))
    (output/'比较结果.txt').write_text('\n'.join(brief)+'\n',encoding='utf-8')
    after = {str(path.relative_to(ROOT)): digest(path) for path in tracked}
    if after!=before:
        raise ValueError('An archived policy or existing primary result changed')
    print(json.dumps(dict(status='complete',methods=len(stats),cases_per_method=128,
                          output=str(output))),flush=True)


if __name__=='__main__':
    run()
