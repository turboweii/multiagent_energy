#!/usr/bin/env python3
"""Matched TRAIN-only comparison of greedy and sampled platform execution.

The private-user policies always execute deterministically. Four stochastic
platform trajectories are evaluated together per case, with independent
categorical draws, to avoid repeated neural-network overhead. No learning or
checkpoint selection is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import t
import torch
from stable_baselines3 import PPO

from carbon_offer_experiment import ROOT, RULE_TRAIN, bundle_for, hashes, inputs, load_users, posted_episode_factory
from energy_net.coordination.carbon_ppo import CarbonConstrainedPPO
from energy_net.coordination.preference_scenario import decode_preference_episode_id


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_snapshot(source, target, attempts=20):
    """Read a complete unchanged ZIP twice, verify every CRC, then atomically copy."""
    temporary = target.with_suffix('.copying.zip')
    for attempt in range(attempts):
        try:
            before = source.stat()
            payload = source.read_bytes()
            after = source.stat()
            digest = hashlib.sha256(payload).hexdigest()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                time.sleep(.1)
                continue
            if file_hash(source) != digest:
                time.sleep(.1)
                continue
            temporary.write_bytes(payload)
            with zipfile.ZipFile(temporary) as archive:
                if archive.testzip() is not None:
                    raise zipfile.BadZipFile('Member CRC failed')
                if not {'data', 'policy.pth'} <= set(archive.namelist()):
                    raise zipfile.BadZipFile('Incomplete SB3 model archive')
            temporary.replace(target)
            return dict(source=str(source), snapshot=str(target), sha256=digest,
                        copied_bytes=len(payload), crc_verified=True, read_attempts=attempt+1)
        except (OSError, zipfile.BadZipFile):
            time.sleep(.1)
    raise RuntimeError('Could not obtain a stable complete model ZIP; no cases evaluated')


def execute_case(factory, bundle, model, case, *, deterministic, draws, action_seed):
    episodes = [factory(case) for _ in range(draws)]
    torch.manual_seed(action_seed)
    while not episodes[0].done:
        if any(episode.done for episode in episodes):
            raise ValueError('Matched tasks must have the same number of quote steps')
        observations = np.stack([bundle.platform_observation(episode) for episode in episodes])
        actions, _ = model.predict(observations, deterministic=deterministic)
        for episode, action in zip(episodes, np.asarray(actions).reshape(-1)):
            episode.set_offer(int(action))
            episode.step_accept(bundle.user_action(episode))
    return [episode.outcome().metrics for episode in episodes]


def matched_summary(frame):
    metrics = ['profit_gain_yuan', 'grid_emissions_reduction_t', 'social_saving_yuan']
    case_means = frame.groupby(['model','decode','public_seed','case_id'])[metrics].mean()
    summaries, contrasts = [], []
    for (label, decode), group in frame.groupby(['model','decode'], sort=False):
        averaged = case_means.xs((label, decode), level=('model','decode'))
        summary = dict(model=label, decode=decode, cases=group.case_id.nunique(),
                       public_weather_days=group.public_seed.nunique(),
                       action_draws_per_case=int(group.groupby('case_id').size().iloc[0]),
                       simulated_days=len(group))
        summary.update({f'mean_{metric}':float(averaged[metric].mean()) for metric in metrics})
        if decode == 'sampled':
            spread = group.groupby('case_id')[metrics].std(ddof=1)
            summary.update({f'mean_within_case_draw_std_{metric}':float(spread[metric].mean()) for metric in metrics})
        summaries.append(summary)
    for label in frame.model.unique():
        greedy = case_means.xs((label,'greedy'), level=('model','decode'))
        sampled = case_means.xs((label,'sampled'), level=('model','decode'))
        difference = (sampled-greedy).groupby('public_seed').mean()
        record = dict(model=label, contrast='sampled_mean_minus_greedy', public_weather_days=len(difference))
        for metric in metrics:
            values = difference[metric]
            half = float(t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
            record[metric] = dict(mean=float(values.mean()),low95=float(values.mean()-half),
                                  high95=float(values.mean()+half))
        contrasts.append(record)
    return summaries, contrasts


def summarize_existing(output, *, optional_original_skipped=False):
    """Finalize already completed model panels without simulating any new case."""
    manifest = json.loads((output/'manifest.json').read_text())
    frame = pd.read_csv(output/'decoding_days.csv')
    expected = len(manifest['cases'])
    for (label,decode), group in frame.groupby(['model','decode']):
        count = 1 if decode=='greedy' else manifest['sampled_draws']
        if (group.case_id.nunique()!=expected or len(group)!=expected*count
                or not group.groupby('case_id').size().eq(count).all()):
            raise ValueError(f'Incomplete execution panel: {label}/{decode}')
    reference = json.loads((ROOT/'experiments/carbon-aware/seed0/reference.json').read_text())
    summaries, contrasts = matched_summary(frame)
    for summary in summaries:
        summary['required_profit_yuan'] = reference['required_profit_yuan']
        summary['profit_qualified_panel_estimate'] = summary['mean_profit_gain_yuan']>=reference['required_profit_yuan']-1e-6
    if file_hash(Path(manifest['model_snapshot']['snapshot']))!=manifest['model_snapshot']['sha256']:
        raise RuntimeError('The frozen snapshot changed')
    report = dict(manifest=manifest,summary=summaries,paired_weather_contrasts=contrasts,
        interval_population='64 public weather means after averaging two private cases and sampled action draws',
        required_profit_source='fixed ALL TRAIN reference; this 128-case panel is a diagnostic estimate',
        optional_original_skipped=optional_original_skipped,
        interpretation='execution-mode diagnostic only; model and decoding selection remain unchanged')
    (output/'decoding_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(status='complete',output=str(output),summary=summaries,contrasts=contrasts),ensure_ascii=False),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT/'experiments/carbon-aware/seed0/carbon/best_platform.zip')
    parser.add_argument('--output', type=Path, default=Path('/tmp/carbon-decode-check'))
    parser.add_argument('--draws', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--include-original', action='store_true')
    parser.add_argument('--summarize-existing', action='store_true',
                        help='Only finish complete CSV panels already written; no model load or rollout')
    args = parser.parse_args()
    if args.draws < 2:
        parser.error('Use at least two independent action draws per case')
    torch.set_num_threads(1)
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'decoding_summary.json').exists():
        raise FileExistsError('A completed diagnostic exists; use a new output folder')
    if args.summarize_existing:
        summarize_existing(output,optional_original_skipped=True)
        return
    snapshot = stable_snapshot(args.model.resolve(),output/'carbon_platform_snapshot.zip')
    reference_path = ROOT/'experiments/carbon-aware/seed0/reference.json'
    reference = json.loads(reference_path.read_text())
    input_hashes = hashes()
    if input_hashes != reference['original_input_sha256']:
        raise ValueError('Original matched inputs differ from the frozen TRAIN reference')
    models = [('carbon_snapshot', CarbonConstrainedPPO.load(output/'carbon_platform_snapshot.zip',device='cpu'))]
    if args.include_original:
        models.append(('original_ppo',PPO.load(inputs()['original'],device='cpu')))
    bundle = bundle_for(load_users(inputs()['users']))
    bundle.training, bundle.sample_opponents = False, False
    factory = posted_episode_factory(inputs()['legacy'], public_cache_size=64,case_cache_size=128)
    manifest = dict(split='TRAIN',cases=list(RULE_TRAIN),root_action_seed=args.seed,
        algorithm_training_seed=0,private_users='matched best_joint; deterministic for every rollout',
        sampled_draws=args.draws,model_snapshot=snapshot,
        snapshot_model_steps=int(models[0][1].num_timesteps),
        snapshot_dual_value=float(models[0][1].dual_value),
        original_input_sha256=input_hashes,test_cases_used=[],
        sampling='independent categorical rows; same case-derived action seed across models',
        no_model_selection=True)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    rows = []
    for label, model in models:
        for case_index,case in enumerate(RULE_TRAIN):
            public, private = decode_preference_episode_id(case)
            action_seed = int(np.random.SeedSequence([args.seed,case_index]).generate_state(1)[0])
            for decode, deterministic, draws in [('greedy',True,1),('sampled',False,args.draws)]:
                results = execute_case(factory,bundle,model,case,deterministic=deterministic,
                                       draws=draws,action_seed=action_seed)
                for draw, metrics in enumerate(results):
                    rows.append(dict(model=label,decode=decode,case_id=case,public_seed=public,
                        preference_seed=private,action_draw=draw,action_rng_seed=action_seed,
                        profit_gain_yuan=float(metrics['platform_gain_vs_outside_yuan']),
                        grid_emissions_reduction_t=float(metrics['grid_emissions_reduction_t']),
                        social_saving_yuan=float(metrics['social_saving_vs_outside_yuan']),
                        grid_emissions_t=float(metrics['carbon_emissions_t']),
                        dc_footprint_change_t=float(metrics['delta_dc_consumption_emissions_t']),
                        posted_offers=float(metrics['posted_offers']),accepted_offers=float(metrics['accepted_offers'])))
            if (case_index+1)%16==0:
                print(json.dumps(dict(model=label,cases=case_index+1,total=len(RULE_TRAIN),
                    elapsed_seconds=time.monotonic()-started)),flush=True)
        pd.DataFrame(rows).to_csv(output/'decoding_days.csv',index=False,encoding='utf-8-sig')
    frame = pd.DataFrame(rows)
    summaries, contrasts = matched_summary(frame)
    for summary in summaries:
        summary['required_profit_yuan'] = reference['required_profit_yuan']
        summary['profit_qualified_panel_estimate'] = summary['mean_profit_gain_yuan'] >= reference['required_profit_yuan']-1e-6
    if file_hash(output/'carbon_platform_snapshot.zip') != snapshot['sha256'] or hashes()!=input_hashes:
        raise RuntimeError('A frozen diagnostic model/input changed')
    report = dict(manifest=manifest,summary=summaries,paired_weather_contrasts=contrasts,
        interval_population='64 public weather means after averaging two private cases and sampled action draws',
        required_profit_source='fixed ALL TRAIN reference; this 128-case panel is a diagnostic estimate',
        runtime_seconds=time.monotonic()-started,
        interpretation='execution-mode diagnostic only; model and decoding selection remain unchanged')
    (output/'decoding_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(status='complete',output=str(output),summary=summaries,
                         contrasts=contrasts,elapsed_seconds=report['runtime_seconds']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
