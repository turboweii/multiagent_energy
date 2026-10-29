#!/usr/bin/env python3
"""Compare GPU pricing agents with train-selected constant-menu benchmarks.

The primary information ablation holds the three user policies, carbon
preference, raw 77-menu action space and scenarios fixed. Its two platform
actors differ only in access to explicit carbon observations. The eta=0 case
changes the firm's preference and is reported as a separate reference.

A constant is selected using only training scenarios, including original
pricing as menu zero. Test outcomes are never used to select a constant.
Financial carbon valuation, paid electricity costs, full-grid resource cost,
consumption-footprint allocation and actual grid emissions remain separate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from energy_net.coordination.gpu_market import gpu_episode_factory
from energy_net.coordination.gpu_scenario import GpuMarketConfig
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue, GpuTariffObserver
from energy_net.coordination.learning import PolicyBundle


EXPERIMENTS = {
    "cash_only": (0.0, False),
    "electricity_only": (100.0, False),
    "electricity_carbon": (100.0, True),
}


def user_hashes(directory):
    return {str(user): hashlib.sha256((directory / f"user_{user}.zip").read_bytes()).hexdigest()
            for user in range(3)}


def load_users(directory):
    return {user: PPO.load(directory / f"user_{user}.zip", device="cpu") for user in range(3)}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def number_means(rows):
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows]))
            for key, value in rows[0].items()
            if isinstance(value, (int, float, bool, np.number))}


def score_complete(metrics):
    return (metrics["completion_rate"] >= 1 - 1e-9
            and metrics["deadline_compliance"] >= 1 - 1e-9
            and metrics["user_ir_rate"] >= 1 - 1e-9)


def play_menu(episode, catalogue, menu, users):
    episode.reset(catalogue.decode(episode, menu))
    return episode.play(users).metrics


def fit_constants(output, config, initial_users, train_seeds):
    """Enumerate each raw menu once; eta affects objective accounting alone."""
    output.mkdir(parents=True, exist_ok=True)
    catalogue = FixedGpuTariffCatalogue(config)
    manifest = dict(
        config=asdict(config), initial_user_sha256=user_hashes(initial_users),
        training_scenario_seeds=list(train_seeds),
        catalogue_labels=list(catalogue.labels),
        selection="maximum training-day mean own objective, all complete-work/user-IR days; ties lowest index",
    )
    # JSON normalises dataclass tuples to lists before comparing a prior cache.
    manifest = json.loads(json.dumps(manifest))
    result_path, manifest_path = output / "selected_constants.json", output / "manifest.json"
    if result_path.exists() and manifest_path.exists():
        if json.loads(manifest_path.read_text()) == manifest:
            return json.loads(result_path.read_text())
        raise ValueError("Constant benchmark cache has different users/config/seeds; use a new output directory")
    users = load_users(initial_users)
    factory = gpu_episode_factory(config, maxsize=max(1, len(train_seeds)), grid_cache_size=1024)
    rows = []
    for seed in train_seeds:
        episode = factory(int(seed))
        for index in range(catalogue.n_actions):
            metrics = play_menu(episode, catalogue, index, users)
            rows.append(dict(seed=int(seed), menu=index, menu_label=catalogue.labels[index],
                             **{key: value for key, value in metrics.items() if key != "seed"}))
        print(json.dumps({"constant_search_training_seed": int(seed), "menus": catalogue.n_actions}), flush=True)
    write_csv(output / "training_menu_outcomes.csv", rows)
    summaries = []
    for index in range(catalogue.n_actions):
        cases = [row for row in rows if row["menu"] == index]
        stats = number_means(cases)
        stats.update(menu=index, menu_label=catalogue.labels[index],
                     all_work_and_user_ir=all(score_complete(row) for row in cases))
        summaries.append(stats)
    selected = {}
    for eta in (0.0, 100.0):
        eligible = [row for row in summaries if row["all_work_and_user_ir"]]
        if not eligible:
            raise RuntimeError("No complete-work, user-participating constant menu, including original, was found")
        # The same physical outcomes apply at either eta: users do not observe
        # the firm's internal preference and raw posted menus are signal-free.
        def target(row):
            return row["delta_platform_profit_yuan"] - eta * row["delta_dc_consumption_emissions_t"]
        best = max(eligible, key=lambda row: (target(row), -row["menu"]))
        selected[str(int(eta))] = dict(
            menu=int(best["menu"]), menu_label=best["menu_label"],
            eta_yuan_per_t=eta, training_mean_objective_gain_yuan=float(target(best)),
            training_mean_cash_gain_yuan=best["delta_platform_profit_yuan"],
            training_mean_dc_footprint_reduction_t=-best["delta_dc_consumption_emissions_t"],
            eligible_menus=len(eligible), searched_menus=catalogue.n_actions,
        )
    write_csv(output / "training_menu_means.csv", summaries)
    write_json(manifest_path, manifest)
    write_json(result_path, selected)
    return selected


def enrich(metrics, *, experiment, phase, seed, menu, menu_label):
    """Keep original accounting columns and add explicit comparison measures."""
    row = dict(metrics, experiment=experiment, phase=phase, seed=int(seed),
               scheme=f"{experiment}__{phase}", menu=int(menu), menu_label=menu_label)
    saving = -row["delta_system_cost_yuan"]
    row.update(
        cash_profit_before_fixed_yuan=row["platform_profit_before_fixed_cost_yuan"],
        cash_profit_after_fixed_yuan=row["platform_profit_yuan"],
        cash_profit_change_yuan=row["delta_platform_profit_yuan"],
        own_objective_change_yuan=row["delta_platform_objective_yuan"],
        dispatch_cost_saving_yuan=saving,
        dispatch_cost_saving_rate=saving / row["baseline_dispatch_system_cost_yuan"],
        full_cost_saving_yuan=-row["delta_full_system_cost_yuan"],
        full_cost_saving_rate=-row["delta_full_system_cost_yuan"] / row["baseline_full_system_cost_yuan"],
        incremental_cost_saving_yuan=-row["delta_incremental_system_cost_yuan"],
        incremental_cost_saving_rate=(-row["delta_incremental_system_cost_yuan"]
                                      / row["baseline_incremental_system_cost_yuan"]),
        actual_grid_emissions_change_t=row["delta_carbon_emissions_t"],
        allocated_dc_footprint_change_t=row["delta_dc_consumption_emissions_t"],
        all_users_participate=row["user_ir_rate"] >= 1 - 1e-9,
        platform_cash_nonloss=row["delta_platform_profit_yuan"] >= -1e-6,
        platform_objective_nonloss=row["delta_platform_objective_yuan"] >= -1e-6,
    )
    row["joint_cash_and_resource_benefit"] = bool(
        score_complete(row) and row["platform_cash_nonloss"] and saving > 1e-6)
    row["joint_objective_and_resource_benefit"] = bool(
        score_complete(row) and row["platform_objective_nonloss"] and saving > 1e-6)
    return row


def evaluate_policy(factory, catalogue, observer, users, platform, seeds, experiment, phase, constant=None):
    rows, actions, contracts = [], [], []
    bundle = PolicyBundle(platform=platform, users=users, platform_codec=catalogue,
                          platform_observer=observer, training=False)
    for seed in seeds:
        episode = factory(int(seed))
        menu = (int(constant) if constant is not None else
                int(np.asarray(platform.predict(bundle.platform_observation(episode), deterministic=True)[0]).item()))
        episode.set_platform_action(catalogue.decode(episode, menu))
        while not episode.done:
            _, _, info = episode.step_user(bundle.user_action(episode))
            contracts.append(dict(experiment=experiment, phase=phase, seed=int(seed), menu=menu, **info))
        metrics = episode.outcome().metrics
        rows.append(enrich(metrics, experiment=experiment, phase=phase, seed=seed,
                           menu=menu, menu_label=catalogue.labels[menu]))
        actions.append(dict(experiment=experiment, phase=phase, seed=int(seed), menu=menu,
                            menu_label=catalogue.labels[menu], raw_menu=episode.platform_action.tolist(),
                            tariffs_yuan_per_gpu_hour=episode.tariff_yuan_per_cu.tolist()))
    return rows, actions, contracts


def original_reference(factory, catalogue, seeds, experiment):
    rows = []
    for seed in seeds:
        episode = factory(int(seed))
        episode.set_platform_action(catalogue.decode(episode, 0))
        while not episode.done:
            episode.step_user(0)
        rows.append(enrich(episode.outcome().metrics, experiment=experiment, phase="original_contracts",
                           seed=seed, menu=0, menu_label=catalogue.labels[0]))
    return rows


def summarise(rows):
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["scheme", "experiment", "phase"], sort=False).mean(numeric_only=True).reset_index()
    pooled = []
    for scheme, group in frame.groupby("scheme", sort=False):
        pooled.append(dict(
            scheme=scheme, days=len(group),
            pooled_dispatch_cost_saving_rate=float(group.dispatch_cost_saving_yuan.sum()
                                                  / group.baseline_dispatch_system_cost_yuan.sum()),
            pooled_full_cost_saving_rate=float(group.full_cost_saving_yuan.sum()
                                              / group.baseline_full_system_cost_yuan.sum()),
            pooled_incremental_cost_saving_rate=float(group.incremental_cost_saving_yuan.sum()
                                                     / group.baseline_incremental_system_cost_yuan.sum()),
            all_users_participating_days=int(group.all_users_participate.sum()),
            platform_cash_nonloss_days=int(group.platform_cash_nonloss.sum()),
            platform_objective_nonloss_days=int(group.platform_objective_nonloss.sum()),
            all_jobs_complete_days=int((group.completion_rate >= 1 - 1e-9).sum()),
            deadlines_compliant_days=int((group.deadline_compliance >= 1 - 1e-9).sum()),
            resource_and_cash_benefit_days=int(group.joint_cash_and_resource_benefit.sum()),
            resource_and_objective_benefit_days=int(group.joint_objective_and_resource_benefit.sum()),
            grid_emissions_reducing_days=int((group.grid_emissions_reduction_t > 1e-7).sum()),
            cash_profit_after_fixed_positive_days=int((group.cash_profit_after_fixed_yuan >= -1e-7).sum()),
            distinct_menus=int(group.menu.nunique()), original_menu_days=int((group.menu == 0).sum()),
        ))
    return summary.merge(pd.DataFrame(pooled), on="scheme", validate="one_to_one")


def information_comparison(rows):
    frame = pd.DataFrame(rows)
    left = frame[(frame.experiment == "electricity_only") & (frame.phase == "ppo_fixed_users")].set_index("seed")
    right = frame[(frame.experiment == "electricity_carbon") & (frame.phase == "ppo_fixed_users")].set_index("seed")
    if set(left.index) != set(right.index):
        raise ValueError("Information ablation needs identical held-out days")
    columns = ("cash_profit_change_yuan", "own_objective_change_yuan", "dispatch_cost_saving_yuan",
               "full_cost_saving_yuan", "grid_emissions_reduction_t", "dc_footprint_reduction_t",
               "migrated_jobs", "delayed_jobs")
    paired = []
    for seed in left.index:
        paired.append(dict(seed=int(seed), **{
            f"carbon_information_minus_electricity_information__{key}": float(right.loc[seed, key] - left.loc[seed, key])
            for key in columns}))
    return paired


def bootstrap_information_effect(paired, *, repetitions=10000, seed=0):
    """Paired day resampling, conditional on these fixed trained policies.

    This quantifies sampled-scenario uncertainty only. One algorithm training
    seed does not identify variability across independently trained algorithms.
    """
    if not paired:
        raise ValueError("At least one paired scenario is required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(paired), size=(repetitions, len(paired)))
    rows = []
    for metric in paired[0]:
        if metric == "seed":
            continue
        values = np.asarray([row[metric] for row in paired], dtype=float)
        sampled_means = values[indices].mean(axis=1)
        lower, upper = np.quantile(sampled_means, [.025, .975])
        rows.append(dict(
            metric=metric, mean_paired_difference=float(values.mean()),
            mean_ci_95_lower=float(lower), mean_ci_95_upper=float(upper),
            scenario_pairs=len(paired), bootstrap_repetitions=repetitions,
            bootstrap_resampling_seed=seed, positive_difference_days=int((values > 1e-7).sum()),
            negative_difference_days=int((values < -1e-7).sum()),
            zero_difference_days=int((np.abs(values) <= 1e-7).sum()),
        ))
    return rows


def load_config(directory, eta):
    summary = json.loads((directory / "summary.json").read_text())
    config = GpuMarketConfig(**summary["config"])
    if not np.isclose(config.internal_carbon_value_yuan_per_t, eta):
        raise ValueError("Saved experiment preference disagrees with the declared ablation")
    return config, summary


def revalue_metrics(metrics, eta):
    """Change the firm's preference accounting without changing any cash flow."""
    row = dict(metrics)
    row["internal_carbon_value_yuan_per_t"] = float(eta)
    row["internal_carbon_valuation_yuan"] = float(eta * row["dc_consumption_emissions_t"])
    row["platform_objective_yuan"] = row["platform_profit_yuan"] - row["internal_carbon_valuation_yuan"]
    row["baseline_platform_objective_yuan"] = (row["baseline_platform_profit_yuan"]
                                               - eta * row["baseline_dc_consumption_emissions_t"])
    row["delta_platform_objective_yuan"] = row["platform_objective_yuan"] - row["baseline_platform_objective_yuan"]
    row["platform_objective_participates"] = row["delta_platform_objective_yuan"] >= -1e-6
    return row


def finite_menu_potential(output, config, initial_users, seeds):
    """Daywise hindsight reference within the disclosed finite 77-menu family.

    Learned users are frozen. No policy receives candidate outcomes and no
    learned parameter or scenario parameter is selected using this reference.
    Cost references impose a cash-profit floor; own-objective ceilings permit
    the same cash/carbon preference tradeoff as the relevant platform agent.
    """
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(json.dumps(dict(
        config=asdict(config), users_sha256=user_hashes(initial_users),
        test_scenario_seeds=list(seeds), n_menus=77,
        scope="daywise finite-menu hindsight with frozen learned responses; no global Pareto or algorithm optimality claim")))
    cache, manifest_path = output / "summary.json", output / "manifest.json"
    if cache.exists() and manifest_path.exists():
        if json.loads(manifest_path.read_text()) == manifest:
            return json.loads(cache.read_text())
        raise ValueError("Finite-menu reference cache has different users/config/scenarios; use a new output directory")
    users = load_users(initial_users)
    catalogue = FixedGpuTariffCatalogue(config)
    factory = gpu_episode_factory(config, maxsize=max(1, len(seeds)), grid_cache_size=1024)
    candidate_rows, selected_rows = [], []
    for seed in seeds:
        episode = factory(int(seed))
        candidates = []
        for menu in range(catalogue.n_actions):
            metrics = play_menu(episode, catalogue, menu, users)
            candidates.append((menu, metrics))
            candidate_rows.append(dict(menu=menu, menu_label=catalogue.labels[menu], **metrics))
        individually_feasible = [(menu, metrics) for menu, metrics in candidates if score_complete(metrics)]
        cash_feasible = [(menu, metrics) for menu, metrics in individually_feasible
                         if metrics["delta_platform_profit_yuan"] >= -1e-6]
        cash_carbon_feasible = [(menu, metrics) for menu, metrics in cash_feasible
                                if metrics["grid_emissions_reduction_t"] >= -1e-7]
        if not cash_feasible or not cash_carbon_feasible:
            raise RuntimeError("Original contracts must be present in both constrained reference sets")
        references = []
        for label, eligible in (
            ("hindsight_cost_cash_nonloss", cash_feasible),
            ("hindsight_cost_cash_nonloss_grid_emissions_nonincrease", cash_carbon_feasible),
        ):
            menu, metrics = max(eligible, key=lambda item: (-item[1]["delta_system_cost_yuan"], -item[0]))
            references.append((label, 100.0, menu, metrics, len(eligible)))
        for eta in (0.0, 100.0):
            menu, metrics = max(individually_feasible, key=lambda item: (
                item[1]["delta_platform_profit_yuan"] - eta * item[1]["delta_dc_consumption_emissions_t"],
                -item[0]))
            references.append((f"hindsight_own_objective_eta{int(eta)}", eta, menu, metrics,
                               len(individually_feasible)))
        for label, eta, menu, metrics, feasible_count in references:
            row = enrich(revalue_metrics(metrics, eta), experiment=f"finite_menu_eta{int(eta)}",
                         phase=label, seed=seed, menu=menu, menu_label=catalogue.labels[menu])
            row.update(reference=label, feasible_menus=feasible_count, hindsight_complete_information=True)
            selected_rows.append(row)
        print(json.dumps({"finite_menu_hindsight_seed": int(seed), "menus": len(candidates)}), flush=True)
    write_csv(output / "test_menu_outcomes.csv", candidate_rows)
    write_csv(output / "selected_references_by_day.csv", selected_rows)
    means = summarise(selected_rows)
    means.to_csv(output / "reference_mean.csv", index=False, encoding="utf-8-sig")
    report = dict(
        status="completed", manifest=manifest, reference_mean=means.to_dict(orient="records"),
        notes=dict(
            cost_cash_nonloss="maximum system resource-cost saving among menus preserving all users' IR, full work and cash profit no worse than original",
            cost_cash_nonloss_grid_emissions_nonincrease="same constraints plus actual total grid emissions no greater than original",
            own_objective_ceiling="maximum own objective over complete-work/user-IR menus; no extra cash floor at eta=100",
            algorithm_comparison="hindsight complete-information references, not deployable policies and not used for model/parameter selection",
            finite_scope="77 disclosed raw menus and the same three deterministic initial learned users; not a global bilevel or Pareto optimum"),
    )
    write_json(manifest_path, manifest)
    write_json(cache, report)
    return report


def potential_capture_comparison(rows, potential):
    """Only attach a potential fraction when all corresponding day constraints hold."""
    frame = pd.DataFrame(rows)
    references = {row["phase"]: row for row in potential["reference_mean"]}
    capture = []
    for experiment in EXPERIMENTS:
        group = frame[(frame.experiment == experiment) & (frame.phase == "ppo_fixed_users")]
        cash_comparable = bool(group.all_users_participate.all() and group.platform_cash_nonloss.all())
        carbon_comparable = bool(cash_comparable and (group.grid_emissions_reduction_t >= -1e-7).all())
        resource_reference = references["hindsight_cost_cash_nonloss"]
        low_carbon_reference = references["hindsight_cost_cash_nonloss_grid_emissions_nonincrease"]
        eta = EXPERIMENTS[experiment][0]
        objective_reference = references[f"hindsight_own_objective_eta{int(eta)}"]
        def fraction(numerator, denominator, comparable=True):
            return float(numerator / denominator) if comparable and denominator > 1e-7 else None
        capture.append(dict(
            experiment=experiment, phase="ppo_fixed_users", test_days=len(group),
            ppo_mean_resource_saving_yuan=float(group.dispatch_cost_saving_yuan.mean()),
            cash_nonloss_reference_mean_resource_saving_yuan=resource_reference["dispatch_cost_saving_yuan"],
            cash_nonloss_grid_emissions_reference_mean_resource_saving_yuan=low_carbon_reference["dispatch_cost_saving_yuan"],
            ppo_days_meeting_cash_reference_constraints=int((group.all_users_participate & group.platform_cash_nonloss).sum()),
            ppo_days_meeting_cash_and_carbon_reference_constraints=int((group.all_users_participate & group.platform_cash_nonloss
                                                                         & (group.grid_emissions_reduction_t >= -1e-7)).sum()),
            cash_reference_fraction_comparable=cash_comparable,
            cash_and_carbon_reference_fraction_comparable=carbon_comparable,
            resource_potential_fraction=fraction(group.dispatch_cost_saving_yuan.mean(),
                                                 resource_reference["dispatch_cost_saving_yuan"], cash_comparable),
            low_carbon_resource_potential_fraction=fraction(group.dispatch_cost_saving_yuan.mean(),
                                                            low_carbon_reference["dispatch_cost_saving_yuan"], carbon_comparable),
            ppo_mean_own_objective_gain_yuan=float(group.own_objective_change_yuan.mean()),
            own_objective_reference_mean_gain_yuan=objective_reference["own_objective_change_yuan"],
            own_objective_reference_fraction=fraction(group.own_objective_change_yuan.mean(),
                                                       objective_reference["own_objective_change_yuan"],
                                                       bool(group.all_users_participate.all())),
            interpretation="Resource fractions withheld if any PPO day violates that reference's constraints; all days remain in reported performance",
        ))
    return capture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("constants", "menu-potential", "full"), default="full")
    parser.add_argument("--train-days", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=30000)
    parser.add_argument("--days", type=int, default=32)
    parser.add_argument("--skip-joint", action="store_true")
    parser.add_argument("--fit-joint-constants", action="store_true")
    parser.add_argument("--physical-precheck", type=Path)
    parser.add_argument("--include-menu-potential", action="store_true")
    args = parser.parse_args(argv)
    if args.train_days < 1 or args.days < 1:
        parser.error("Positive train-days and days are required")
    torch.set_num_threads(1)
    run_root, output = args.run_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    initial_users = run_root / "initial_users"
    config = GpuMarketConfig(internal_carbon_value_yuan_per_t=100.0)
    seeds = tuple(range(args.seed_start, args.seed_start + args.days))
    if args.stage == "menu-potential":
        potential = finite_menu_potential(output / "finite_menu_potential", config, initial_users, seeds)
        print(json.dumps({"finite_menu_potential": potential["reference_mean"],
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
        return
    constants = fit_constants(output / "constant_train", config, initial_users, tuple(range(args.train_days)))
    if args.stage == "constants":
        print(json.dumps({"selected_constants": constants, "elapsed_seconds": time.monotonic() - started}), flush=True)
        return
    users = load_users(initial_users)
    all_rows, all_actions, all_contracts, manifests = [], [], [], {}
    physical_config = None
    for experiment, (eta, aware) in EXPERIMENTS.items():
        directory = run_root / experiment
        cfg, training_summary = load_config(directory, eta)
        signature = asdict(replace(cfg, internal_carbon_value_yuan_per_t=100.0))
        if physical_config is not None and signature != physical_config:
            raise ValueError("Preference/information experiments changed physical or economic inputs")
        physical_config = signature
        if training_summary["carbon_aware"] != aware:
            raise ValueError("Saved experiment carbon-observation flag disagrees with the declared ablation")
        catalogue = FixedGpuTariffCatalogue(cfg)
        observer = GpuTariffObserver(catalogue, carbon_aware=aware)
        factory = gpu_episode_factory(cfg, maxsize=args.days, grid_cache_size=1024)
        all_rows.extend(original_reference(factory, catalogue, seeds, experiment))
        menu = constants[str(int(eta))]["menu"]
        phases = [("constant_fixed_users", None, users, menu),
                  ("ppo_fixed_users", PPO.load(directory / "best_fixed/platform.zip", device="cpu"), users, None)]
        if not args.skip_joint:
            joint_users = load_users(directory / "checkpoints")
            phases.append(("ppo_joint_users", PPO.load(directory / "checkpoints/platform.zip", device="cpu"), joint_users, None))
            if args.fit_joint_constants:
                joint_constants = fit_constants(output / "joint_constant_train" / experiment, cfg,
                                                directory / "checkpoints", tuple(range(args.train_days)))
                phases.append(("constant_joint_users", None, joint_users, joint_constants[str(int(eta))]["menu"]))
        for phase, platform, response_users, constant in phases:
            rows, actions, contracts = evaluate_policy(factory, catalogue, observer, response_users, platform,
                                                       seeds, experiment, phase, constant=constant)
            all_rows.extend(rows); all_actions.extend(actions); all_contracts.extend(contracts)
        manifests[experiment] = dict(config=asdict(cfg), carbon_aware=aware,
                                    fixed_users_sha256=user_hashes(initial_users),
                                    joint_users_sha256=(user_hashes(directory / "checkpoints") if not args.skip_joint else None),
                                    selected_constant=constants[str(int(eta))],
                                    selected_platform_steps=training_summary["selected_platform_steps"])
        print(json.dumps({"evaluated_experiment": experiment, "test_days": args.days}), flush=True)
    write_csv(output / "comparison_by_day.csv", all_rows)
    summary = summarise(all_rows)
    summary.to_csv(output / "comparison_mean.csv", index=False, encoding="utf-8-sig")
    write_csv(output / "actions.csv", all_actions)
    write_csv(output / "contracts.csv", all_contracts)
    paired = information_comparison(all_rows)
    write_csv(output / "paired_information_ablation.csv", paired)
    intervals = bootstrap_information_effect(paired)
    write_csv(output / "paired_information_bootstrap.csv", intervals)
    potential, potential_capture = None, []
    if args.include_menu_potential:
        potential = finite_menu_potential(output / "finite_menu_potential", config, initial_users, seeds)
        potential_capture = potential_capture_comparison(all_rows, potential)
        write_csv(output / "finite_menu_potential" / "ppo_potential_capture.csv", potential_capture)
    precheck = args.physical_precheck
    if precheck is None:
        precheck = run_root.parents[2] / "work/gpu-potential-precheck.json"
    if precheck.exists():
        diagnostic = output / "physical_diagnostic"; diagnostic.mkdir(exist_ok=True)
        shutil.copy2(precheck, diagnostic / precheck.name)
        if precheck.with_suffix(".txt").exists():
            shutil.copy2(precheck.with_suffix(".txt"), diagnostic / precheck.with_suffix(".txt").name)
    report = dict(
        status="completed", algorithm_training_seed=0, elapsed_seconds=time.monotonic() - started,
        training_scenario_seeds=list(range(args.train_days)), test_scenario_seeds=list(seeds),
        primary_comparison="electricity_only versus electricity_carbon, fixed identical learned users, eta=100",
        preference_reference="cash_only changes eta to zero; not a pure observation ablation",
        constant_benchmark="one signal-independent raw menu selected on training scenarios, frozen before test",
        joint_user_comparison="secondary; user parameters can differ after independent joint updates",
        physical_diagnostic="separate 8-day scale/physical potential diagnostic; never policy/model selection",
        accounting=dict(
            dispatch_cost="all-grid fuel resource plus variable compute/network operations plus user inconvenience",
            full_cost="dispatch cost plus fixed installed-GPU capacity cost",
            incremental_cost="dispatch cost minus same-day background-only grid resource cost",
            carbon_policy="generation-side price already enters electricity bills",
            internal_carbon_value="voluntary preference deduction in objective, no cash payment",
            actual_emissions="total grid generator emissions after re-clearing",
            consumption_footprint="flow-traced allocation to DC electricity; does not by itself establish avoided emissions"),
        experiments=manifests, mean_paired_information_effect={
            key: value for key, value in number_means(paired).items() if key != "seed"},
        paired_information_bootstrap=intervals,
        finite_menu_potential=(dict(reference_mean=potential["reference_mean"],
                                   ppo_comparison=potential_capture) if potential is not None else None),
        bootstrap_interpretation=("paired scenario-day resampling, 10000 replicates, RNG seed zero; "
                                  "conditional on fixed trained models, not multi-training-seed robustness"),
        comparison_mean=summary.to_dict(orient="records"),
    )
    write_json(output / "summary.json", report)
    print(json.dumps({"completed": {"output": str(output), "elapsed_seconds": report["elapsed_seconds"],
                                    "paired_information_effect": report["mean_paired_information_effect"]}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
