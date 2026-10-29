#!/usr/bin/env python3
"""Evaluate conditional rebates and signed resource-sharing settlements.

All four arms use identical physical days, the same initial learned users,
and the same 77 raw posted-price actions. Constant menus are selected on
training days only. The primary frozen-user comparisons separate the rebate
qualification rule, resource sharing, and actual-grid-carbon qualification.
Joint-training outcomes are reported separately because opponent weights can
differ. Monetary transfers never enter resource system-cost savings.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from energy_net.coordination.flex_market import FlexMarketConfig, flex_episode_factory
from energy_net.coordination.flex_tariffs import FlexTariffObserver
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue
from energy_net.coordination.learning import PolicyBundle


EXPERIMENTS = {
    "published_no_share": dict(conditional_rebates=False, sharing_enabled=False, carbon_eligibility=False),
    "conditional_no_share": dict(conditional_rebates=True, sharing_enabled=False, carbon_eligibility=False),
    "conditional_share": dict(conditional_rebates=True, sharing_enabled=True, carbon_eligibility=False),
    "conditional_green_share": dict(conditional_rebates=True, sharing_enabled=True, carbon_eligibility=True),
}

COMPARISONS = (
    ("conditional_contract", "published_no_share", "conditional_no_share"),
    ("resource_sharing", "conditional_no_share", "conditional_share"),
    ("carbon_qualification", "conditional_share", "conditional_green_share"),
    ("complete_mechanism", "conditional_no_share", "conditional_green_share"),
)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def number_means(rows):
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows]))
            for key, value in rows[0].items()
            if isinstance(value, (int, float, bool, np.number))}


def load_users(directory):
    return {user: PPO.load(directory / f"user_{user}.zip", device="cpu") for user in range(3)}


def user_hashes(directory):
    return {str(user): hashlib.sha256((directory / f"user_{user}.zip").read_bytes()).hexdigest()
            for user in range(3)}


def complete_and_user_ir(row):
    return bool(row["completion_rate"] >= 1 - 1e-9
                and row["deadline_compliance"] >= 1 - 1e-9
                and row["user_ir_rate"] >= 1 - 1e-9)


def declared_config(experiment):
    return FlexMarketConfig(internal_carbon_value_yuan_per_t=0., resource_share=.8,
                            **EXPERIMENTS[experiment])


def load_config(directory, experiment):
    report = json.loads((directory / "summary.json").read_text())
    config = FlexMarketConfig(**report["config"])
    for field, expected in EXPERIMENTS[experiment].items():
        if getattr(config, field) != expected:
            raise ValueError(f"Saved {experiment} disagrees with declared mechanism field {field}")
    if config.internal_carbon_value_yuan_per_t != 0:
        raise ValueError("All mechanism arms must maximize cash profit with eta zero")
    return config, report


def physical_signature(config):
    fields = asdict(config)
    for field in ("conditional_rebates", "sharing_enabled", "carbon_eligibility",
                  "resource_share", "carbon_tolerance_t"):
        fields.pop(field)
    return json.loads(json.dumps(fields))


def play_menu(episode, catalogue, index, users):
    episode.reset(catalogue.decode(episode, index))
    return episode.play(users).metrics


def fit_constant(output, config, initial_users, train_seeds):
    """Select an implementable constant using training cash returns alone."""
    catalogue = FixedGpuTariffCatalogue(config)
    manifest = json.loads(json.dumps(dict(
        config=asdict(config), initial_user_sha256=user_hashes(initial_users),
        training_scenario_seeds=list(train_seeds), catalogue_labels=list(catalogue.labels),
        selection="highest training-day mean post-settlement cash profit; complete work and all user IR; ties lower menu index")))
    cache, manifest_path = output / "selected_constant.json", output / "manifest.json"
    if cache.exists() and manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Constant cache has different config, initial users or training days; use a new output directory")
        return json.loads(cache.read_text())
    users = load_users(initial_users)
    factory = flex_episode_factory(config, maxsize=max(1, len(train_seeds)), grid_cache_size=1024)
    rows = []
    for seed in train_seeds:
        episode = factory(int(seed))
        for index in range(catalogue.n_actions):
            metrics = play_menu(episode, catalogue, index, users)
            rows.append(dict(menu=index, menu_label=catalogue.labels[index], **metrics))
        print(json.dumps({"constant_training_seed": int(seed), "conditional": config.conditional_rebates,
                          "sharing": config.sharing_enabled, "green": config.carbon_eligibility}), flush=True)
    summaries = []
    for index in range(catalogue.n_actions):
        group = [row for row in rows if row["menu"] == index]
        summaries.append(dict(number_means(group), menu=index, menu_label=catalogue.labels[index],
                              all_work_and_user_ir=all(complete_and_user_ir(row) for row in group)))
    eligible = [row for row in summaries if row["all_work_and_user_ir"]]
    if not eligible:
        raise RuntimeError("The original menu must preserve all work and user participation")
    best = max(eligible, key=lambda row: (row["delta_platform_profit_yuan"], -row["menu"]))
    selected = dict(menu=int(best["menu"]), menu_label=best["menu_label"],
                    training_mean_cash_gain_yuan=best["delta_platform_profit_yuan"],
                    training_mean_cost_saving_yuan=-best["delta_system_cost_yuan"],
                    training_mean_actual_grid_emissions_reduction_t=best["grid_emissions_reduction_t"],
                    eligible_menus=len(eligible), searched_menus=catalogue.n_actions)
    write_csv(output / "training_menu_outcomes.csv", rows)
    write_csv(output / "training_menu_means.csv", summaries)
    write_json(manifest_path, manifest)
    write_json(cache, selected)
    return selected


def enrich(metrics, experiment, phase, seed, menu, label):
    row = dict(metrics, experiment=experiment, phase=phase, scheme=f"{experiment}__{phase}",
               seed=int(seed), menu=int(menu), menu_label=label)
    cost_saving = -row["delta_system_cost_yuan"]
    # All carbon-related cash effects already in electricity bills remain
    # visible before the signed procurement adjustment. The model resource
    # account is separately identified; it is not called an actual ISO profit.
    row.update(
        dispatch_cost_saving_yuan=cost_saving,
        dispatch_cost_saving_rate=cost_saving / row["baseline_dispatch_system_cost_yuan"],
        full_cost_saving_yuan=-row["delta_full_system_cost_yuan"],
        full_cost_saving_rate=-row["delta_full_system_cost_yuan"] / row["baseline_full_system_cost_yuan"],
        incremental_cost_saving_yuan=-row["delta_incremental_system_cost_yuan"],
        incremental_cost_saving_rate=-row["delta_incremental_system_cost_yuan"] / row["baseline_incremental_system_cost_yuan"],
        cash_profit_before_fixed_yuan=row["platform_profit_before_fixed_cost_yuan"],
        cash_profit_after_fixed_yuan=row["platform_profit_yuan"],
        cash_profit_before_transfer_change_yuan=row["cash_change_before_transfer_yuan"],
        cash_profit_change_yuan=row["delta_platform_profit_yuan"],
        own_objective_change_yuan=row["delta_platform_objective_yuan"],
        actual_grid_emissions_change_t=row["delta_carbon_emissions_t"],
        allocated_dc_footprint_change_t=row["delta_dc_consumption_emissions_t"],
        all_users_participate=row["user_ir_rate"] >= 1 - 1e-9,
        platform_cash_nonloss=row["delta_platform_profit_yuan"] >= -1e-6,
        purchaser_nonloss=row["buyer_post_settlement_benefit_yuan"] >= -1e-6,
        actual_emissions_nonincrease=row["grid_emissions_reduction_t"] >= -1e-7,
        surplus_identity_absolute_error_yuan=abs(row["surplus_identity_error_yuan"]),
    )
    row["resource_platform_user_buyer_benefit"] = bool(
        complete_and_user_ir(row) and row["platform_cash_nonloss"]
        and row["purchaser_nonloss"] and cost_saving > 1e-6)
    row["resource_platform_user_buyer_carbon_benefit"] = bool(
        row["resource_platform_user_buyer_benefit"] and row["actual_emissions_nonincrease"])
    if not np.isclose(row["delta_platform_profit_yuan"], row["cash_change_after_transfer_yuan"], atol=1e-6, rtol=0):
        raise ValueError("Post-settlement cash change does not match paired platform profit")
    if not np.isclose(cost_saving, row["contract_system_cost_saving_yuan"], atol=1e-6, rtol=0):
        raise ValueError("Settlement accounting changed resource system cost")
    return row


def evaluate_policy(factory, catalogue, observer, users, platform, seeds, experiment, phase, constant=None):
    bundle = PolicyBundle(platform=platform, users=users, platform_codec=catalogue,
                          platform_observer=observer, training=False)
    rows, actions, contracts = [], [], []
    for seed in seeds:
        episode = factory(int(seed))
        index = (int(constant) if constant is not None else
                 int(np.asarray(platform.predict(bundle.platform_observation(episode), deterministic=True)[0]).item()))
        episode.set_platform_action(catalogue.decode(episode, index))
        while not episode.done:
            _, _, info = episode.step_user(bundle.user_action(episode))
            contracts.append(dict(experiment=experiment, phase=phase, seed=int(seed), menu=index, **info))
        rows.append(enrich(episode.outcome().metrics, experiment, phase, seed, index, catalogue.labels[index]))
        actions.append(dict(experiment=experiment, phase=phase, seed=int(seed), menu=index,
                            menu_label=catalogue.labels[index], raw_menu=episode.platform_action.tolist(),
                            tariffs_yuan_per_gpu_hour=episode.tariff_yuan_per_cu.tolist(),
                            conditional_rebates=bool(episode.config.conditional_rebates)))
    return rows, actions, contracts


def original_reference(factory, catalogue, seeds, experiment):
    rows = []
    for seed in seeds:
        episode = factory(int(seed))
        episode.set_platform_action(catalogue.decode(episode, 0))
        while not episode.done:
            episode.step_user(0)
        rows.append(enrich(episode.outcome().metrics, experiment, "original_contracts", seed, 0,
                           catalogue.labels[0]))
    return rows


def summarise(rows):
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["scheme", "experiment", "phase"], sort=False).mean(numeric_only=True).reset_index()
    totals = []
    for scheme, group in frame.groupby("scheme", sort=False):
        totals.append(dict(
            scheme=scheme, days=len(group),
            pooled_dispatch_cost_saving_rate=float(group.dispatch_cost_saving_yuan.sum() / group.baseline_dispatch_system_cost_yuan.sum()),
            pooled_full_cost_saving_rate=float(group.full_cost_saving_yuan.sum() / group.baseline_full_system_cost_yuan.sum()),
            pooled_incremental_cost_saving_rate=float(group.incremental_cost_saving_yuan.sum() / group.baseline_incremental_system_cost_yuan.sum()),
            all_users_participating_days=int(group.all_users_participate.sum()),
            platform_cash_nonloss_days=int(group.platform_cash_nonloss.sum()),
            purchaser_nonloss_days=int(group.purchaser_nonloss.sum()),
            actual_emissions_nonincrease_days=int(group.actual_emissions_nonincrease.sum()),
            carbon_eligible_days=int(group.carbon_eligible.sum()),
            grid_emissions_reducing_days=int((group.grid_emissions_reduction_t > 1e-7).sum()),
            all_jobs_complete_days=int((group.completion_rate >= 1 - 1e-9).sum()),
            deadlines_compliant_days=int((group.deadline_compliance >= 1 - 1e-9).sum()),
            resource_platform_user_buyer_benefit_days=int(group.resource_platform_user_buyer_benefit.sum()),
            resource_platform_user_buyer_carbon_benefit_days=int(group.resource_platform_user_buyer_carbon_benefit.sum()),
            positive_settlement_days=int((group.flexibility_settlement_yuan > 1e-7).sum()),
            negative_settlement_days=int((group.flexibility_settlement_yuan < -1e-7).sum()),
            max_surplus_identity_absolute_error_yuan=float(group.surplus_identity_absolute_error_yuan.max()),
            cash_profit_after_fixed_positive_days=int((group.cash_profit_after_fixed_yuan >= -1e-7).sum()),
            distinct_menus=int(group.menu.nunique()), original_menu_days=int((group.menu == 0).sum())))
    return summary.merge(pd.DataFrame(totals), on="scheme", validate="one_to_one")


def paired_comparisons(rows):
    frame = pd.DataFrame(rows)
    metrics = ("dispatch_cost_saving_yuan", "cash_profit_change_yuan", "cash_profit_before_transfer_change_yuan",
               "buyer_post_settlement_benefit_yuan", "user_gain_yuan", "user_rebate_yuan",
               "flexibility_settlement_yuan", "grid_emissions_reduction_t", "dc_footprint_reduction_t",
               "migrated_jobs", "delayed_jobs")
    pairs = []
    for label, before, after in COMPARISONS:
        left = frame[(frame.experiment == before) & (frame.phase == "ppo_fixed_users")].set_index("seed")
        right = frame[(frame.experiment == after) & (frame.phase == "ppo_fixed_users")].set_index("seed")
        if left.empty or right.empty:
            continue
        if set(left.index) != set(right.index):
            raise ValueError("Mechanism comparisons require identical test days")
        for seed in left.index:
            pairs.append(dict(comparison=label, before=before, after=after, seed=int(seed),
                              **{metric: float(right.loc[seed, metric] - left.loc[seed, metric]) for metric in metrics}))
    return pairs


def bootstrap_pairs(pairs, repetitions=10000, seed=0):
    """Paired test-day uncertainty with fixed trained models, not new training seeds."""
    if not pairs:
        return []
    frame = pd.DataFrame(pairs)
    results = []
    for comparison, group in frame.groupby("comparison", sort=False):
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(group), size=(repetitions, len(group)))
        for metric in group.select_dtypes(include=np.number).columns:
            if metric == "seed":
                continue
            values = group[metric].to_numpy(float)
            sampled = values[indices].mean(axis=1)
            lower, upper = np.quantile(sampled, [.025, .975])
            results.append(dict(comparison=comparison, metric=metric, mean_paired_difference=float(values.mean()),
                                mean_ci_95_lower=float(lower), mean_ci_95_upper=float(upper),
                                scenario_pairs=len(group), bootstrap_repetitions=repetitions,
                                bootstrap_resampling_seed=seed, positive_difference_days=int((values > 1e-7).sum()),
                                negative_difference_days=int((values < -1e-7).sum()),
                                zero_difference_days=int((np.abs(values) <= 1e-7).sum())))
    return results


def fixed_plan_share_sensitivity(rows, contracts, shares=(.5, .8, 1.)):
    """Reallocate one realised plan without changing policy actions or dispatch.

    This is a distributional calculation, not a re-optimised policy at each
    share. Every row uses exactly the same accepted jobs, posted invoices,
    generator dispatch, inconvenience, and resource saving as its source run.
    It must not be used to select an allegedly optimal sharing parameter.
    """
    if any(not np.isfinite(share) or not 0 <= share <= 1 for share in shares):
        raise ValueError("Fixed-plan sharing fractions must be finite and in [0,1]")
    contract_index = {}
    for contract in contracts:
        key = (contract["experiment"], contract["phase"], int(contract["seed"]))
        contract_index.setdefault(key, []).append(contract)
    results = []
    for source in rows:
        if (source["experiment"] != "conditional_green_share"
                or source["phase"] not in ("ppo_fixed_users", "ppo_joint_users")):
            continue
        key = (source["experiment"], source["phase"], int(source["seed"]))
        realised = contract_index.get(key, [])
        if len(realised) != int(source["jobs"]):
            raise ValueError("Fixed-plan sensitivity requires every realised job contract")
        accepted = sorted((str(contract["job_id"]), int(contract["chosen"])) for contract in realised)
        plan_json = json.dumps(dict(scenario_seed=int(source["seed"]), accepted_job_options=accepted),
                               ensure_ascii=False, separators=(",", ":"))
        plan_sha = hashlib.sha256(plan_json.encode("utf-8")).hexdigest()
        resource = float(source["flexibility_resource_saving_yuan"])
        already_captured = float(source["platform_pre_rebate_saving_yuan"])
        rebate = float(source["user_rebate_yuan"])
        user_loss = float(source["user_loss_yuan"])
        user_gain = float(source["user_gain_yuan"])
        system_saving = resource - user_loss
        eligible = bool(source["carbon_eligible"])
        if not np.isclose(system_saving, source["dispatch_cost_saving_yuan"], rtol=0, atol=1e-6):
            raise ValueError("Fixed-plan sensitivity disagrees with realised resource savings")
        if not np.isclose(user_gain, rebate - user_loss, rtol=0, atol=1e-6):
            raise ValueError("Fixed-plan sensitivity disagrees with realised user utility")
        for share in shares:
            entitlement = share * resource if resource > 0 and eligible else min(resource, 0.)
            settlement = entitlement - already_captured
            cash_change = already_captured - rebate + settlement
            buyer_residual = resource - already_captured - settlement
            identity_error = cash_change + user_gain + buyer_residual - system_saving
            if np.isclose(share, source["resource_share"], rtol=0, atol=1e-12):
                if (not np.isclose(settlement, source["flexibility_settlement_yuan"], rtol=0, atol=1e-6)
                        or not np.isclose(cash_change, source["cash_profit_change_yuan"], rtol=0, atol=1e-6)):
                    raise ValueError("Source-alpha fixed-plan reallocation does not recover the measured settlement")
            results.append(dict(
                experiment=source["experiment"], source_phase=source["phase"], seed=int(source["seed"]),
                selected_joint_cycle=int(source.get("selected_joint_cycle", -1)),
                evaluated_joint_cycle=int(source.get("evaluated_joint_cycle", -1)),
                menu=int(source["menu"]), source_resource_share=float(source["resource_share"]),
                counterfactual_resource_share=float(share), source_plan_sha256=plan_sha,
                source_accepted_job_options=plan_json,
                scope="fixed realised plan; allocation only; no policy prediction or re-optimisation; not parameter selection",
                flexibility_resource_saving_yuan=resource,
                platform_pre_rebate_saving_yuan=already_captured,
                user_rebate_yuan=rebate, user_loss_yuan=user_loss, user_gain_yuan=user_gain,
                carbon_eligible=eligible, actual_grid_emissions_t=float(source["carbon_emissions_t"]),
                grid_emissions_reduction_t=float(source["grid_emissions_reduction_t"]),
                resource_system_cost_yuan=float(source["system_cost_yuan"]),
                full_system_cost_yuan=float(source["full_system_cost_yuan"]),
                dispatch_cost_saving_yuan=system_saving,
                full_cost_saving_yuan=float(source["full_cost_saving_yuan"]),
                shared_resource_entitlement_yuan=float(entitlement),
                flexibility_settlement_yuan=float(settlement),
                buyer_payment_to_platform_yuan=float(max(settlement, 0.)),
                platform_payment_to_buyer_yuan=float(max(-settlement, 0.)),
                cash_profit_before_transfer_yuan=float(source["platform_profit_before_transfer_yuan"]),
                cash_profit_after_transfer_yuan=float(source["platform_profit_before_transfer_yuan"] + settlement),
                cash_profit_change_yuan=float(cash_change),
                buyer_post_settlement_benefit_yuan=float(buyer_residual),
                platform_cash_nonloss=bool(cash_change >= -1e-6),
                purchaser_nonloss=bool(buyer_residual >= -1e-6),
                surplus_identity_error_yuan=float(identity_error),
                surplus_identity_absolute_error_yuan=float(abs(identity_error)),
            ))
    # Confirm no hidden change of plan, resource saving, emissions or user terms
    # occurred while varying allocation. These invariants use source raw days.
    if results:
        frame = pd.DataFrame(results)
        invariants = ("source_plan_sha256", "flexibility_resource_saving_yuan", "user_rebate_yuan",
                      "user_loss_yuan", "actual_grid_emissions_t", "dispatch_cost_saving_yuan",
                      "resource_system_cost_yuan", "full_system_cost_yuan")
        for _, group in frame.groupby(["source_phase", "seed"], sort=False):
            if any(group[key].nunique(dropna=False) != 1 for key in invariants):
                raise ValueError("Sharing sensitivity changed the realised contract plan or physical outcome")
    return results


def summarise_fixed_plan_shares(rows):
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["source_phase", "counterfactual_resource_share"], sort=False).mean(numeric_only=True).reset_index()
    extra = []
    for (phase, share), group in frame.groupby(["source_phase", "counterfactual_resource_share"], sort=False):
        extra.append(dict(
            source_phase=phase, counterfactual_resource_share=share, days=len(group),
            source_plan_count=int(group.source_plan_sha256.nunique()),
            platform_cash_nonloss_days=int(group.platform_cash_nonloss.sum()),
            purchaser_nonloss_days=int(group.purchaser_nonloss.sum()),
            positive_settlement_days=int((group.flexibility_settlement_yuan > 1e-7).sum()),
            negative_settlement_days=int((group.flexibility_settlement_yuan < -1e-7).sum()),
            max_surplus_identity_absolute_error_yuan=float(group.surplus_identity_absolute_error_yuan.max()),
            interpretation="same realised schedules and dispatch; income allocation only; no re-optimised policy"))
    return summary.merge(pd.DataFrame(extra), on=["source_phase", "counterfactual_resource_share"], validate="one_to_one")


def finite_menu_potential(output, config, initial_users, seeds, experiment):
    """Hindsight reference; never used to choose contracts, parameters or models."""
    catalogue = FixedGpuTariffCatalogue(config)
    manifest = json.loads(json.dumps(dict(config=asdict(config), users_sha256=user_hashes(initial_users),
                                         test_scenario_seeds=list(seeds), n_menus=catalogue.n_actions)))
    cache, manifest_path = output / "summary.json", output / "manifest.json"
    if cache.exists() and manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Finite-menu reference cache has different users, config or days")
        return json.loads(cache.read_text())
    users = load_users(initial_users)
    factory = flex_episode_factory(config, maxsize=max(1, len(seeds)), grid_cache_size=1024)
    candidates, selected = [], []
    for seed in seeds:
        episode = factory(int(seed))
        day = []
        for index in range(catalogue.n_actions):
            result = play_menu(episode, catalogue, index, users)
            row = enrich(result, experiment, "finite_menu_candidate", seed, index, catalogue.labels[index])
            candidates.append(row); day.append(row)
        private_feasible = [row for row in day if complete_and_user_ir(row)]
        cash_buyer = [row for row in private_feasible if row["platform_cash_nonloss"] and row["purchaser_nonloss"]]
        cash_buyer_green = [row for row in cash_buyer if row["actual_emissions_nonincrease"]]
        references = (("hindsight_cash_objective", private_feasible, "cash_profit_change_yuan"),
                      ("hindsight_cost_all_participants", cash_buyer, "dispatch_cost_saving_yuan"),
                      ("hindsight_cost_all_participants_and_carbon", cash_buyer_green, "dispatch_cost_saving_yuan"))
        for label, feasible, objective in references:
            if not feasible:
                raise RuntimeError("Original contracts must remain in all reference sets")
            best = max(feasible, key=lambda row: (row[objective], -row["menu"]))
            selected.append(dict(best, phase=label, scheme=f"{experiment}__{label}",
                                 reference=label, feasible_menus=len(feasible), hindsight_complete_information=True))
        print(json.dumps({"finite_menu_test_seed": int(seed), "experiment": experiment}), flush=True)
    reference_mean = summarise(selected)
    report = dict(status="completed", reference_mean=reference_mean.to_dict(orient="records"),
                  scope="daywise hindsight within 77 disclosed menus with identical frozen learned users; not a global or implementable optimum")
    write_csv(output / "test_menu_outcomes.csv", candidates)
    write_csv(output / "selected_references_by_day.csv", selected)
    reference_mean.to_csv(output / "reference_mean.csv", index=False, encoding="utf-8-sig")
    write_json(manifest_path, manifest); write_json(cache, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("constants", "full"), default="full")
    parser.add_argument("--experiments", choices=tuple(EXPERIMENTS), nargs="+", default=list(EXPERIMENTS))
    parser.add_argument("--train-days", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=40000)
    parser.add_argument("--days", type=int, default=32)
    parser.add_argument("--skip-joint", action="store_true")
    parser.add_argument("--fit-joint-constants", action="store_true")
    parser.add_argument("--include-menu-potential", action="store_true")
    args = parser.parse_args(argv)
    if args.train_days < 1 or args.days < 1:
        parser.error("Positive train-days and test days are required")
    torch.set_num_threads(1)
    started = time.monotonic()
    root, output = args.run_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    initial_users = root / "initial_users"
    training_seeds, test_seeds = tuple(range(args.train_days)), tuple(range(args.seed_start, args.seed_start + args.days))
    configs, reports, constants = {}, {}, {}
    signature = None
    for experiment in args.experiments:
        directory = root / experiment
        if (directory / "summary.json").exists():
            config, report = load_config(directory, experiment)
        else:
            if args.stage == "full":
                raise FileNotFoundError(f"Training output is not ready: {directory / 'summary.json'}")
            config, report = declared_config(experiment), None
        if signature is not None and physical_signature(config) != signature:
            raise ValueError("Contract arms changed physical/economic inputs or company preferences")
        signature = physical_signature(config)
        configs[experiment], reports[experiment] = config, report
        constants[experiment] = fit_constant(output / "constant_train" / experiment, config, initial_users, training_seeds)
    if args.stage == "constants":
        print(json.dumps({"selected_constants": constants, "elapsed_seconds": time.monotonic() - started}), flush=True)
        return
    users = load_users(initial_users)
    rows, actions, contracts, manifests, potentials = [], [], [], {}, {}
    for experiment in args.experiments:
        config, report, directory = configs[experiment], reports[experiment], root / experiment
        catalogue, observer = FixedGpuTariffCatalogue(config), FlexTariffObserver(FixedGpuTariffCatalogue(config))
        factory = flex_episode_factory(config, maxsize=args.days, grid_cache_size=1024)
        rows.extend(original_reference(factory, catalogue, test_seeds, experiment))
        phases = [("constant_fixed_users", None, users, constants[experiment]["menu"]),
                  ("ppo_fixed_users", PPO.load(directory / "best_fixed/platform.zip", device="cpu"), users, None)]
        if not args.skip_joint:
            joint_users = load_users(directory / "checkpoints")
            phases.append(("ppo_joint_users", PPO.load(directory / "checkpoints/platform.zip", device="cpu"), joint_users, None))
            if args.fit_joint_constants:
                best_joint = fit_constant(output / "joint_constant_train" / experiment, config, directory / "checkpoints", training_seeds)
                phases.append(("constant_joint_users", None, joint_users, best_joint["menu"]))
        for phase, platform, response_users, constant in phases:
            result, action_rows, contract_rows = evaluate_policy(factory, catalogue, observer, response_users,
                platform, test_seeds, experiment, phase, constant=constant)
            rows.extend(result); actions.extend(action_rows); contracts.extend(contract_rows)
        selected_joint_cycle = int(report.get("selected_joint_cycle", -1))
        for records in (rows, actions, contracts):
            for record in records:
                if record["experiment"] != experiment:
                    continue
                is_joint_bundle = record["phase"] in ("ppo_joint_users", "constant_joint_users")
                record.update(
                    selected_joint_cycle=selected_joint_cycle,
                    evaluated_joint_cycle=selected_joint_cycle if is_joint_bundle else -1,
                    selected_joint_bundle_has_updates=selected_joint_cycle > 0,
                    joint_selection="highest validation cash among cycles 0-4; cycle zero keeps fixed-stage platform and initial users",
                )
        manifests[experiment] = dict(config=asdict(config), initial_user_sha256=user_hashes(initial_users),
            joint_user_sha256=(user_hashes(directory / "checkpoints") if not args.skip_joint else None),
            selected_constant=constants[experiment],
            selected_platform_steps=report.get("selected_platform_steps", report.get("selected_fixed_platform_step")),
            selected_joint_cycle=selected_joint_cycle,
            selected_joint_bundle_has_updates=selected_joint_cycle > 0,
            joint_selection="highest validation cash among cycles 0-4; cycle zero is fixed-stage platform plus initial users")
        if args.include_menu_potential:
            potentials[experiment] = finite_menu_potential(output / "finite_menu_potential" / experiment,
                                                          config, initial_users, test_seeds, experiment)
        print(json.dumps({"evaluated_experiment": experiment, "days": len(test_seeds)}), flush=True)
    write_csv(output / "comparison_by_day.csv", rows)
    means = summarise(rows)
    means.to_csv(output / "comparison_mean.csv", index=False, encoding="utf-8-sig")
    write_csv(output / "actions.csv", actions); write_csv(output / "contracts.csv", contracts)
    paired = paired_comparisons(rows)
    intervals = bootstrap_pairs(paired)
    write_csv(output / "paired_mechanisms.csv", paired)
    write_csv(output / "paired_mechanisms_bootstrap.csv", intervals)
    allocation_rows = fixed_plan_share_sensitivity(rows, contracts)
    allocation_mean = summarise_fixed_plan_shares(allocation_rows)
    write_csv(output / "fixed_plan_share_sensitivity.csv", allocation_rows)
    allocation_mean.to_csv(output / "fixed_plan_share_sensitivity_mean.csv", index=False, encoding="utf-8-sig")
    summary = dict(status="completed", algorithm_training_seed=0, elapsed_seconds=time.monotonic() - started,
        training_scenario_seeds=list(training_seeds), test_scenario_seeds=list(test_seeds), experiments=manifests,
        primary_comparison="same initial learned users and physical days; private cash-profit objective eta=0 in all arms",
        joint_comparison="secondary: validation cash selects a matched four-policy bundle from cycles 0-4; selected cycle zero keeps the fixed-stage platform and unchanged initial users; positive selected cycles contain joint updates",
        joint_cycle_metadata="selected_joint_cycle records the arm's validation checkpoint choice on every row; evaluated_joint_cycle is -1 for phases that do not evaluate a joint checkpoint",
        constant_selection="highest mean settled cash return on training days; unchanged through testing",
        sharing_rule="G=original minus realised generation+compute/migration resource cost; a=pre-rebate bill+operations saving; T=entitlement(G)-a",
        carbon_qualification="only positive shared resource credit requires actual total grid emissions not above original; non-delivery is not hidden by changing tasks",
        accounting=dict(system_cost="all-grid resource generation cost plus variable compute/network operations plus private user loss; excludes transfers",
                        full_cost="system cost plus common fixed installed-capacity cost",
                        user_gain="rebate minus private inconvenience",
                        buyer="modeled cost-recovery system-resource procurement account, not actual ISO cash profit",
                        identity="post-settlement platform profit change + user gain + buyer residual = resource system cost saving"),
        bootstrap_interpretation="10000 paired test-day resamples with RNG seed zero; fixed trained policies, not multi-training-seed robustness",
        comparison_mean=means.to_dict(orient="records"), paired_mechanism_bootstrap=intervals,
        fixed_plan_share_sensitivity=dict(
            interpretation="alpha .5/.8/1.0 reallocates the same realised main-arm contracts and dispatch; no new policy actions, parameter selection, or policy re-optimisation",
            daily_rows=len(allocation_rows), means=allocation_mean.to_dict(orient="records")),
        finite_menu_potential=potentials if args.include_menu_potential else None)
    write_json(output / "summary.json", summary)
    print(json.dumps({"completed": {"output": str(output), "elapsed_seconds": summary["elapsed_seconds"]}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
