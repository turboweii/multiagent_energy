#!/usr/bin/env python3
"""Evaluate compensation-before-sharing against accepted old bookings.

The old platform and its old users make one frozen, accepted outside booking
before each amendment round. Primary comparisons all use the same adapted
initial user PPOs. Constant menus are selected using training days only;
joint-user results are separate. A platform participation floor and the common
carbon cap are pre-binding market rules, not evidence of PPO learning.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from energy_net.coordination.amendment_features import AmendmentTariffCatalogue, AmendmentTariffObserver
from energy_net.coordination.amendment_market import AmendmentConfig
from energy_net.coordination.learning import PolicyBundle
from train_amendments import make_factories


ORDERS = ("direct", "gross", "net")
ORDER_LABELS = {
    "direct": "Direct quotes, no transfer",
    "gross": "Share gross saving, then compensate",
    "net": "Compensate, then share net surplus",
}
PHASE_LABELS = {
    "accepted_old_bookings": "Accepted old bookings",
    "zero_rebate_fixed_users": "Free amendment",
    "constant_fixed_users": "Training-selected constant (78 menus)",
    "extra_rebate_constant_fixed_users": "Training-selected extra-rebate constant",
    "ppo_fixed_users": "PPO, common frozen users",
    "ppo_joint_users": "PPO, validation-selected joint users",
}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def json_value(data):
    return json.loads(json.dumps(data, ensure_ascii=False))


def means(rows):
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key, value in rows[0].items()
            if isinstance(value, (int, float, bool, np.number))}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def user_hashes(directory):
    return {str(user): file_hash(directory / f"user_{user}.zip") for user in range(3)}


def load_users(directory):
    return {user: PPO.load(directory / f"user_{user}.zip", device="cpu") for user in range(3)}


def physical_signature(config):
    fields = asdict(config)
    fields.pop("settlement_order")
    return json_value(fields)


def load_config(directory, order, *, required):
    path = directory / "summary.json"
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Training selection is not complete: {path}")
        return AmendmentConfig(settlement_order=order), None
    report = json.loads(path.read_text(encoding="utf-8"))
    config = AmendmentConfig(**report["config"])
    if config.settlement_order != order:
        raise ValueError(f"Saved {order} policy uses another settlement rule")
    return config, report


def complete_and_ir(metrics):
    return bool(metrics["completion_rate"] >= 1 - 1e-9
                and metrics["deadline_compliance"] >= 1 - 1e-9
                and metrics["user_ir_rate"] >= 1 - 1e-9)


def play_menu(episode, catalogue, menu, users):
    """Replay one disclosed quote with learned, deterministic private users."""
    episode.reset(catalogue.decode(episode, menu))
    while not episode.done:
        job = episode.jobs[episode.position]
        action = users[job.user_id].predict(episode.current_user_observation(), deterministic=True)[0]
        episode.step_user(action)
    return episode.outcome().metrics


def fit_constant(output, legacy_path, config, initial_users, train_seeds):
    """Search all 78 constant menus using mean actual extra platform cash."""
    catalogue = AmendmentTariffCatalogue(config)
    manifest = json_value(dict(
        config=asdict(config), initial_user_sha256=user_hashes(initial_users),
        legacy_selection_sha256=file_hash(legacy_path / "selection.json"),
        legacy_user_sha256=user_hashes(legacy_path / "selected"),
        training_scenario_seeds=list(train_seeds), catalogue_labels=list(catalogue.labels),
        selection="highest training mean platform_gain_vs_outside_yuan; complete work and user IR; lower menu breaks ties",
        outside="one frozen legacy policy and its frozen users, accepted before amendment quotes",
    ))
    cache, manifest_path = output / "selected_constant.json", output / "manifest.json"
    if cache.exists() and manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Constant selection cache belongs to different users, old bookings, config or training days")
        return json.loads(cache.read_text(encoding="utf-8"))
    users = load_users(initial_users)
    factory = make_factories(legacy_path, config, maxsize=max(1, len(train_seeds)))
    rows = []
    for seed in train_seeds:
        episode = factory(int(seed))
        for menu in range(catalogue.n_actions):
            metrics = play_menu(episode, catalogue, menu, users)
            rows.append(dict(menu=menu, menu_label=catalogue.labels[menu], **metrics))
        print(json.dumps(dict(constant_training_seed=int(seed), order=config.settlement_order)), flush=True)
    summaries = []
    for menu in range(catalogue.n_actions):
        group = [row for row in rows if row["menu"] == menu]
        summaries.append(dict(means(group), menu=menu, menu_label=catalogue.labels[menu],
                              all_work_and_user_ir=all(complete_and_ir(row) for row in group)))
    eligible = [row for row in summaries if row["all_work_and_user_ir"]]
    positive_rebate = [row for row in eligible if row["menu"] >= 2]
    if not eligible or not positive_rebate:
        raise RuntimeError("Outside fallback must preserve workload and user participation in every quote")
    key = lambda row: (row["platform_gain_vs_outside_yuan"], -row["menu"])
    best, best_extra = max(eligible, key=key), max(positive_rebate, key=key)
    selected = dict(
        menu=int(best["menu"]), menu_label=best["menu_label"],
        training_mean_extra_profit_yuan=best["platform_gain_vs_outside_yuan"],
        training_mean_extra_cost_saving_yuan=best["incremental_system_cost_saving_vs_outside_yuan"],
        extra_rebate_menu=int(best_extra["menu"]), extra_rebate_menu_label=best_extra["menu_label"],
        extra_rebate_training_mean_extra_profit_yuan=best_extra["platform_gain_vs_outside_yuan"],
        searched_menus=catalogue.n_actions, eligible_menus=len(eligible),
        selection="training mean actual additional platform profit; no test-day selection",
    )
    write_csv(output / "training_menu_outcomes.csv", rows)
    write_csv(output / "training_menu_means.csv", summaries)
    write_json(manifest_path, manifest)
    write_json(cache, selected)
    return selected


def enrich(metrics, order, phase, seed, menu, label):
    row = dict(metrics, order=order, phase=phase, scheme=f"{order}__{phase}",
               seed=int(seed), menu=int(menu), menu_label=label)
    old_cost = float(row["outside_system_cost_yuan"])
    saved = float(row["incremental_system_cost_saving_vs_outside_yuan"])
    profit = float(row["platform_gain_vs_outside_yuan"])
    cap = float(row["carbon_budget_t"])
    original_cost = float(row["original_system_cost_yuan"])
    old_incremental = float(row["outside_incremental_system_cost_yuan"])
    row.update(
        old_system_cost_yuan=old_cost,
        old_platform_profit_yuan=float(row["outside_platform_profit_yuan"]),
        old_grid_emissions_t=float(row["outside_carbon_emissions_t"]),
        new_system_cost_yuan=float(row["system_cost_yuan"]),
        new_platform_profit_yuan=float(row["platform_profit_yuan"]),
        new_grid_emissions_t=float(row["carbon_emissions_t"]),
        extra_cost_saving_yuan=saved,
        extra_resource_saving_yuan=float(row["flexibility_resource_saving_yuan"]),
        extra_cost_saving_rate=saved / old_cost if old_cost else 0.,
        extra_incremental_cost_saving_rate=saved / old_incremental if old_incremental else 0.,
        extra_platform_profit_yuan=profit,
        extra_user_gain_yuan=float(row["user_gain_yuan"]),
        extra_buyer_gain_yuan=float(row["buyer_post_settlement_benefit_yuan"]),
        extra_user_inconvenience_yuan=float(row["user_loss_yuan"] - row["outside_user_loss_yuan"]),
        extra_grid_emissions_reduction_t=float(row["outside_carbon_emissions_t"] - row["carbon_emissions_t"]),
        cost_saving_vs_original_yuan=original_cost - float(row["system_cost_yuan"]),
        cost_saving_vs_original_rate=(original_cost - float(row["system_cost_yuan"])) / original_cost if original_cost else 0.,
        platform_profit_change_vs_original_yuan=float(row["platform_profit_yuan"] - row["original_platform_profit_yuan"]),
        same_cap_compliant=bool(row["carbon_emissions_t"] <= cap + 1e-6),
        old_same_cap_compliant=bool(row["outside_carbon_emissions_t"] <= cap + 1e-6),
        cap_excess_t=float(max(row["carbon_emissions_t"] - cap, 0.)),
        platform_old_floor_satisfied=bool(profit >= -1e-6),
        user_ir_satisfied=bool(row["user_ir_rate"] >= 1 - 1e-9),
        buyer_nonloss=bool(row["buyer_post_settlement_benefit_yuan"] >= -1e-6),
        all_work_complete=bool(row["completion_rate"] >= 1 - 1e-9),
        all_deadlines_met=bool(row["deadline_compliance"] >= 1 - 1e-9),
        menu_open=bool(menu != 0),
        nonzero_extra_rebate_quote=bool(menu >= 2),
        has_accepted_amendment=bool(row["accepted_amendments"] > 0),
        accepted_job_fraction=float(row["accepted_amendments"] / row["jobs"]),
        has_additional_effect=bool(row["accepted_amendments"] > 0 and
                                   (abs(saved) > 1e-6 or abs(profit) > 1e-6 or abs(row["user_gain_yuan"]) > 1e-6)),
        surplus_identity_absolute_error_yuan=float(abs(row["surplus_identity_error_yuan"])),
    )
    if not np.isclose(profit, row["platform_profit_yuan"] - row["outside_platform_profit_yuan"], atol=1e-6, rtol=0):
        raise ValueError("Extra platform profit does not match accepted-old-booking cash difference")
    if not np.isclose(saved, row["contract_system_cost_saving_yuan"], atol=1e-6, rtol=0):
        raise ValueError("Additional social cost saving must equal resource saving minus additional inconvenience")
    if not np.isclose(saved, profit + row["user_gain_yuan"] + row["buyer_post_settlement_benefit_yuan"], atol=1e-6, rtol=0):
        raise ValueError("Platform, user and buyer increments do not sum to additional social cost saving")
    return row


def evaluate_policy(factory, catalogue, observer, users, platform, seeds, order, phase, *, constant=None):
    bundle = PolicyBundle(platform=platform, users=users, platform_codec=catalogue,
                          platform_observer=observer, training=False)
    rows, actions, contracts = [], [], []
    for seed in seeds:
        episode = factory(int(seed))
        menu = (int(constant) if constant is not None else int(np.asarray(
            platform.predict(bundle.platform_observation(episode), deterministic=True)[0]).item()))
        episode.set_platform_action(catalogue.decode(episode, menu))
        while not episode.done:
            # Exiting the second round executes accepted old contracts exactly.
            action = 0 if menu == 0 else bundle.user_action(episode)
            _, _, info = episode.step_user(action)
            contracts.append(dict(order=order, phase=phase, seed=int(seed), menu=menu, **info))
        rows.append(enrich(episode.outcome().metrics, order, phase, seed, menu, catalogue.labels[menu]))
        actions.append(dict(order=order, phase=phase, seed=int(seed), menu=menu,
                            menu_label=catalogue.labels[menu], raw_menu=episode.platform_action.tolist(),
                            amendment_open=episode.amendment_open,
                            extra_discounts=episode.discounts.tolist(),
                            outside_policy_label=episode.outside_policy_label,
                            outside_source_options=list(episode.outside_selected_options)))
    return rows, actions, contracts


def summarise(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    result = frame.groupby(["scheme", "order", "phase"], sort=False).mean(numeric_only=True).reset_index()
    counts = []
    for scheme, group in frame.groupby("scheme", sort=False):
        counts.append(dict(
            scheme=scheme, days=len(group),
            pooled_extra_cost_saving_rate=float(group.extra_cost_saving_yuan.sum() / group.old_system_cost_yuan.sum()),
            pooled_extra_incremental_cost_saving_rate=float(group.extra_cost_saving_yuan.sum() / group.outside_incremental_system_cost_yuan.sum()),
            old_floor_satisfied_days=int(group.platform_old_floor_satisfied.sum()),
            max_old_floor_shortfall_yuan=float(group.platform_floor_shortfall_yuan.max()),
            same_cap_compliant_days=int(group.same_cap_compliant.sum()),
            max_cap_excess_t=float(group.cap_excess_t.max()),
            complete_work_days=int(group.all_work_complete.sum()),
            deadlines_met_days=int(group.all_deadlines_met.sum()),
            user_ir_days=int(group.user_ir_satisfied.sum()),
            buyer_nonloss_days=int(group.buyer_nonloss.sum()),
            exit_days=int((group.menu == 0).sum()),
            free_amendment_quote_days=int((group.menu == 1).sum()),
            nonzero_rebate_quote_days=int(group.nonzero_extra_rebate_quote.sum()),
            accepted_amendment_days=int(group.has_accepted_amendment.sum()),
            positive_extra_cost_saving_days=int((group.extra_cost_saving_yuan > 1e-6).sum()),
            positive_extra_platform_profit_days=int((group.extra_platform_profit_yuan > 1e-6).sum()),
            additional_effect_days=int(group.has_additional_effect.sum()),
            distinct_menus=int(group.menu.nunique()),
            max_surplus_identity_error_yuan=float(group.surplus_identity_absolute_error_yuan.max()),
        ))
    return result.merge(pd.DataFrame(counts), on="scheme", validate="one_to_one")


def paired_comparisons(rows):
    frame = pd.DataFrame(rows)
    comparisons = []
    for order in ORDERS:
        for before in ("zero_rebate_fixed_users", "constant_fixed_users", "extra_rebate_constant_fixed_users"):
            comparisons.append((f"{order}_ppo_vs_{before}", order, before, order, "ppo_fixed_users"))
    comparisons += [("net_vs_gross_fixed_users", "gross", "ppo_fixed_users", "net", "ppo_fixed_users"),
                    ("net_vs_direct_fixed_users", "direct", "ppo_fixed_users", "net", "ppo_fixed_users")]
    metrics = ("extra_cost_saving_yuan", "extra_resource_saving_yuan", "extra_platform_profit_yuan",
               "extra_user_gain_yuan", "extra_buyer_gain_yuan", "additional_user_rebate_yuan",
               "extra_grid_emissions_reduction_t", "accepted_amendments", "amendment_admission_rejections")
    pairs = []
    for label, before_order, before_phase, after_order, after_phase in comparisons:
        left = frame[(frame.order == before_order) & (frame.phase == before_phase)].set_index("seed")
        right = frame[(frame.order == after_order) & (frame.phase == after_phase)].set_index("seed")
        if left.empty or right.empty:
            continue
        if set(left.index) != set(right.index):
            raise ValueError("Paired comparisons need the same physical days and outside bookings")
        for seed in left.index:
            pairs.append(dict(comparison=label, before_order=before_order, before_phase=before_phase,
                after_order=after_order, after_phase=after_phase, seed=int(seed),
                **{name: float(right.loc[seed, name] - left.loc[seed, name]) for name in metrics}))
    return pairs


def finite_menu_potential(output, factory, catalogue, users, seeds, order):
    """Daywise hindsight within 78 menus, never a selectable test policy."""
    candidates, selected = [], []
    for seed in seeds:
        episode, day = factory(int(seed)), []
        for menu in range(catalogue.n_actions):
            metrics = play_menu(episode, catalogue, menu, users)
            row = enrich(metrics, order, "hindsight_candidate", seed, menu, catalogue.labels[menu])
            candidates.append(row)
            if complete_and_ir(row):
                day.append(row)
        for label, objective in (("hindsight_extra_profit", "extra_platform_profit_yuan"),
                                 ("hindsight_extra_cost_saving", "extra_cost_saving_yuan")):
            best = max(day, key=lambda row: (row[objective], -row["menu"]))
            selected.append(dict(best, phase=label, scheme=f"{order}__{label}",
                                 hindsight_only=True, candidate_count=len(day)))
        print(json.dumps(dict(hindsight_reference_seed=int(seed), order=order)), flush=True)
    write_csv(output / "candidate_menus.csv", candidates)
    write_csv(output / "hindsight_by_day.csv", selected)
    report = dict(scope="post-hoc per-day reference within 78 menus using common frozen user PPOs; no test strategy or model selection",
                  results=summarise(selected).to_dict(orient="records"))
    write_json(output / "summary.json", report)
    return report


def plot_primary(summary, output):
    """Plot measured increments; the final panel identifies rule outcomes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = summary[summary.phase.isin(("zero_rebate_fixed_users", "constant_fixed_users", "ppo_fixed_users"))].copy()
    if selected.empty:
        return
    short_phase = {"zero_rebate_fixed_users": "Free", "constant_fixed_users": "Constant", "ppo_fixed_users": "PPO"}
    selected["sort_key"] = selected.order.map({o: i for i, o in enumerate(ORDERS)}) * 3 + selected.phase.map({p:i for i,p in enumerate(short_phase)})
    selected = selected.sort_values("sort_key")
    labels = [f"{order}\n{short_phase[phase]}" for order, phase in zip(selected.order, selected.phase)]
    x = np.arange(len(selected))
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for axis, column, title, unit in ((axes[0, 0], "extra_cost_saving_yuan", "Additional social cost saved", "yuan/day"),
            (axes[0, 1], "extra_platform_profit_yuan", "Additional platform cash profit", "yuan/day"),
            (axes[1, 1], "accepted_job_fraction", "Accepted amendments", "fraction of jobs"),
            (axes[2, 0], "extra_grid_emissions_reduction_t", "Additional grid emission reduction", "tonnes/day")):
        axis.bar(x, selected[column], color=[{"direct":"#777777", "gross":"#d18d36", "net":"#257f78"}[o] for o in selected.order])
        axis.axhline(0, color="#555555", linewidth=.7)
        axis.set(title=title, ylabel=unit, xticks=x, xticklabels=labels)
    axis, positive, negative = axes[1, 0], np.zeros(len(selected)), np.zeros(len(selected))
    for column, label, color in (("extra_platform_profit_yuan", "Platform", "#257f78"),
                                ("extra_user_gain_yuan", "Users", "#78ae59"),
                                ("extra_buyer_gain_yuan", "Buyer", "#d18d36")):
        values = selected[column].to_numpy(float)
        axis.bar(x, values, bottom=np.where(values >= 0, positive, negative), label=label, color=color)
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    axis.axhline(0, color="#555555", linewidth=.7)
    axis.set(title="Allocation of additional social saving", ylabel="yuan/day", xticks=x, xticklabels=labels)
    axis.legend(frameon=False, ncol=3)
    axes[2, 1].axis("off")
    cells = [[f"{order}/{short_phase[phase]}", f"{int(floor)}/{int(days)}", f"{int(cap)}/{int(days)}", f"{int(work)}/{int(days)}"]
             for order, phase, floor, cap, work, days in zip(selected.order, selected.phase,
                selected.old_floor_satisfied_days, selected.same_cap_compliant_days,
                selected.complete_work_days, selected.days)]
    axes[2, 1].table(cellText=cells, colLabels=["Scheme", "Old profit floor", "Common cap", "All work"], loc="center", cellLoc="center")
    axes[2, 1].set_title("Pre-binding rule outcomes, not PPO learning scores")
    fig.suptitle("Same accepted old bookings and frozen users; mean increments over old contracts", fontsize=13)
    fig.savefig(output / "primary_comparison.png", dpi=180)
    fig.savefig(output / "primary_comparison.svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("constants", "full"), default="full")
    parser.add_argument("--orders", choices=ORDERS, nargs="+", default=list(ORDERS))
    parser.add_argument("--train-days", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument("--days", type=int, default=32)
    parser.add_argument("--skip-joint", action="store_true")
    parser.add_argument("--include-menu-potential", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if args.train_days < 1 or args.days < 1:
        parser.error("Positive training and evaluation day counts are required")
    torch.set_num_threads(1)
    started = time.monotonic()
    root, legacy, output = args.run_root.resolve(), args.legacy.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    initial_users = root / "initial_users"
    train_seeds = tuple(range(args.train_days))
    test_seeds = tuple(range(args.seed_start, args.seed_start + args.days))
    configs, reports = {}, {}
    signature = None
    # Complete every model-selection prerequisite before touching test days.
    for order in args.orders:
        config, report = load_config(root / order, order, required=args.stage == "full")
        if signature is not None and physical_signature(config) != signature:
            raise ValueError("Settlement arms changed physical inputs or economic scales")
        signature, configs[order], reports[order] = physical_signature(config), config, report
        if args.stage == "full":
            if report.get("legacy_path") and Path(report["legacy_path"]).resolve() != legacy:
                raise ValueError("Evaluation legacy reference differs from the frozen training reference")
            required = [root / order / "best_fixed" / "platform.zip"]
            if not args.skip_joint:
                required += [root / order / "checkpoints" / name for name in
                             ("platform.zip", "user_0.zip", "user_1.zip", "user_2.zip")]
            for path in required:
                if not path.exists():
                    raise FileNotFoundError(path)
    constants = {order: fit_constant(output / "constant_train" / order, legacy, configs[order],
                                   initial_users, train_seeds) for order in args.orders}
    if args.stage == "constants":
        write_json(output / "constant_selection.json", constants)
        print(json.dumps(dict(selected_constants=constants, elapsed_seconds=time.monotonic() - started)), flush=True)
        return
    users = load_users(initial_users)
    rows, actions, contracts, manifests, potentials = [], [], [], {}, {}
    for order in args.orders:
        config, report, directory = configs[order], reports[order], root / order
        catalogue = AmendmentTariffCatalogue(config)
        observer = AmendmentTariffObserver(catalogue)
        factory = make_factories(legacy, config, maxsize=max(args.days, 1))
        phases = [("accepted_old_bookings", None, users, 0),
                  ("zero_rebate_fixed_users", None, users, 1),
                  ("constant_fixed_users", None, users, constants[order]["menu"]),
                  ("extra_rebate_constant_fixed_users", None, users, constants[order]["extra_rebate_menu"]),
                  ("ppo_fixed_users", PPO.load(directory / "best_fixed" / "platform.zip", device="cpu"), users, None)]
        if not args.skip_joint:
            phases.append(("ppo_joint_users", PPO.load(directory / "checkpoints" / "platform.zip", device="cpu"),
                           load_users(directory / "checkpoints"), None))
        for phase, platform, response_users, constant in phases:
            day_rows, day_actions, day_contracts = evaluate_policy(factory, catalogue, observer,
                response_users, platform, test_seeds, order, phase, constant=constant)
            for records in (day_rows, day_actions, day_contracts):
                for record in records:
                    record.update(selected_joint_cycle=int(report.get("selected_joint_cycle", 0)),
                        evaluated_joint_cycle=int(report.get("selected_joint_cycle", 0)) if phase == "ppo_joint_users" else -1,
                        common_initial_users=phase != "ppo_joint_users")
            rows.extend(day_rows); actions.extend(day_actions); contracts.extend(day_contracts)
            print(json.dumps(dict(evaluated_order=order, phase=phase, days=len(day_rows))), flush=True)
        manifests[order] = dict(config=asdict(config), initial_user_sha256=user_hashes(initial_users),
            platform_sha256=file_hash(directory / "best_fixed" / "platform.zip"),
            selected_platform_steps=int(report["selected_platform_steps"]),
            selected_joint_cycle=int(report.get("selected_joint_cycle", 0)),
            fixed_constant=constants[order], joint_users_sha256=None if args.skip_joint else user_hashes(directory / "checkpoints"))
        if args.include_menu_potential:
            potentials[order] = finite_menu_potential(output / "menu_potential" / order,
                factory, catalogue, users, test_seeds, order)
    summary = summarise(rows)
    primary = summary[summary.phase != "ppo_joint_users"].copy()
    joint = summary[summary.phase == "ppo_joint_users"].copy()
    pairs = paired_comparisons(rows)
    pair_summary = (pd.DataFrame(pairs).groupby("comparison", sort=False).mean(numeric_only=True).reset_index()
                    if pairs else pd.DataFrame())
    write_csv(output / "outcomes_by_day.csv", rows)
    write_json(output / "actions_by_day.json", actions)
    write_json(output / "contracts_by_job.json", contracts)
    summary.to_csv(output / "summary_mean.csv", index=False, encoding="utf-8-sig")
    primary.to_csv(output / "primary_mean.csv", index=False, encoding="utf-8-sig")
    joint.to_csv(output / "joint_mean.csv", index=False, encoding="utf-8-sig")
    write_csv(output / "paired_differences_by_day.csv", pairs)
    pair_summary.to_csv(output / "paired_mean.csv", index=False, encoding="utf-8-sig")
    report = dict(
        study="accepted old contracts; compensation before net-surplus sharing; one training seed",
        run_root=str(root), legacy_path=str(legacy), elapsed_seconds=time.monotonic() - started,
        training_scenario_seeds=list(train_seeds), test_scenario_seeds=list(test_seeds),
        outside_reference="fixed legacy platform and fixed legacy users book before each amendment; never test-day best policy",
        primary_users="same frozen initial_users in every rule, quote and platform PPO comparison",
        cost="same-cap generation resource cost plus execution operations and private inconvenience; cash transfers excluded",
        profit="actual post-settlement platform cash after fixed compute cost, relative to accepted old contracts",
        rule_outcomes="old-profit floor, common cap and contract fallback arise from pre-binding admission; they are not learning achievements",
        learning_effect="PPO increments relative to free amendments and both training-selected constant benchmarks with the same users",
        all_exit_interpretation="if PPO always selects outside action 0, its additional effect is zero",
        joint_interpretation="joint bundle selected by validation among cycles 0-4; cycle 0 retains fixed-stage platform and initial users",
        primary=primary.to_dict(orient="records"), joint=joint.to_dict(orient="records"),
        paired_mean=pair_summary.to_dict(orient="records"), manifests=manifests, menu_potential=potentials,
    )
    write_json(output / "report.json", report)
    if not args.no_plot:
        plot_primary(summary, output)
    print(json.dumps(dict(completed=True, output=str(output), elapsed_seconds=time.monotonic() - started)), flush=True)


if __name__ == "__main__":
    main()
