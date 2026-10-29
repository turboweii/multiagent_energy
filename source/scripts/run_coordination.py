#!/usr/bin/env python3
"""Compare complete, identical synthetic task days under each pricing scheme."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from energy_net.coordination.baselines import (
    evaluate_baselines, finite_price_candidates, finite_price_search, physical_oracle,
)
from energy_net.coordination.market import MarketEpisode
from energy_net.coordination.scenario import MarketConfig, make_scenario


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_policies(directory):
    import torch
    from stable_baselines3 import PPO
    from energy_net.coordination.learning import PolicyBundle
    torch.set_num_threads(1)
    platform = PPO.load(directory / "platform.zip", device="cpu")
    codec = None
    if hasattr(platform.action_space, "n"):
        from energy_net.coordination.tariffs import TariffCatalogue
        codec = TariffCatalogue()
        if platform.action_space.n != codec.n_actions:
            raise ValueError("Saved platform action count does not match the tariff catalogue.")
    observer = None
    if len(platform.observation_space.shape) == 2:
        from energy_net.coordination.platform_learning import CandidateTariffObserver
        observer = CandidateTariffObserver(codec)
    return PolicyBundle(
        platform=platform, platform_codec=codec, platform_observer=observer,
        users={u: PPO.load(directory / f"user_{u}.zip", device="cpu") for u in range(3)},
    )


def diagnose(scenario, outcome):
    spec, clearing = scenario.grid_spec, outcome.clearing
    renewable = spec.generator_emissions_t_per_mwh == 0
    return {
        "line_binding_fraction": float(np.mean(np.abs(clearing.flows_mw)
                                               >= spec.line_capacity_mw - 1e-6)),
        "renewable_curtailment_mwh": float(np.maximum(
            0, spec.generator_capacity_mw[:, renewable]
            - clearing.generation_mw[:, renewable]).sum() * spec.dt_hours),
        "carbon_slack_t": scenario.carbon_budget_t - clearing.emissions_t,
    }


def export_scenario(scenario, output):
    arrays = {
        "background_mw": scenario.background_mw,
        "renewable_mw": scenario.renewable_mw,
        "baseline_load_mw": scenario.baseline_load_mw,
        "carbon_budget_t": scenario.carbon_budget_t,
    }
    arrays.update({f"grid_{k}": v for k, v in asdict(scenario.grid_spec).items()})
    jobs = []
    for job in scenario.jobs:
        row = {k: v for k, v in asdict(job).items() if k != "options"}
        row["options"] = []
        for option in job.options:
            item = {}
            for key, value in asdict(option).items():
                if isinstance(value, np.ndarray):
                    reference = f"{job.job_id}_option_{option.index}_{key}"
                    arrays[reference] = value
                    item[key + "_npz_key"] = reference
                else:
                    item[key] = value
            row["options"].append(item)
        jobs.append(row)
    np.savez_compressed(output / "scenario_seed0.npz", **arrays)
    (output / "scenario_seed0.json").write_text(json.dumps({
        "seed": scenario.seed, "config": asdict(scenario.config),
        "calibration": scenario.calibration, "jobs": jobs,
        "array_file": "scenario_seed0.npz",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_example(scenario, outcomes, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base, chosen = outcomes["uniform"], outcomes["reservation_plan"]
    hour = np.arange(48) * scenario.config.dt_hours
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for region in range(3):
        axes[0, 0].step(hour, base.clearing.lmp_yuan_per_mwh[:, region],
                        where="post", label=f"Region {region}")
    axes[0, 0].set(ylabel="Baseline LMP (yuan/MWh)", xlabel="Hour")
    axes[0, 0].legend(fontsize=8)
    ep = MarketEpisode(scenario)
    for label, outcome in (("Original", base), ("Posted-price candidate", chosen)):
        alloc, _ = ep.allocations_for(outcome.selected_options)
        axes[0, 1].step(hour, alloc.sum(axis=1) / scenario.config.dt_hours,
                        where="post", label=label)
    axes[0, 1].set(ylabel="Compute work rate (CU/hour)", xlabel="Hour")
    axes[0, 1].legend(fontsize=8)
    limit = scenario.grid_spec.line_capacity_mw
    for line in range(len(limit)):
        axes[1, 0].step(hour, base.clearing.flows_mw[:, line] / limit[line],
                        where="post", label=f"Line {line}")
    axes[1, 0].axhline(1, color="k", ls=":", lw=1)
    axes[1, 0].axhline(-1, color="k", ls=":", lw=1)
    axes[1, 0].set(ylabel="Baseline flow / line capacity", xlabel="Hour")
    axes[1, 0].legend(fontsize=8)
    names = list(outcomes)
    savings = [base.metrics["system_cost_yuan"] - outcomes[k].metrics["system_cost_yuan"]
               for k in names]
    axes[1, 1].bar(np.arange(len(names)), savings)
    axes[1, 1].set_xticks(np.arange(len(names)), names, rotation=15, fontsize=8)
    axes[1, 1].set(ylabel="Social-cost saving vs original (yuan/day)")
    fig.suptitle("Synthetic day, seed 0: fixed work and deadlines")
    fig.savefig(output / "example_seed0.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--days", type=int, default=8)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--reference-checkpoints", type=Path)
    parser.add_argument("--ablation-checkpoints", type=Path)
    parser.add_argument("--catalogue-benchmark", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("days must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = MarketConfig()
    bundle = load_policies(args.checkpoints) if args.checkpoints else None
    reference = load_policies(args.reference_checkpoints) if args.reference_checkpoints else None
    ablation = load_policies(args.ablation_checkpoints) if args.ablation_checkpoints else None
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.days):
        scenario = make_scenario(seed, config)
        outcomes = evaluate_baselines(scenario)
        if args.catalogue_benchmark:
            from energy_net.coordination.tariffs import TariffCatalogue
            codec, probe = TariffCatalogue(config), MarketEpisode(scenario)
            search = finite_price_search(scenario, {
                codec.labels[i]: codec.decode(probe, i) for i in range(codec.n_actions)})
            outcomes["catalogue_menu_benchmark"] = search.best_outcome
        # Fixed public rules disclose their amplitude in advance. Unlike the
        # finite-family benchmarks, they do not select a winner using true beta.
        candidates = finite_price_candidates(scenario)
        for label, candidate in (
            ("public_time_rule", "time_only_0.20"),
            ("public_time_region_rule", "time_region_0.20"),
            ("public_quota_rule", "reservation_0.20_0.75"),
        ):
            ep = MarketEpisode(scenario)
            ep.set_platform_action(candidates[candidate])
            outcomes[label] = ep.play()
        # Strong small-discount rule: public template, no learned actor or
        # realised-profit screening. Useful when PPO selects this same menu.
        from energy_net.coordination.tariffs import TariffCatalogue
        codec = TariffCatalogue(config)
        index = codec.labels.index("time_region_discount_0.05_quota_1.00")
        ep = MarketEpisode(scenario)
        ep.set_platform_action(codec.decode(ep, index))
        outcomes["public_low_discount_rule"] = ep.play()
        if bundle:
            ep = MarketEpisode(scenario)
            ep.set_platform_action(bundle.platform_action(ep))
            outcomes["independent_ppo"] = ep.play(bundle.users)
            ep = MarketEpisode(scenario)
            ep.set_platform_action(bundle.platform_action(ep))
            outcomes["ppo_platform_exact_users"] = ep.play()
            ep = MarketEpisode(scenario)
            ep.set_platform_action(candidates["time_region_0.20"])
            outcomes["public_rule_ppo_users"] = ep.play(bundle.users)
        if reference:
            ep = MarketEpisode(scenario)
            ep.set_platform_action(reference.platform_action(ep))
            outcomes["previous_independent_ppo"] = ep.play(reference.users)
        if ablation:
            ep = MarketEpisode(scenario)
            ep.set_platform_action(ablation.platform_action(ep))
            outcomes["staged_long_horizon_ablation"] = ep.play(ablation.users)
        baseline = outcomes["uniform"].metrics["system_cost_yuan"]
        for name, outcome in outcomes.items():
            row = dict(outcome.metrics, scheme=name, **diagnose(scenario, outcome))
            row["comparison_role"] = (
                "finite_full_information_benchmark" if name in (
                    "time_only", "time_region", "reservation_plan", "catalogue_menu_benchmark") else
                "learning_diagnostic" if name in (
                    "ppo_platform_exact_users", "public_rule_ppo_users") else
                "public_rule" if name.startswith("public_") else
                "four_learning_actors" if name == "independent_ppo" else "original_contracts")
            row["social_saving_yuan"] = baseline - row["system_cost_yuan"]
            row["cooperation_success"] = bool(
                row["social_saving_yuan"] > 1e-6
                and row["user_ir_rate"] >= 1 - 1e-9
                and row["platform_participates"]
                and row["completion_rate"] >= 1 - 1e-9
                and row["deadline_compliance"] >= 1 - 1e-9)
            for owner in range(3):
                choices = [job.options[index] for job, index
                           in zip(scenario.jobs, outcome.selected_options)
                           if job.user_id == owner]
                row[f"user_{owner}_migrated_jobs"] = sum(o.migrated for o in choices)
                row[f"user_{owner}_delayed_jobs"] = sum(o.delay_hours > 0 for o in choices)
            rows.append(row)
        oracle = physical_oracle(scenario)
        rows.append(dict(oracle.to_dict(), scheme="physical_milp",
                         comparison_role="physical_full_information_benchmark",
                         social_saving_yuan=baseline - oracle.objective_yuan))
        print(json.dumps({"seed": seed, "schemes": len(outcomes) + 1,
                          "work_cu": scenario.total_work_cu,
                          "physical_optimal": oracle.optimal}), flush=True)
    write_csv(output / "comparison_by_day.csv", rows)
    summary = []
    for name in dict.fromkeys(row["scheme"] for row in rows):
        selected = [row for row in rows if row["scheme"] == name]
        numeric = {key: float(np.mean([row[key] for row in selected]))
                   for key in selected[0]
                   if all(key in row and isinstance(row[key], (int, float, bool, np.number))
                          for row in selected)}
        summary.append(dict(scheme=name, days=len(selected), **numeric))
    write_csv(output / "comparison_mean.csv", summary)
    example = make_scenario(0, config)
    export_scenario(example, output)
    example_outcomes = evaluate_baselines(example)
    write_csv(output / "example_seed0.csv", [dict(o.metrics, scheme=k)
                                            for k, o in example_outcomes.items()])
    plot_example(example, example_outcomes, output)
    (output / "run_config.json").write_text(json.dumps({
        "config": asdict(config), "evaluation_seeds": list(range(
            args.seed_start, args.seed_start + args.days)),
        "checkpoint_directory": str(args.checkpoints) if args.checkpoints else None,
        "reference_checkpoint_directory": str(args.reference_checkpoints) if args.reference_checkpoints else None,
        "ablation_checkpoint_directory": str(args.ablation_checkpoints) if args.ablation_checkpoints else None,
        "catalogue_benchmark": args.catalogue_benchmark,
        "tariff_baselines": "best eligible member of each disclosed finite family; exact current-task responses",
        "physical_milp": "complete-information fixed-work scheduling; excludes payment and participation constraints",
        "ppo": "four independently trained actors; no post-training profit rollback",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
