#!/usr/bin/env python3
"""Report physical/economic sensitivities of original GPU batch contracts.

This script neither loads nor trains a policy and never selects a model.
All jobs retain their original time/location.  Carbon-price sensitivities
re-clear the same physical load.  Common installed-capacity cost sensitivities
change cash/full-cost accounting but leave dispatch and contract choices fixed.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from energy_net.coordination.gpu_market import GpuMarketEpisode
from energy_net.coordination.gpu_scenario import GpuMarketConfig, make_gpu_scenario


def baseline_row(seed, config, experiment):
    scenario = make_gpu_scenario(seed, config)
    episode = GpuMarketEpisode(scenario)
    metrics = episode.baseline_metrics
    installed_gpu_hours = sum(config.capacity_cu_per_hour) * config.n_steps * config.dt_hours
    contribution = metrics["platform_profit_before_fixed_cost_yuan"]
    clearing = episode.baseline_clearing
    row = {
        "experiment": experiment,
        "scenario_seed": int(seed),
        "site_scale": config.site_scale,
        "policy_carbon_price_yuan_per_t": config.carbon_price_yuan_per_t,
        "internal_carbon_value_yuan_per_t": config.internal_carbon_value_yuan_per_t,
        "fixed_capacity_fee_yuan_per_gpu_capacity_hour": config.fixed_compute_cost_yuan_per_gpu_capacity_hour,
        "jobs": len(scenario.jobs),
        "work_gpu_hours": scenario.total_work_cu,
        "installed_gpus_per_site": config.installed_nodes_per_site * config.gpus_per_node,
        "installed_gpu_capacity_hours_per_day": installed_gpu_hours,
        "gpu_capacity_utilization": scenario.calibration["gpu_capacity_utilization"],
        "dc_energy_mwh": metrics["dc_energy_mwh"],
        "background_energy_mwh": scenario.calibration["background_energy_mwh"],
        "dc_energy_share": metrics["dc_energy_share"],
        "generation_resource_cost_yuan": metrics["generation_cost_yuan"],
        "background_only_generation_resource_cost_yuan": metrics["background_only_generation_cost_yuan"],
        "dispatch_system_cost_yuan": metrics["dispatch_system_cost_yuan"],
        "incremental_system_cost_yuan": metrics["incremental_system_cost_yuan"],
        "fixed_compute_cost_yuan": metrics["fixed_compute_cost_yuan"],
        "full_system_cost_yuan": metrics["full_system_cost_yuan"],
        "platform_revenue_yuan": metrics["service_revenue_yuan"],
        "platform_electricity_bill_yuan": metrics["electricity_bill_yuan"],
        "platform_variable_compute_migration_cost_yuan": metrics["compute_migration_cost_yuan"],
        "platform_contribution_before_fixed_cost_yuan": contribution,
        "platform_profit_after_fixed_cost_yuan": metrics["platform_profit_yuan"],
        "break_even_fixed_fee_yuan_per_gpu_capacity_hour": contribution / installed_gpu_hours,
        "grid_carbon_emissions_t": metrics["carbon_emissions_t"],
        "dc_consumption_footprint_t": metrics["dc_consumption_emissions_t"],
        "generator_carbon_payment_yuan": metrics["generator_carbon_payment_yuan"],
        "grid_dispatch_objective_yuan": metrics["grid_dispatch_objective_yuan"],
        "mean_lmp_yuan_per_mwh": float(clearing.lmp_yuan_per_mwh.mean()),
        "min_lmp_yuan_per_mwh": float(clearing.lmp_yuan_per_mwh.min()),
        "max_lmp_yuan_per_mwh": float(clearing.lmp_yuan_per_mwh.max()),
        "original_contracts_only": True,
        "baseline_load_sha256": hashlib.sha256(np.ascontiguousarray(
            scenario.baseline_load_mw, dtype=np.float64).tobytes()).hexdigest(),
        "same_day_common_fixed_cost_cancels_in_paired_changes": True,
    }
    assert np.isclose(row["full_system_cost_yuan"],
                      row["dispatch_system_cost_yuan"] + row["fixed_compute_cost_yuan"])
    assert np.isclose(row["platform_profit_after_fixed_cost_yuan"],
                      contribution - row["fixed_compute_cost_yuan"])
    assert np.isclose(row["grid_dispatch_objective_yuan"],
                      row["generation_resource_cost_yuan"] + row["generator_carbon_payment_yuan"])
    return row, scenario


def summarize(rows):
    groups = {}
    fields = ("dc_energy_share", "dc_energy_mwh", "gpu_capacity_utilization",
              "generation_resource_cost_yuan", "dispatch_system_cost_yuan",
              "full_system_cost_yuan", "platform_electricity_bill_yuan",
              "platform_contribution_before_fixed_cost_yuan",
              "platform_profit_after_fixed_cost_yuan",
              "break_even_fixed_fee_yuan_per_gpu_capacity_hour",
              "grid_carbon_emissions_t", "mean_lmp_yuan_per_mwh",
              "generator_carbon_payment_yuan")
    for row in rows:
        key = (row["experiment"], row["site_scale"],
               row["fixed_capacity_fee_yuan_per_gpu_capacity_hour"],
               row["policy_carbon_price_yuan_per_t"])
        groups.setdefault(key, []).append(row)
    return [dict(experiment=key[0], site_scale=key[1],
                 fixed_capacity_fee_yuan_per_gpu_capacity_hour=key[2],
                 policy_carbon_price_yuan_per_t=key[3], days=len(values),
                 **{field: float(np.mean([value[field] for value in values])) for field in fields})
            for key, values in groups.items()]


def run(output_dir, seed_start=20000, days=8):
    if days < 1:
        raise ValueError("days must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = GpuMarketConfig()
    scales, fixed_fees, carbon_prices = (.5, 1., 2.), (2., 3., 5.), (0., 100., 300., 600.)
    seeds = tuple(range(seed_start, seed_start + days))
    rows = []
    reference_calibration = None
    for seed in seeds:
        main_row, main_scenario = baseline_row(seed, base_config, "scale")
        reference_calibration = reference_calibration or main_scenario.calibration
        for scale in scales:
            row, _ = ((main_row, main_scenario) if scale == 1 else baseline_row(
                seed, replace(base_config, site_scale=scale), "scale"))
            rows.append(row)
        for fee in fixed_fees:
            row, scenario = baseline_row(seed, replace(
                base_config, fixed_compute_cost_yuan_per_gpu_capacity_hour=fee),
                "fixed_capacity_fee")
            assert np.array_equal(scenario.baseline_load_mw, main_scenario.baseline_load_mw)
            assert np.isclose(row["dispatch_system_cost_yuan"], main_row["dispatch_system_cost_yuan"])
            assert np.isclose(row["platform_electricity_bill_yuan"], main_row["platform_electricity_bill_yuan"])
            assert np.isclose(row["grid_carbon_emissions_t"], main_row["grid_carbon_emissions_t"])
            rows.append(row)
        for price in carbon_prices:
            row, scenario = baseline_row(seed, replace(
                base_config, carbon_price_yuan_per_t=price), "generation_carbon_price")
            assert np.array_equal(scenario.baseline_load_mw, main_scenario.baseline_load_mw)
            assert [job.work_cu for job in scenario.jobs] == [job.work_cu for job in main_scenario.jobs]
            rows.append(row)
    csv_path = output_dir / "尺度与成本敏感性.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    configuration = {
        "purpose": "Physical and economic sensitivity of original contracts; no policy training or model selection",
        "reference_seeds": list(seeds),
        "scope": "All contracts retain original time and location; these are accounting/dispatch sensitivities, not learned mechanism comparisons",
        "training_performed": False,
        "used_for_model_selection": False,
        "base_config": asdict(base_config),
        "site_scales": scales,
        "fixed_capacity_fees_yuan_per_gpu_capacity_hour": fixed_fees,
        "generation_carbon_prices_yuan_per_t": carbon_prices,
        "carbon_price_status": "Synthetic exogenous generation-side prices; 600 yuan/t is a stress test, not a market forecast",
        "internal_carbon_value_status": "100 yuan/t is a voluntary company valuation, not a cash carbon fee",
        "physical_unit": "H100-equivalent GPU-hour on homogeneous eight-GPU nodes",
        "source_access_date": "2026-10-29",
        "sources": [
            {"url": "https://docs.nvidia.com/dgx-superpod/design-guides/dgx-superpod-data-center-design-h100/latest/planning.html",
             "supports": "DGX H100 system has eight H100 GPUs and 10.2 kW maximum power; 8.0 kW active and 0.8 kW idle are study assumptions"},
            {"url": "https://arxiv.org/abs/2412.08602",
             "supports": "Measured eight-H100 HGX training node reached approximately 8.4 kW; not a measured average for this synthetic fleet"},
            {"url": "https://datacenters.google/efficiency/",
             "supports": "PUE definition and production scale; modeled PUE 1.15/1.25/1.35 is assumed"},
            {"url": "https://lambda.ai/instances",
             "supports": "Official H100 eight-GPU-instance scale USD 3.99/GPU-hour as accessed; 18 yuan/GPU-hour is a study assumption, not an FX conversion"},
        ],
        "calibration_status": "Hardware-referenced synthetic assumptions; not measured workloads or a completed market calibration",
        "fixed_cost_scope": "Daily cost of the complete existing installed fleet; batch-only modeled utilization is low and can make absolute profit negative",
        "cost_accounting": {
            "dispatch_system_cost": "generation fuel/resource cost + variable compute/network cost + user inconvenience",
            "full_system_cost": "dispatch_system_cost + common existing-capacity cost",
            "platform_contribution_before_fixed": "compute revenue - final-LMP electricity bill - variable compute/network cost",
            "platform_profit_after_fixed": "platform contribution - common existing-capacity cost",
            "break_even_fixed_fee": "platform contribution / (installed GPUs * 24 hours)",
            "common_cost_in_paired_changes": "The same installed-capacity cost appears in both schedules and cancels in absolute paired cost/profit changes; percentage denominators still differ",
            "carbon_cash": "Generation-side carbon payments enter market electricity prices; no extra platform carbon charge and no addition of transfer payments to system resource cost",
            "carbon_footprint": "DC consumption footprint uses flow-traced average intensity; actual grid emissions are separately reported",
        },
        "example_seed_calibration": reference_calibration,
        "group_means": summarize(rows),
        "csv_file": csv_path.name,
        "csv_rows": len(rows),
    }
    config_path = output_dir / "配置.json"
    config_path.write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": str(csv_path), "config": str(config_path), "rows": len(rows),
            "group_means": configuration["group_means"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/电碳机制与物理尺度"))
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--days", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.seed_start, args.days), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
