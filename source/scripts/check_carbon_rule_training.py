#!/usr/bin/env python3
"""Check two frozen controls on ALL TRAIN; never select or open test cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from carbon_offer_experiment import (
    ROOT, RULE_TRAIN, TRAIN, bundle_for, evaluate, hashes, inputs, load_users,
    mean, posted_episode_factory, write_csv, write_json,
)
from energy_net.coordination.carbon_offer_baselines import CarbonResponseGreedy


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(output):
    started = time.monotonic()
    reference_path = output / "reference.json"
    selection_path = output / "rule_selection.json"
    selected_path = output / "carbon_rule.json"
    source_paths = (reference_path, selection_path, selected_path,
                    output / "reference_train_days.csv")
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    target = output / "rule_all_train_check.json"
    if target.exists():
        raise FileExistsError(f"ALL TRAIN check already exists; no repeated evaluation: {target}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert reference["cases"] == len(TRAIN) == 2048
    before = hashes()
    if before != reference["original_input_sha256"]:
        raise ValueError("Frozen original users/platform/legacy inputs differ from the TRAIN reference")
    source_hashes = {path.name: sha256(path) for path in source_paths}
    selected = CarbonResponseGreedy.from_json(selected_path)
    if selected.cash_limit or selected.lambda_profit != 4.:
        raise ValueError("This check was authorized for the frozen selected lambda=4 control")
    selected_dict = selected.to_dict()
    # The limiting control uses the identical frozen response prior/MAP method.
    # This changes neither the selected file nor any fitted response parameter.
    limit_dict = dict(selected_dict, cash_limit=True, lambda_profit=None)
    cash_limit = CarbonResponseGreedy.from_dict(limit_dict)
    controllers = (("carbon_rule_lambda_4", selected), ("cash_limit", cash_limit))
    factory = posted_episode_factory(inputs()["legacy"], public_cache_size=256, case_cache_size=256)
    bundle = bundle_for(load_users(inputs()["users"]))
    required = float(reference["required_profit_yuan"])
    results, all_days = [], []
    for label, controller in controllers:
        parameters = controller.to_dict()
        rows, _ = evaluate(factory, bundle, TRAIN, controller=controller,
                           label=label + "_ALL_TRAIN", offers=False)
        if len(rows) != len(TRAIN) or {int(row["case_id"]) for row in rows} != set(TRAIN):
            raise ValueError("Expected exactly the declared ALL TRAIN cases")
        stats = mean(rows)
        profit = float(stats["platform_gain_vs_outside_yuan"])
        gap = required - profit
        results.append(dict(method=label, controller=parameters, cases=len(rows),
            public_weather_days=256, private_draws_per_day=8,
            mean_platform_gain_yuan=profit,
            mean_grid_emissions_t=float(stats["carbon_emissions_t"]),
            mean_grid_emissions_reduction_t=float(stats["grid_emissions_reduction_t"]),
            mean_social_saving_yuan=float(stats["social_saving_vs_outside_yuan"]),
            mean_procurement_net_saving_yuan=float(stats["procurement_net_saving_yuan"]),
            mean_dc_footprint_change_t=float(stats["delta_dc_consumption_emissions_t"]),
            mean_user_ir_rate=float(stats["user_ir_rate"]),
            mean_completion_rate=float(stats["completion_rate"]),
            required_profit_yuan=required, profit_gap_yuan=gap,
            profit_shortfall_yuan=max(0., gap),
            profit_qualified_full_train=bool(profit >= required - 1e-6),
            retained_reference_profit_fraction=profit / float(reference["reference_profit_yuan"])))
        all_days.extend(dict(row, method=label, split="TRAIN", evaluation_scope="ALL_TRAIN") for row in rows)
        write_csv(output / "rule_all_train_days.csv", all_days)
        if controller.to_dict() != parameters:
            raise RuntimeError("Frozen control parameters changed during evaluation")
        print(json.dumps(dict(stage="full_train_method_complete", **{k:v for k,v in results[-1].items()
                         if k != "controller"}), ensure_ascii=False, allow_nan=False), flush=True)
    if hashes() != before or any(sha256(path) != source_hashes[path.name] for path in source_paths):
        raise RuntimeError("A frozen reference/control/input file changed during evaluation")
    result = dict(status="complete", split="TRAIN", evaluation_scope="ALL_TRAIN",
        root_seed=0, users="matched original best_joint users", cases_per_method=len(TRAIN),
        reference_source=str(reference_path.relative_to(ROOT)),
        reference_days_source=str((output / "reference_train_days.csv").relative_to(ROOT)),
        reference_profit_yuan=float(reference["reference_profit_yuan"]),
        profit_fraction=float(reference["q"]), required_profit_yuan=required,
        qualification_tolerance_yuan=1e-6,
        original_subset_selection=dict(source=str(selection_path.relative_to(ROOT)),
            cases=len(RULE_TRAIN), feasible=selection["feasible"],
            profit_shortfall_yuan=selection["profit_shortfall_yuan"],
            selected_index=selection["selected_index"],
            selected_lambda_profit=selection["selected"]["lambda_profit"],
            flag_preserved=True),
        results=results, test_cases_used=[], controller_reselected=False,
        original_input_sha256=before, original_source_files_sha256=source_hashes,
        elapsed_seconds=time.monotonic()-started)
    write_json(target, result)
    print(json.dumps(dict(stage="all_train_check_complete", output=str(target),
                         elapsed_seconds=result["elapsed_seconds"]), ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "carbon-aware" / "seed0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    check(args.output.resolve())


if __name__ == "__main__":
    main()
