#!/usr/bin/env python3
"""Select and evaluate controls for one binding posted offer per task.

Selection uses training public days and independent private draws only.
The test stage never fits or chooses a template. It evaluates frozen user
models and may suppress explicit platform history without changing its
weights, normalisation, physical state or funded-action mask.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from energy_net.coordination.dynamic_tariffs import ExpandedAmendmentCatalogue
from energy_net.coordination.posted_offer_baselines import (
    FixedMatrixGreedy, FixedRawOffer, LearnedResponseGreedy,
    RandomFundedOffer, ValueShareGreedy,
)
from energy_net.coordination.preference_scenario import decode_preference_episode_id
from train_posted_offers import (
    bundle_for, load_users, mean, posted_episode_factory, preference_cases, rollouts,
)


SELECT_CASES = preference_cases(range(64), 2, 110000)
CALIBRATION_CASES = preference_cases(range(64), 2, 120000)
TEST_CASES = preference_cases(range(70000, 70032), 4, 300000)
VALUE_SHARES = (.15, .3, .45, .6, .75, .9)
OBJECTIVE = "procurement_net_saving_yuan"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                   default=lambda item: item.item() if isinstance(item, np.generic) else str(item)),
                         encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_inputs(legacy, users):
    selection = json.loads((legacy / "selection.json").read_text(encoding="utf-8"))
    result = dict(legacy_selection=file_hash(legacy / "selection.json"),
                  legacy_users={str(owner): file_hash(legacy / "selected" / f"user_{owner}.zip") for owner in range(3)},
                  users={str(owner): file_hash(users / f"user_{owner}.zip") for owner in range(3)})
    if selection["policy_kind"] == "ppo":
        result["legacy_platform"] = file_hash(legacy / selection.get("platform_model", "selected/platform.zip"))
    return result


def matrix_templates(config):
    codec = ExpandedAmendmentCatalogue(config)
    prototype = SimpleNamespace(config=config)
    result = []
    for index in range(codec.n_actions):
        raw = codec.decode(prototype, index)
        matrix = np.asarray(config.discount_levels)[raw[:12]].reshape(3, 4)
        result.append(dict(template_index=index, label=codec.labels[index],
                           matrix=matrix.tolist(), open=bool(raw[15])))
    return result


def matrix_controller(template):
    # Index0 means no publication; zero-discount but open index1 is a distinct
    # public greedy controller that can offer a free physical amendment.
    return FixedMatrixGreedy(template["matrix"]) if template["open"] else FixedRawOffer(0)


def enrich_day(row, scheme):
    result = dict(row, scheme=scheme)
    public, private = decode_preference_episode_id(int(row["case_id"]))
    result.update(public_seed=public, preference_seed=private,
        actual_G_yuan=float(row["flexibility_resource_saving_yuan"]),
        actual_R_yuan=float(row["additional_user_rebate_yuan"]),
        actual_M_yuan=float(row[OBJECTIVE]),
        actual_S_yuan=float(row["social_saving_vs_outside_yuan"]),
        actual_platform_gain_yuan=float(row["platform_gain_vs_outside_yuan"]),
        actual_user_gain_yuan=float(row["user_gain_yuan"]),
        actual_buyer_gain_yuan=float(row["buyer_post_settlement_benefit_yuan"]),
        same_cap_compliant=bool(row["carbon_emissions_t"] <= row["carbon_budget_t"] + 1e-6),
        old_profit_floor_satisfied=bool(row["platform_floor_shortfall_yuan"] <= 1e-6))
    if not np.isclose(result["actual_G_yuan"] - result["actual_R_yuan"], result["actual_M_yuan"], atol=1e-6, rtol=0):
        raise ValueError("Realised procurement net saving must equal accepted G minus paid R")
    if not np.isclose(result["actual_platform_gain_yuan"] + result["actual_user_gain_yuan"]
                      + result["actual_buyer_gain_yuan"], result["actual_S_yuan"], atol=1e-6, rtol=0):
        raise ValueError("Realised platform, user and buyer gains must sum to social saving")
    return result


def _private_test_rollouts(factory, bundle, cases, scheme, *, controller, model,
                           record_offers, private_diagnostics):
    """Read a private utility only after the actual contract is published."""
    rows, offers = [], []
    previous = bundle.training
    bundle.training = False
    try:
        for case in cases:
            episode = factory(case)
            day_private = []
            while not episode.done:
                action = (controller.choose(episode) if controller is not None else
                          int(model.predict(bundle.platform_observation(episode), deterministic=True)[0]))
                proposal = episode.preview_offer(action)
                episode.set_offer(action)
                prospective = None
                if proposal["offered"]:
                    # No alternative user response is queried. This number is
                    # retained in a separate evaluation record, never state.
                    job = episode.jobs[episode.position]
                    option = job.options[episode.current_offer["option_index"]]
                    prospective = float(episode.current_offer["R"]
                        - (option.user_loss_yuan - episode.outside_losses[episode.position]))
                response = bundle.user_action(episode)
                _, _, info = episode.step_accept(response)
                if prospective is not None:
                    positive = prospective > 1e-7
                    indifferent = abs(prospective) <= 1e-7
                    accepted = bool(info["chosen_accept"])
                    diagnostic = dict(case_id=int(case), scheme=scheme, job_id=info["job_id"],
                        user_id=int(info["user_id"]), offered_action=int(action),
                        private_evaluation_only=True, prospective_user_gain_yuan=prospective,
                        positive_utility_offer=positive, negative_utility_offer=prospective < -1e-7,
                        indifferent_utility_offer=indifferent, accepted=accepted,
                        positive_utility_rejection=positive and not accepted,
                        accepted_positive_utility=positive and accepted,
                        accepted_indifferent=indifferent and accepted,
                        actual_user_gain_yuan=float(info["utility_gain_yuan"]))
                    day_private.append(diagnostic)
                    private_diagnostics.append(diagnostic)
                if record_offers:
                    offers.append(dict(case_id=case, **info["public_feedback"],
                        G=proposal["G"], R=proposal["R"], M=proposal["M"],
                        platform_gain_yuan=proposal["platform_gain_yuan"] if info["chosen_accept"] else 0.))
            metrics = episode.outcome().metrics
            counts = dict(positive_utility_offers=sum(row["positive_utility_offer"] for row in day_private),
                positive_utility_rejections=sum(row["positive_utility_rejection"] for row in day_private),
                negative_utility_offers=sum(row["negative_utility_offer"] for row in day_private),
                indifferent_utility_offers=sum(row["indifferent_utility_offer"] for row in day_private),
                accepted_positive_utility_offers=sum(row["accepted_positive_utility"] for row in day_private),
                accepted_indifferent_offers=sum(row["accepted_indifferent"] for row in day_private))
            rows.append(dict(case_id=case, **metrics, **counts))
    finally:
        bundle.training = previous
    return rows, offers


def run_policy(factory, users, cases, scheme, *, controller=None, model=None,
               explicit_history=True, record_offers=False, private_diagnostics=None):
    bundle = bundle_for(users, history=explicit_history)
    if private_diagnostics is None:
        rows, offers = rollouts(factory, bundle, tuple(cases), controller=controller,
                                model=model, record_offers=record_offers)
    else:
        rows, offers = _private_test_rollouts(factory, bundle, tuple(cases), scheme,
            controller=controller, model=model, record_offers=record_offers,
            private_diagnostics=private_diagnostics)
    rows = [enrich_day(row, scheme) for row in rows]
    public_offers = []
    for row in offers:
        accepted = bool(row.get("accepted", False))
        # Offered values are not credited if the one actual user declines.
        public_offers.append(dict(row, scheme=scheme,
            offered_G_yuan=float(row["G"]), offered_R_yuan=float(row["R"]), offered_M_yuan=float(row["M"]),
            realised_G_yuan=float(row["G"]) if accepted else 0.,
            realised_R_yuan=float(row["R"]) if accepted else 0.,
            realised_M_yuan=float(row["M"]) if accepted else 0.,
            realised_platform_gain_yuan=float(row["platform_gain_yuan"])))
    return rows, public_offers


def query_counts(rows):
    return dict(day_case_rollouts=len(rows), resolved_task_responses=int(sum(row["jobs"] for row in rows)),
                actually_published_offers=int(sum(row["posted_offers"] for row in rows)),
                accepted_offer_responses=int(sum(row["accepted_offers"] for row in rows)),
                unique_private_cases=len({int(row["case_id"]) for row in rows}),
                unique_public_days=len({int(row["public_seed"]) for row in rows}))


def select_controls(factory, users, cases, calibration_cases, *, raw_actions=tuple(range(61)),
                    templates=None, rhos=VALUE_SHARES, seed=0, on_family=None,
                    candidate_evaluator=None):
    """Fit one implementable control per family using mean actual M only."""
    templates = matrix_templates(factory.config) if templates is None else templates
    records, summaries, selected = {}, {}, {}
    families = (
        ("raw", [(dict(action=int(action)), FixedRawOffer(action)) for action in raw_actions]),
        ("matrix", [(dict(template), matrix_controller(template)) for template in templates]),
        ("value_share", [(dict(rho=float(rho)), ValueShareGreedy(rho)) for rho in rhos]),
    )
    for family, candidates in families:
        family_rows, family_means = [], []
        parallel_rows = (None if candidate_evaluator is None else iter(candidate_evaluator(
            family,[parameters for parameters,_ in candidates],cases)))
        for order, (parameters, controller) in enumerate(candidates):
            print(json.dumps(dict(stage="train_selection", family=family,
                                   candidate=order, candidates=len(candidates))), flush=True)
            if parallel_rows is None:
                rows, _ = run_policy(factory, users, cases, f"{family}_{order}", controller=controller)
            else:
                rows = next(parallel_rows)
            for row in rows:
                row.update(candidate_index=order)
            stats = dict(mean(rows), candidate_index=order, parameters=parameters,
                         queries=query_counts(rows))
            family_rows.extend(rows)
            family_means.append(stats)
        eligible = [row for row in family_means if family != "raw" or row["parameters"]["action"] != 0]
        if not eligible:
            raise ValueError("The raw-action benchmark must include at least one non-exit action")
        best = max(eligible, key=lambda row: (row[OBJECTIVE], -row["candidate_index"]))
        selected[family] = dict(parameters=best["parameters"], training_mean_M_yuan=float(best[OBJECTIVE]),
            training_mean_profit_yuan=float(best["actual_platform_gain_yuan"]),
            training_mean_social_saving_yuan=float(best["actual_S_yuan"]),
            candidate_count=len(family_means), selected_candidate_index=best["candidate_index"],
            queries=query_counts(family_rows))
        records[family], summaries[family] = family_rows, family_means
        if on_family is not None:
            on_family(family, family_rows, family_means)
    print(json.dumps(dict(stage="random_public_feedback_calibration", cases=len(calibration_cases))), flush=True)
    random = RandomFundedOffer(np.random.default_rng(np.random.SeedSequence([int(seed), 831904])))
    calibration, feedback = run_policy(factory, users, calibration_cases, "random_calibration",
                                        controller=random, record_offers=True)
    prior = LearnedResponseGreedy.fit(feedback, enable_history=False)
    online_values = prior.to_dict()
    online_values["enable_history"] = True
    online = LearnedResponseGreedy.from_dict(online_values)
    report = dict(selection="one fixed parameter per family maximizing training mean realised M; ties lower candidate index; raw exit excluded from its selected benchmark",
                  selected=selected, calibration_queries=query_counts(calibration),
                  calibration_fit_samples_by_owner=list(prior.sample_counts),
                  logit_scope="Fit only actual public offered/replied records from random calibration. Refusals are probabilistic labels; no private preference bounds are inferred.",
                  all_selection_queries=dict(day_case_rollouts=sum(len(value) for value in records.values()) + len(calibration),
                      resolved_task_responses=sum(sum(row["jobs"] for row in value) for value in records.values()) + sum(row["jobs"] for row in calibration)))
    return report, records, summaries, calibration, feedback, prior, online


def test_controls(factory, users, cases, selection, prior, online, *, model=None, history=True,
                  extra_models=(), private_diagnostics=None):
    chosen = selection["selected"]
    controls = [
        ("exit", FixedRawOffer(0)),
        ("strong_fixed_raw", FixedRawOffer(chosen["raw"]["parameters"]["action"])),
        ("strong_fixed_matrix_greedy_route", matrix_controller(chosen["matrix"]["parameters"])),
        ("strong_value_share_rule", ValueShareGreedy(chosen["value_share"]["parameters"]["rho"])),
        ("learned_response_prior", prior), ("learned_response_online_map", online),
    ]
    rows, offers = [], []
    for scheme, controller in controls:
        day_rows, public = run_policy(factory, users, cases, scheme, controller=controller, record_offers=True,
                                      private_diagnostics=private_diagnostics)
        rows.extend(day_rows)
        offers.extend(public)
        print(json.dumps(dict(stage="frozen_test", scheme=scheme, **query_counts(day_rows))), flush=True)
    policy_runs = [] if model is None else [("ppo", model, bool(history)),
        ("ppo_same_weights_zero_explicit_history", model, False)]
    reserved = {name for name, _ in controls} | {name for name, _, _ in policy_runs}
    for name, extra_model, enabled in extra_models:
        if name in reserved:
            raise ValueError("Every extra-model label must be distinct from the other schemes")
        reserved.add(name)
        policy_runs.append((name, extra_model, bool(enabled)))
    for scheme, current_model, enabled in policy_runs:
        day_rows, public = run_policy(factory, users, cases, scheme, model=current_model,
                                      explicit_history=enabled, record_offers=True,
                                      private_diagnostics=private_diagnostics)
        rows.extend(day_rows)
        offers.extend(public)
        print(json.dumps(dict(stage="frozen_test", scheme=scheme, **query_counts(day_rows))), flush=True)
    summaries = []
    for scheme in dict.fromkeys(row["scheme"] for row in rows):
        group = [row for row in rows if row["scheme"] == scheme]
        summaries.append(dict(scheme=scheme, **mean(group), **query_counts(group)))
    return rows, offers, summaries


# Prevent pytest from mistaking this public evaluation helper for a test case.
test_controls.__test__ = False


def summarise_private_responses(records, schemes):
    result = []
    for scheme in schemes:
        rows = [row for row in records if row["scheme"] == scheme]
        positive = sum(row["positive_utility_offer"] for row in rows)
        rejected_positive = sum(row["positive_utility_rejection"] for row in rows)
        result.append(dict(scheme=scheme, private_evaluation_only=True, actual_offers=len(rows),
            positive_utility_offers=positive, positive_utility_rejections=rejected_positive,
            negative_utility_offers=sum(row["negative_utility_offer"] for row in rows),
            indifferent_utility_offers=sum(row["indifferent_utility_offer"] for row in rows),
            accepted_positive_utility_offers=sum(row["accepted_positive_utility"] for row in rows),
            accepted_indifferent_offers=sum(row["accepted_indifferent"] for row in rows),
            positive_utility_rejection_fraction=rejected_positive / positive if positive else 0.))
    return result


def manifest_for(legacy, users, config, seed):
    return dict(protocol="one binding posted offer per task", config=asdict(config), seed=int(seed),
                frozen_inputs=frozen_inputs(legacy, users), selection_cases=list(SELECT_CASES),
                calibration_cases=list(CALIBRATION_CASES), raw_actions=list(range(61)),
                matrix_template_count=204, value_shares=list(VALUE_SHARES),
                objective=OBJECTIVE, matrix_zero_policy="closed index0 exits; open index1 is free greedy routing",
                quantization="allocation-weighted rebate, nearest among0/.0025/.005/.01/.02; ties smaller rebate; no unfunded quote repair")


def load_platform(directory, users_path, checkpoint):
    metadata = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    expected_users = directory / "best_joint" if checkpoint == "best_joint" else Path(metadata["users_path"])
    if any(file_hash(expected_users / f"user_{owner}.zip") != file_hash(users_path / f"user_{owner}.zip")
           for owner in range(3)):
        raise ValueError("Platform checkpoint and evaluation baseline use different user models")
    path = directory / checkpoint / "platform.zip"
    return PPO.load(path, device="cpu"), bool(metadata.get("history", True)), metadata, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select", "test"), required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--users", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--extra-model", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--checkpoint", choices=("best_fixed", "best_joint"), default="best_fixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1,
                        help="Spawned processes for deterministic select templates; random calibration and test remain serial")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    torch.set_num_threads(1)
    legacy, users_path, output = args.legacy.resolve(), args.users.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    users_ready = ((users_path / "summary.json").exists() or
                   (users_path.name == "best_joint" and (users_path.parent / "summary.json").exists()))
    if not users_ready:
        raise FileNotFoundError(f"User adaptation is not complete: {users_path / 'summary.json'}")
    started = time.monotonic()
    # Constructing the factory does not request any public/private case.
    factory = posted_episode_factory(legacy)
    expected = manifest_for(legacy, users_path, factory.config, args.seed)
    manifest_path, selection_path = output / "selection_manifest.json", output / "selection.json"
    if args.stage == "select":
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != json.loads(json.dumps(expected)):
            raise ValueError("Selection output belongs to different users, cases, rules or legacy contracts")
        if selection_path.exists():
            if not manifest_path.exists():
                raise ValueError("A completed selection needs its frozen-input manifest")
            print(json.dumps(dict(selection_already_complete=str(selection_path))), flush=True)
            return
        write_json(manifest_path, expected)
        users = load_users(users_path)
        def checkpoint(family, rows, summaries):
            write_csv(output / f"{family}_training_outcomes.csv", rows)
            write_csv(output / f"{family}_training_means.csv", summaries)
        if args.workers == 1:
            selected_controls = select_controls(
                factory, users, SELECT_CASES, CALIBRATION_CASES, seed=args.seed, on_family=checkpoint)
        else:
            from energy_net.coordination.posted_offer_parallel import SpawnedTemplateEvaluator
            with SpawnedTemplateEvaluator(legacy,users_path,config=factory.config,
                                          workers=args.workers) as evaluator:
                selected_controls = select_controls(
                    factory,users,SELECT_CASES,CALIBRATION_CASES,seed=args.seed,
                    on_family=checkpoint,candidate_evaluator=evaluator)
        result, records, summaries, calibration, feedback, prior, online = selected_controls
        prior.to_json(output / "logit_prior.json")
        online.to_json(output / "logit_online_map.json")
        write_csv(output / "random_calibration_outcomes.csv", calibration)
        write_json(output / "random_calibration_public_feedback.json", feedback)
        if frozen_inputs(legacy, users_path) != expected["frozen_inputs"]:
            raise RuntimeError("A frozen model changed during baseline selection")
        result.update(stage="selection_complete", elapsed_seconds=time.monotonic() - started,
                      train_case_count=len(SELECT_CASES), calibration_case_count=len(CALIBRATION_CASES),
                      users_path=str(users_path), legacy_path=str(legacy), test_cases_used=[])
        write_json(selection_path, result)
        print(json.dumps(dict(completed_selection=str(selection_path), selected=result["selected"])), flush=True)
        return
    # Test never fills in a missing selection or changes its parameters.
    if not manifest_path.exists() or not selection_path.exists():
        raise FileNotFoundError("Run training-only --stage select before frozen test evaluation")
    if json.loads(manifest_path.read_text(encoding="utf-8")) != json.loads(json.dumps(expected)):
        raise ValueError("Test users/cases/rules do not match this selected baseline cohort")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    prior = LearnedResponseGreedy.from_json(output / "logit_prior.json")
    online = LearnedResponseGreedy.from_json(output / "logit_online_map.json")
    model, history, model_metadata = None, True, None
    models_before, extra_models, extra_metadata = {}, [], {}
    if args.model_dir is not None:
        model_dir = args.model_dir.resolve()
        model, history, model_metadata, path = load_platform(model_dir, users_path, args.checkpoint)
        models_before[str(path)] = file_hash(path)
    for value in args.extra_model:
        if "=" not in value:
            raise ValueError("An extra model must be NAME=DIR")
        name, directory = value.split("=", 1)
        if not name:
            raise ValueError("An extra model needs a nonempty scheme label")
        directory = Path(directory).resolve()
        extra, enabled, metadata, path = load_platform(directory, users_path, args.checkpoint)
        extra_models.append((name, extra, enabled))
        extra_metadata[name] = dict(directory=str(directory), history=enabled, metadata=metadata)
        models_before[str(path)] = file_hash(path)
    private_diagnostics = []
    rows, offers, summaries = test_controls(factory, load_users(users_path), TEST_CASES,
                                           selection, prior, online, model=model, history=history,
                                           extra_models=extra_models, private_diagnostics=private_diagnostics)
    if frozen_inputs(legacy, users_path) != expected["frozen_inputs"]:
        raise RuntimeError("A frozen model changed during test evaluation")
    if any(file_hash(Path(path)) != value for path, value in models_before.items()):
        raise RuntimeError("A platform checkpoint changed during frozen evaluation")
    write_csv(output / "test_outcomes.csv", rows)
    write_json(output / "test_actual_offer_history.json", offers)
    write_csv(output / "test_summary.csv", summaries)
    private_summary = summarise_private_responses(private_diagnostics, [row["scheme"] for row in summaries])
    write_csv(output / "test_private_response_diagnostics.csv", private_diagnostics)
    write_csv(output / "test_private_response_summary.csv", private_summary)
    write_json(output / "test_report.json", dict(protocol="single binding offer; same old bookings and same frozen users in every comparison",
        selection=selection, cases=list(TEST_CASES), results=summaries,
        model_dir=None if args.model_dir is None else str(args.model_dir.resolve()), model_metadata=model_metadata,
        checkpoint=args.checkpoint, extra_models=extra_metadata, platform_files_sha256=models_before,
        private_response_summary=private_summary,
        explicit_history_ablation="same platform weights and normalization; only explicit history features zeroed; current physical state and admissible-action masks retained",
        private_accounts="S/user gains are aggregate outcome accounts. Prospective private utility is read only for the one actual published offer and stored separately; it never enters any controller, observer or public-response fit.",
        elapsed_seconds=time.monotonic() - started))
    print(json.dumps(dict(completed_test=str(output), summaries=summaries)), flush=True)


if __name__ == "__main__":
    main()
