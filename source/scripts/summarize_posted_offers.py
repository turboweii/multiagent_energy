#!/usr/bin/env python3
"""Summarise frozen posted-offer comparisons and draw standalone figures.

Example:
  python scripts/summarize_posted_offers.py --experiment /path/to/seed0

Read baselines-initial, optionally baselines-joint, and the full-history /
no-history training directories. Private draws are first averaged within each
public weather day. Intervals therefore describe weather cases conditional
on the fitted policies, not repeated algorithm seeds. No policy is selected,
updated or evaluated by this script.
"""
from __future__ import annotations

import argparse
import json
from math import isqrt
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


MATRIX = "strong_fixed_matrix_greedy_route"
RULES = ("strong_fixed_raw", MATRIX, "strong_value_share_rule",
         "learned_response_prior", "learned_response_online_map")
ZERO_HISTORY = "ppo_same_weights_zero_explicit_history"
LABELS = {
    "exit": "Old contracts / no new offer",
    "strong_fixed_raw": "Train-selected fixed offer",
    MATRIX: "Fixed matrix + greedy routing",
    "strong_value_share_rule": "Resource-value share rule",
    "learned_response_prior": "Logit response prior",
    "learned_response_online_map": "Logit + online MAP",
    ZERO_HISTORY: "PPO / same weights, history zeroed",
}
COLORS = {
    "exit": "#949494", "strong_fixed_raw": "#6b7b8a", MATRIX: "#355c7d",
    "strong_value_share_rule": "#8065a5", "learned_response_prior": "#b29140",
    "learned_response_online_map": "#ce7534", "ppo": "#087f8c",
    ZERO_HISTORY: "#71aaa9", "no_history": "#a65d4b",
}
ALIASES = {
    "M": ("actual_M_yuan", "procurement_net_saving_yuan", "net_distributable_surplus_yuan"),
    "S": ("actual_S_yuan", "social_saving_vs_outside_yuan", "contract_system_cost_saving_yuan"),
    "profit": ("actual_platform_gain_yuan", "platform_gain_vs_outside_yuan"),
    "G": ("actual_G_yuan", "flexibility_resource_saving_yuan"),
    "R": ("actual_R_yuan", "additional_user_rebate_yuan"),
    "user_gain": ("actual_user_gain_yuan", "user_gain_yuan"),
    "buyer_gain": ("actual_buyer_gain_yuan", "buyer_post_settlement_benefit_yuan"),
    "old_profit": ("outside_platform_profit_yuan",),
    "new_profit": ("platform_profit_yuan",),
    "fixed_cost": ("fixed_compute_cost_yuan",),
    "work": ("work_gpu_hours", "work_cu"),
    "revenue": ("service_revenue_yuan",),
    "outside_revenue": ("outside_service_revenue_yuan",),
    "accepted": ("accepted_offers", "accepted_amendments"),
    "offered": ("posted_offers", "offered_jobs"),
}
WEATHER_METRICS = tuple(ALIASES) + ("pre_fixed_surplus", "utilisation", "average_bill_price",
                                    "ledger_cover_price")
PAIR_METRICS = ("M", "S", "profit", "G", "R", "user_gain", "buyer_gain", "accepted", "offered")


def decode_case(value):
    """Exact negative Cantor address, matching preference_scenario.py."""
    value = int(value)
    if value >= 0:
        raise ValueError("Expected an encoded private-preference case_id")
    paired = -value - 1
    diagonal = (isqrt(8 * paired + 1) - 1) // 2
    private = paired - diagonal * (diagonal + 1) // 2
    return diagonal - private, private


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def finite_number(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt(value, places=2):
    return f"{float(value):,.{places}f}" if finite_number(value) else "未记录"


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(mean=np.nan, low=np.nan, high=np.nan, half=np.nan, n=0)
    average = float(values.mean())
    half = (float(student_t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
            if len(values) > 1 else np.nan)
    return dict(mean=average, low=average-half, high=average+half, half=half, n=len(values))


def is_no_history(scheme, cohort):
    extra = cohort["metadata"].get("extra_models", {}).get(scheme, {})
    if extra:
        return extra.get("history") is False
    lower = scheme.lower()
    return scheme != ZERO_HISTORY and any(token in lower for token in
        ("nohist", "no_history", "no-history", "without_history"))


def scheme_label(scheme, cohort):
    if scheme in LABELS:
        return LABELS[scheme]
    if scheme == "ppo":
        enabled = (cohort["metadata"].get("model_metadata") or {}).get("history", True)
        return "PPO / full explicit history" if enabled else "PPO / trained without explicit history"
    if is_no_history(scheme, cohort):
        return f"PPO / trained without history [{scheme}]"
    return scheme


def scheme_color(scheme, cohort):
    return COLORS["no_history"] if is_no_history(scheme, cohort) else COLORS.get(scheme, "#33807a")


def capacity_gpu_hours(config):
    capacity = config.get("capacity_cu_per_hour")
    if capacity is None:
        return np.nan
    return float(sum(capacity) * config.get("n_steps", 48) * config.get("dt_hours", .5))


def load_cohort(directory, name):
    required = ("test_summary.csv", "test_outcomes.csv", "test_private_response_summary.csv")
    for filename in required:
        if not (directory / filename).exists():
            raise FileNotFoundError(directory / filename)
    summaries = pd.read_csv(directory / required[0])
    frame = pd.read_csv(directory / required[1], dtype={"case_id": "int64"}).copy()
    private = pd.read_csv(directory / required[2])
    if frame.empty:
        raise ValueError(f"No evaluated outcomes in {directory}")
    frame["scheme"] = frame.scheme.astype(str)
    if frame.duplicated(["scheme", "case_id"]).any():
        raise ValueError(f"Repeated scheme/case rows in {directory}")
    addresses = [decode_case(case) for case in frame.case_id]
    frame["public_seed"] = [public for public, _ in addresses]
    frame["preference_seed"] = [private_seed for _, private_seed in addresses]
    metadata = read_json(directory / "test_report.json")
    manifest = read_json(directory / "selection_manifest.json")
    config = (metadata.get("model_metadata") or {}).get("config") or manifest.get("config", {})
    capacity_hours = capacity_gpu_hours(config)
    for canonical, options in ALIASES.items():
        source = next((column for column in options if column in frame), None)
        if source is None:
            raise ValueError(f"Missing {canonical} account in {directory}")
        frame[canonical] = pd.to_numeric(frame[source], errors="raise")
    frame["pre_fixed_surplus"] = frame.new_profit + frame.fixed_cost
    frame["utilisation"] = frame.work / capacity_hours if finite_number(capacity_hours) else np.nan
    frame["average_bill_price"] = frame.revenue / frame.work
    # Static ledger coverage, holding this realised execution and settlement
    # fixed. This does not re-solve demand or the purchase contract at a new price.
    frame["ledger_cover_price"] = (frame.revenue - frame.new_profit) / frame.work
    weather = frame.groupby(["scheme", "public_seed"], sort=False)[list(WEATHER_METRICS)].mean()
    weather["private_draws"] = frame.groupby(["scheme", "public_seed"]).case_id.nunique()
    initial_order = list(dict.fromkeys(summaries.scheme.astype(str).tolist() + frame.scheme.tolist()))
    initial_order = [scheme for scheme in initial_order if scheme in set(frame.scheme)]
    baseline_order = {scheme: i for i, scheme in enumerate(("exit", *RULES))}
    def order(scheme):
        if scheme in baseline_order:
            return baseline_order[scheme]
        if scheme == "ppo":
            return 6
        if is_no_history(scheme, {"metadata": metadata}):
            return 7
        if scheme == ZERO_HISTORY:
            return 8
        return 9
    schemes = sorted(initial_order, key=lambda scheme: (order(scheme), initial_order.index(scheme)))
    cohort = dict(name=name, directory=directory, summaries=summaries, outcomes=frame,
                  private=private, metadata=metadata, config=config, weather=weather,
                  capacity_gpu_hours=capacity_hours, schemes=schemes,
                  selection=read_json(directory / "selection.json"))
    # Diversity is descriptive and comes only from actually published offers.
    diversity = {}
    actual = pd.DataFrame()
    history_path = directory / "test_actual_offer_history.json"
    if history_path.exists():
        actual = pd.DataFrame(json.loads(history_path.read_text(encoding="utf-8")))
        if not actual.empty and {"scheme", "case_id", "action", "offered"} <= set(actual):
            offered = actual[actual.offered.astype(str).str.lower().isin(("true", "1", "1.0"))]
            counts = offered.groupby(["scheme", "case_id"]).action.nunique()
            for scheme in schemes:
                cases = frame.loc[frame.scheme == scheme, "case_id"]
                values = [int(counts.get((scheme, int(case)), 0)) for case in cases]
                diversity[scheme] = float(np.mean(values))
    cohort["diversity"] = diversity
    cohort["actual_offer_history"] = actual
    cohort["price_history_available"] = history_path.exists()
    return cohort


def weather_stats(cohort, scheme, metric):
    return interval(cohort["weather"].xs(scheme, level="scheme")[metric].to_numpy())


def private_counts(cohort, scheme):
    rows = cohort["private"].loc[cohort["private"].scheme == scheme]
    if rows.empty:
        return 0, 0
    return int(rows.positive_utility_offers.sum()), int(rows.positive_utility_rejections.sum())


def published_offer_prices(cohort):
    """Only observed, published contracts; no-offer events have no price."""
    history = cohort["actual_offer_history"]
    if history.empty:
        return pd.DataFrame()
    needed = {"scheme", "case_id", "user_id", "action", "offered", "rebate_rate",
              "region", "start_step", "job_id"}
    if not needed <= set(history):
        raise ValueError(f"Actual-offer log lacks price fields: {sorted(needed - set(history))}")
    mask = history.offered.astype(str).str.lower().isin(("true", "1", "1.0"))
    data = history.loc[mask].copy()
    if data.empty:
        return data
    data["offer_id"] = np.arange(len(data))
    for name in ("user_id", "region", "case_id", "start_step", "action"):
        data[name] = pd.to_numeric(data[name], errors="raise").astype("int64")
    data["rebate_rate"] = pd.to_numeric(data.rebate_rate, errors="raise").round(10)
    data["public_seed"] = [decode_case(case)[0] for case in data.case_id]
    data["accepted_flag"] = data.accepted.astype(str).str.lower().isin(("true", "1", "1.0"))
    data["start_hour"] = data.start_step * float(cohort["config"].get("dt_hours", .5))
    data["duration_hours"] = float(cohort["config"].get("batch_duration_hours", 4.))
    return data


def price_sample_record(cohort, scheme, data, row_type, **identifiers):
    """Descriptive frequencies, never a score for policy selection."""
    counts = data.rebate_rate.value_counts().sort_index() if not data.empty else pd.Series(dtype=int)
    total = int(len(data))
    modal = (float(counts.loc[counts == counts.max()].index.min()) if total else np.nan)
    proportions = counts.to_numpy(dtype=float) / total if total else np.array([])
    compensation = (float(data.rebate_yuan_per_gpuh.mean())
                    if total and "rebate_yuan_per_gpuh" in data else np.nan)
    return dict(cohort=cohort["name"], scheme=scheme, row_type=row_type,
        history_status="observed" if cohort["price_history_available"] else "history_not_provided",
        n_offers=total if cohort["price_history_available"] else np.nan,
        n_accepted=int(data.accepted_flag.sum()) if total else 0,
        n_public_weather=int(data.public_seed.nunique()) if total else 0,
        n_private_cases=int(data.case_id.nunique()) if total else 0,
        n_rate_levels=len(counts), mean_rate=float(data.rebate_rate.mean()) if total else np.nan,
        rate_std=float(data.rebate_rate.std(ddof=0)) if total else np.nan,
        modal_rate=modal, modal_frequency=float(counts.max()/total) if total else np.nan,
        rate_entropy_nats=float(-np.sum(proportions*np.log(proportions))) if total else np.nan,
        mean_rebate_yuan_per_gpuh=compensation,
        rate_counts_json=json.dumps({f"{rate:.10g}": int(count) for rate, count in counts.items()}),
        n_contract_actions=int(data.action.nunique()) if total else 0,
        n_target_regions=int(data.region.nunique()) if total else 0,
        **identifiers)


def offer_price_patterns(cohorts):
    """Within-owner rates and execution-block price/support descriptions.

    An offer touching two blocks is counted once in each touched block.
    The weighted mean uses execution-hour overlap, not GPU-work weights:
    per-task GPU quantities are absent from the public offer log. Sample
    counts here are quoted contracts, not independent weather observations.
    """
    records = []
    for cohort in cohorts:
        data = published_offer_prices(cohort)
        if not data.empty:
            block_records = []
            for row in data.to_dict(orient="records"):
                start, duration = row["start_hour"], row["duration_hours"]
                for block in range(4):
                    overlap = max(0., min(start+duration, 6.*(block+1)) - max(start, 6.*block))
                    if overlap > 1e-10:
                        block_records.append(dict(row, execution_block=block,
                            execution_overlap_hours=overlap, equivalent_offer_weight=overlap/duration))
            blocks = pd.DataFrame(block_records)
        else:
            blocks = pd.DataFrame()
        for scheme in cohort["schemes"]:
            scheme_data = data.loc[data.scheme == scheme] if not data.empty else pd.DataFrame()
            for owner in range(3):
                own = scheme_data.loc[scheme_data.user_id == owner] if not scheme_data.empty else pd.DataFrame()
                records.append(price_sample_record(cohort, scheme, own, "owner_summary", user_id=owner))
                if not own.empty:
                    for rate, group in own.groupby("rebate_rate", sort=True):
                        records.append(price_sample_record(cohort, scheme, group,
                            "owner_rate_frequency", user_id=owner, rate=float(rate),
                            rate_percent=float(rate*100), fraction_within_owner=len(group)/len(own)))
                    for job_id, group in own.groupby("job_id", sort=True):
                        records.append(price_sample_record(cohort, scheme, group,
                            "owner_job_slot_summary", user_id=owner, job_id=job_id))
                    for public, group in own.groupby("public_seed", sort=True):
                        records.append(price_sample_record(cohort, scheme, group,
                            "owner_weather_summary", user_id=owner, public_seed=int(public)))
            # Keep all scheme/region/block cells, including zero-support cells.
            for owner in (None, 0, 1, 2):
                for region in range(3):
                    for block in range(4):
                        if blocks.empty:
                            group = pd.DataFrame()
                        else:
                            mask = ((blocks.scheme == scheme) & (blocks.region == region)
                                    & (blocks.execution_block == block))
                            if owner is not None:
                                mask &= blocks.user_id == owner
                            group = blocks.loc[mask]
                        record = price_sample_record(cohort, scheme, group,
                            "region_execution_block" if owner is None else "owner_region_execution_block",
                            user_id=owner, target_region=region, execution_block=block,
                            block_start_hour=6*block, block_end_hour=6*(block+1))
                        overlap = float(group.execution_overlap_hours.sum()) if not group.empty else 0.
                        record.update(
                            n_touching_contracts=len(group) if cohort["price_history_available"] else np.nan,
                            execution_overlap_hours=overlap,
                            equivalent_full_offer_count=float(group.equivalent_offer_weight.sum()) if not group.empty else 0.,
                            mean_rate_execution_overlap_weighted=float(np.sum(group.rebate_rate*group.execution_overlap_hours)/overlap)
                                if overlap > 0 else np.nan,
                            block_weighting="execution hours, not GPU-work; contracts counted in each block touched")
                        records.append(record)
    return pd.DataFrame(records)


def paired_weather_rows(cohorts):
    records = []
    for cohort in cohorts:
        schemes = cohort["schemes"]
        if MATRIX not in schemes:
            raise ValueError(f"The predeclared fixed-matrix reference is missing in {cohort['directory']}")
        pairs = {}
        for scheme in schemes:
            if scheme != MATRIX:
                pairs[(scheme, MATRIX)] = "vs_training_selected_matrix"
        methods = [scheme for scheme in schemes if scheme not in ("exit", *RULES)]
        for method in methods:
            for reference in RULES:
                if reference in schemes:
                    pairs.setdefault((method, reference), "policy_vs_prespecified_rule")
        if "ppo" in schemes:
            for other in methods:
                if other != "ppo":
                    pairs[("ppo", other)] = "history_policy_contrast"
        for (scheme, reference), category in pairs.items():
            left = cohort["outcomes"].loc[cohort["outcomes"].scheme == scheme]
            right = cohort["outcomes"].loc[cohort["outcomes"].scheme == reference]
            if set(left.case_id) != set(right.case_id):
                raise ValueError(f"Unpaired private/weather cases: {scheme} versus {reference}")
            a = cohort["weather"].xs(scheme, level="scheme")
            b = cohort["weather"].xs(reference, level="scheme")
            for public in sorted(a.index):
                row = dict(cohort=cohort["name"], scheme=scheme, reference_scheme=reference,
                           contrast=category, public_seed=int(public),
                           private_draws_scheme=int(a.loc[public, "private_draws"]),
                           private_draws_reference=int(b.loc[public, "private_draws"]))
                for metric in PAIR_METRICS:
                    row[f"scheme_{metric}"] = float(a.loc[public, metric])
                    row[f"reference_{metric}"] = float(b.loc[public, metric])
                    row[f"delta_{metric}"] = float(a.loc[public, metric] - b.loc[public, metric])
                records.append(row)
    return pd.DataFrame(records)


def plot_comparison(cohorts, output, dpi):
    entries = [(cohort, scheme) for cohort in cohorts for scheme in cohort["schemes"]]
    labels = [textwrap.fill(f"{cohort['name']} | {scheme_label(scheme, cohort)}", 38)
              for cohort, scheme in entries]
    y = np.arange(len(entries))
    colors = [scheme_color(scheme, cohort) for cohort, scheme in entries]
    height = max(10., .58 * len(entries) + 3.)
    fig, axes = plt.subplots(2, 3, figsize=(21, height), sharey=True, constrained_layout=True)
    panels = ((axes[0, 0], "M", "M: realised procurement net saving"),
              (axes[0, 1], "S", "S: true social cost saving"),
              (axes[0, 2], "profit", r"Cash $\Delta\Pi$ (= 0.8 M by settlement)"))
    for axis, metric, title in panels:
        stats = [weather_stats(cohort, scheme, metric) for cohort, scheme in entries]
        values = np.array([item["mean"] for item in stats]) / 1000
        error = np.nan_to_num([item["half"] for item in stats], nan=0.) / 1000
        axis.barh(y, values, xerr=error, color=colors, error_kw=dict(elinewidth=1., capsize=2))
        axis.set(title=title, xlabel="Thousand yuan / day")
    positive, negative = np.zeros(len(entries)), np.zeros(len(entries))
    for metric, label, color in (("profit", "Platform", "#087f8c"),
            ("user_gain", "Users", "#8eae5b"), ("buyer_gain", "Procurement account", "#c38d38")):
        values = np.array([weather_stats(c, s, metric)["mean"] for c, s in entries]) / 1000
        axes[1, 0].barh(y, values, left=np.where(values >= 0, positive, negative), label=label, color=color)
        positive += np.maximum(values, 0.)
        negative += np.minimum(values, 0.)
    axes[1, 0].set(title="Allocation of true social saving S", xlabel="Thousand yuan / day")
    axes[1, 0].legend(fontsize=9, loc="lower right")
    accepted = [weather_stats(c, s, "accepted")["mean"] for c, s in entries]
    axes[1, 1].barh(y, accepted, color=colors)
    axes[1, 1].set(title="Accepted offers (descriptive)", xlabel="Accepted tasks / day")
    old = np.array([weather_stats(c, s, "old_profit")["mean"] for c, s in entries]) / 1000
    new = np.array([weather_stats(c, s, "new_profit")["mean"] for c, s in entries]) / 1000
    axes[1, 2].barh(y-.17, old, height=.31, color="#aaa9a8", label="Old absolute profit")
    axes[1, 2].barh(y+.17, new, height=.31, color=colors, label="New absolute profit")
    axes[1, 2].set(title="Absolute profit, including fixed fleet cost", xlabel="Thousand yuan / day")
    axes[1, 2].legend(fontsize=9, loc="lower right")
    for axis in axes.ravel():
        axis.axvline(0, color="#555555", linewidth=.7)
        axis.set_yticks(y)
        axis.set_ylim(len(entries)-.5, -.5)
        axis.grid(axis="x", alpha=.15)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[:, 0]:
        axis.set_yticklabels(labels, fontsize=9)
    for axis in axes[:, 1:].ravel():
        axis.tick_params(labelleft=False)
    first, first_scheme = entries[0]
    fixed = weather_stats(first, first_scheme, "fixed_cost")["mean"]
    utilisation = weather_stats(first, first_scheme, "utilisation")["mean"]
    fig.suptitle("Sequential posted offers — every prespecified scheme; user cohorts kept separate", fontsize=15)
    footer = ("Intervals: 95% t intervals across public-weather means after averaging private draws; "
              "one algorithm seed, not repeated training. Cash/procurement co-movement is imposed by settlement.\n"
              f"Fixed fleet cost: {fixed/1000:,.1f} thousand yuan/day; utilisation: {utilisation:.1%}. "
              "A positive profit increment need not imply positive absolute profit.")
    fig.supxlabel(footer, fontsize=9)
    fig.savefig(output / "comparison.png", dpi=dpi)
    plt.close(fig)


def training_sources(experiment):
    sources = []
    for directory, label, color in (("full-history", "Full explicit history", "#087f8c"),
                                    ("no-history", "No explicit history", "#a65d4b")):
        path = experiment / directory
        if not path.exists():
            continue
        sources.append(dict(path=path, label=label, color=color,
            metadata=read_json(path / "summary.json"),
            validation=pd.read_csv(path / "validation.csv") if (path / "validation.csv").exists() else pd.DataFrame(),
            optimization=pd.read_csv(path / "platform_training.csv") if (path / "platform_training.csv").exists() else pd.DataFrame(),
            joint=pd.read_csv(path / "joint_validation.csv") if (path / "joint_validation.csv").exists() else pd.DataFrame()))
    return sources


def validation_controls(experiment):
    """Optional frozen controls evaluated on the PPO validation cases.

    This deliberately reads no training-selection means: their case pool is
    different from the validation curves. The evaluator may write either its
    original averaged account names or explicit mean_actual_* aliases.
    """
    path = experiment / "baselines-initial" / "validation_controls_summary.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "scheme" not in frame:
        raise ValueError(f"Validation-control summary lacks scheme: {path}")
    if "split" in frame and not frame.split.eq("validation").all():
        raise ValueError(f"Expected validation controls, not another data split: {path}")
    if "cohort" in frame and not frame.cohort.eq("initial_users").all():
        raise ValueError(f"Expected initial-user controls for fixed-user validation: {path}")
    selected = frame.loc[frame.scheme.isin((MATRIX, "learned_response_online_map", "learned_response_prior"))].copy()
    if selected.duplicated("scheme").any():
        raise ValueError(f"Repeated validation-control scheme: {path}")
    for metric, names in (("M", ("mean_actual_M_yuan", *ALIASES["M"])),
                          ("S", ("mean_actual_S_yuan", *ALIASES["S"]))):
        name = next((column for column in names if column in frame), None)
        if name is None:
            raise ValueError(f"Validation-control summary lacks averaged {metric}: {path}")
        selected[metric] = pd.to_numeric(selected[name], errors="raise")
    return dict(path=path, frame=selected)


def plot_training(sources, output, dpi, controls=None):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    specifications = ((axes[0, 0], "procurement_net_saving_yuan", "Validation M", 1000.),
                      (axes[0, 1], "social_saving_vs_outside_yuan", "Validation true S", 1000.),
                      (axes[2, 0], "accepted_offers", "Validation accepted tasks", 1.))
    for source in sources:
        validation, optimization, joint = source["validation"], source["optimization"], source["joint"]
        style = dict(label=source["label"], color=source["color"])
        for axis, column, _, divisor in specifications:
            if not validation.empty and column in validation:
                axis.plot(validation.steps, validation[column]/divisor, marker="o", markersize=3, **style)
                selected = source["metadata"].get("selected_step")
                chosen = validation.loc[validation.steps == selected]
                if not chosen.empty:
                    axis.scatter(chosen.steps, chosen[column]/divisor, marker="*", s=100, color=source["color"], zorder=5)
        if not optimization.empty:
            for axis, column, factor in ((axes[1, 0], "train/entropy_loss", -1.),
                                        (axes[1, 1], "train/approx_kl", 1.)):
                if column in optimization:
                    axis.plot(optimization.steps, optimization[column]*factor, marker="o", markersize=3, **style)
        if not joint.empty and "procurement_net_saving_yuan" in joint:
            axes[2, 1].plot(joint.cycle, joint.procurement_net_saving_yuan/1000,
                            marker="o", **style)
    if controls is not None:
        # All three are fixed before these validation evaluations. Constant
        # lines are controls on the same cases, not observations during PPO.
        control_styles = ((MATRIX, "Fixed matrix (validation)", "#424242", (0, (7, 3))),
            ("learned_response_online_map", "Logit + MAP (validation)", "#6e6e6e", (0, (4, 2))),
            ("learned_response_prior", "Logit prior (validation)", "#999999", (0, (2, 2))))
        frame = controls["frame"].set_index("scheme")
        for scheme, label, color, dashed in control_styles:
            if scheme not in frame.index:
                continue
            for axis, metric in ((axes[0, 0], "M"), (axes[0, 1], "S")):
                value = frame.loc[scheme, metric]
                if finite_number(value):
                    axis.axhline(value/1000, label=label, color=color,
                                 linestyle=dashed, linewidth=1.3, zorder=1)
    for axis, _, title, divisor in specifications:
        axis.set(title=title, xlabel="Platform offers sampled", ylabel="Thousand yuan / day" if divisor == 1000 else "Tasks / day")
    axes[1, 0].set(title="Policy entropy (last PPO update)", xlabel="Platform offers sampled", ylabel="Nats")
    axes[1, 1].set(title="Approximate KL (last PPO update)", xlabel="Platform offers sampled", ylabel="Approx KL")
    axes[2, 1].set(title="Joint-policy validation M (users may change)", xlabel="Alternating-training cycle", ylabel="Thousand yuan / day")
    for axis in axes.ravel():
        axis.grid(alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=9)
        else:
            axis.text(.5, .5, "No corresponding training log", ha="center", va="center", transform=axis.transAxes, color="#777777")
    fig.suptitle("Validation and PPO optimization; stars mark validation-selected checkpoints", fontsize=14)
    footer = ("One training seed. Validation curves have no cross-seed uncertainty bands. "
              "Joint curves are separate because user policies can differ.")
    if controls is not None:
        footer += "\nGrey dashed lines: frozen initial-user controls on the same validation cases; no training-set means."
    fig.supxlabel(footer, fontsize=9)
    fig.savefig(output / "training.png", dpi=dpi)
    plt.close(fig)


def report_text(cohorts, sources, pairs, price_patterns, controls=None):
    seeds = sorted({source["metadata"]["seed"] for source in sources if "seed" in source["metadata"]})
    lines = ["序贯合同协同：冻结策略汇总", "",
        f"训练算法种子：{seeds if seeds else '元数据未记录'}。公开天气与私有偏好编号不是算法训练种子。",
        "统计先在每个公开天气日内平均私有偏好draw，再对天气日等权汇总及配对。",
        "95%区间基于天气日均值的t区间，只反映给定策略下的场景差异，不代表多算法种子的训练稳定性。",
        "所有预设方案完整列示，顺序不按测试表现排序。固定参数只是在声明的候选族内按训练数据选择，不称全局最优。", "",
        "口径：G=接受改签的资源节约；R=实际补偿；M=G-R为采购净收益；S=G-ΔD=M+用户增量效用。",
        "平台增量现金ΔΠ=0.8M、合并采购账户增量=0.2M是结算公式；二者正相关不是PPO学习证据。",
        "全部增量相对当天真实已签旧合同。正ΔΠ只表示改善旧经营结果，绝对利润需另看旧/新Π。", ""]
    for cohort in cohorts:
        frame = cohort["outcomes"]
        lines.extend([f"队列：{cohort['name']}；输入：{cohort['directory']}",
                      f"评价检查点：{cohort['metadata'].get('checkpoint', '元数据未记录')}。"])
        matrix = cohort["selection"].get("selected", {}).get("matrix", {})
        if matrix:
            lines.append("训练选定固定矩阵参数：" + json.dumps(matrix.get("parameters", {}), ensure_ascii=False))
        lines.append("方案ID\tM元/日\t真实S元/日\tΔΠ元/日\tG元/日\tR元/日\t用户增量元/日\t采购账户增量元/日\t接受任务/日")
        for scheme in cohort["schemes"]:
            values = [weather_stats(cohort, scheme, metric)["mean"]
                      for metric in ("M", "S", "profit", "G", "R", "user_gain", "buyer_gain", "accepted")]
            lines.append(scheme + "\t" + "\t".join(fmt(value) for value in values))
        lines.extend(["", "经营尺度（包含同一套固定容量成本）：",
            "方案ID\t旧绝对Π元/日\t新绝对Π元/日\t固定成本元/日\t未扣固定成本经营结余元/日\tGPUh/日\t容量利用率\t实际账单元/GPUh\t账面覆盖单价元/GPUh"])
        for scheme in cohort["schemes"]:
            values = {metric: weather_stats(cohort, scheme, metric)["mean"] for metric in
                      ("old_profit", "new_profit", "fixed_cost", "pre_fixed_surplus", "work", "utilisation", "average_bill_price", "ledger_cover_price")}
            lines.append(scheme + "\t" + "\t".join((fmt(values["old_profit"]), fmt(values["new_profit"]),
                fmt(values["fixed_cost"]), fmt(values["pre_fixed_surplus"]), fmt(values["work"]),
                f"{values['utilisation']:.2%}" if finite_number(values["utilisation"]) else "未记录",
                fmt(values["average_bill_price"], 4), fmt(values["ledger_cover_price"], 4))))
        first = cohort["schemes"][0]
        cost = weather_stats(cohort, first, "fixed_cost")["mean"]
        old_profit = weather_stats(cohort, first, "old_profit")["mean"]
        negative_new = sum(weather_stats(cohort, scheme, "new_profit")["mean"] < 0
                           for scheme in cohort["schemes"])
        hours = cohort["capacity_gpu_hours"]
        price = cohort["config"].get("base_price_yuan_per_cu")
        lines.append(f"旧合同平均绝对Π={fmt(old_profit)}元/日；{negative_new}/{len(cohort['schemes'])}个方案的平均新绝对Π仍低于零。")
        lines.append(f"安装容量GPU小时/日：{fmt(hours)}；配置原单价：{fmt(price)}元/GPUh。")
        if finite_number(hours) and finite_number(price) and float(price) > 0:
            lines.append(f"仅按配置原单价覆盖固定成本的利用率下限约为{cost/(hours*float(price)):.2%}，尚未计电费、可变运行与采购转账。")
        lines.extend(["账面覆盖单价=(当前服务收入−当前绝对利润)/当前GPUh，只固定当前执行与结算作静态说明，不是重新定价后的需求/市场均衡。",
            "固定成本对同日同服务策略相同，短期增量比较不受其影响；长期经营可行性尚未由这一固定任务量实验得到证明。", "",
            "相对训练选定固定矩阵+贪心路由的天气配对差值（本方案−固定矩阵）：",
            "方案ID\t天气数\tΔM均值[95%区间]元/日\tΔS均值[95%区间]元/日\tΔΠ均值元/日"])
        local = pairs[(pairs.cohort == cohort["name"]) & (pairs.reference_scheme == MATRIX)]
        for scheme in cohort["schemes"]:
            selected = local.loc[local.scheme == scheme]
            if selected.empty:
                continue
            m, s, profit = (interval(selected[f"delta_{metric}"]) for metric in ("M", "S", "profit"))
            lines.append(f"{scheme}\t{m['n']}\t{fmt(m['mean'])}[{fmt(m['low'])}, {fmt(m['high'])}]\t"
                         f"{fmt(s['mean'])}[{fmt(s['low'])}, {fmt(s['high'])}]\t{fmt(profit['mean'])}")
        lines.extend(["", "PPO类策略与其他预设强规则的配对结果：", "策略\t规则\tΔM元/日\tΔS元/日\tΔΠ元/日"])
        other = pairs[(pairs.cohort == cohort["name"]) & (pairs.contrast == "policy_vs_prespecified_rule")]
        for (scheme, reference), group in other.groupby(["scheme", "reference_scheme"], sort=False):
            lines.append(f"{scheme}\t{reference}\t{fmt(group.delta_M.mean())}\t{fmt(group.delta_S.mean())}\t{fmt(group.delta_profit.mean())}")
        lines.extend(["", "用户即时效用描述（γ=1下拒绝正即时效用不自动等于决策错误）：",
                      "方案ID\t正即时效用报价数\t其中被拒数\t被拒比例\t已发合同动作种类/日（含路由，仅描述）"])
        for scheme in cohort["schemes"]:
            positive, rejected = private_counts(cohort, scheme)
            fraction = f"{rejected/positive:.2%}" if positive else "无正即时效用报价"
            lines.append(f"{scheme}\t{positive}\t{rejected}\t{fraction}\t{fmt(cohort['diversity'].get(scheme))}")
        lines.extend(["", "同一用户内的已发回扣分布（no_offer不计入价格）：",
            "方案ID\t用户\t已发报价数\trate档数\t主rate\t主rate频率\t各rate报价次数"])
        prices = price_patterns[(price_patterns.cohort == cohort["name"])
                                & (price_patterns.row_type == "owner_summary")]
        for scheme in cohort["schemes"]:
            for _, row in prices.loc[prices.scheme == scheme].sort_values("user_id").iterrows():
                if row.history_status != "observed":
                    distribution, modal, frequency = "报价日志未提供", "未记录", "未记录"
                elif row.n_offers == 0:
                    distribution, modal, frequency = "未发报价", "—", "—"
                else:
                    counts = json.loads(row.rate_counts_json)
                    distribution = "; ".join(f"{float(rate):.2%}:{count}" for rate, count in counts.items())
                    modal, frequency = f"{row.modal_rate:.2%}", f"{row.modal_frequency:.2%}"
                lines.append(f"{scheme}\t{int(row.user_id)}\t{fmt(row.n_offers, 0)}\t"
                             f"{int(row.n_rate_levels)}\t{modal}\t{frequency}\t{distribution}")
        lines.extend([
            "同一user内有多档rate只能排除‘每个用户始终一个coupon’，还可能来自固定地区/时段映射或任务槽位差异。",
            "offer_price_patterns.csv另列同user、同job槽位、同公开天气，以及目标region×执行6h块/owner×region×块的rate档数、平均rate与样本量。",
            "跨块合同按实际执行时段的重叠小时计算加权均价；n_touching_contracts为触及该块的报价合同数，不是独立天气样本，也不是GPU工作量权重。",
            "rate是相对旧账单的回扣比例，同时给出实际补偿/GPUh。action包含路由与时间，动作多样性不等于价格随状态调整；价变本身也不证明收益改善。"])
        lines.extend(["", "已实现约束（逐私有draw记录，计数仅说明协议执行）：",
                      "方案ID\t天气日数\t每天气私有draw范围\t工作量一致\t截止期满足\t共同cap满足\t旧利润底线满足\t用户IR满足"])
        for scheme in cohort["schemes"]:
            group = frame.loc[frame.scheme == scheme]
            draws = cohort["weather"].xs(scheme, level="scheme")["private_draws"]
            n = len(group)
            work = np.isclose(group.work_cu, group.required_work_cu, rtol=1e-9, atol=1e-6).sum()
            deadlines = (group.deadline_compliance >= 1-1e-9).sum()
            cap = (group.carbon_emissions_t <= group.carbon_budget_t + 1e-6).sum()
            floor = (group.profit >= -1e-6).sum()
            ir = (group.user_ir_rate >= 1-1e-9).sum()
            lines.append(f"{scheme}\t{len(draws)}\t{int(draws.min())}–{int(draws.max())}\t"
                         f"{work}/{n}\t{deadlines}/{n}\t{cap}/{n}\t{floor}/{n}\t{ir}/{n}")
        cap_spread = frame.groupby("case_id").carbon_budget_t.agg(lambda x: x.max()-x.min()).max()
        work_spread = frame.groupby("case_id").required_work_cu.agg(lambda x: x.max()-x.min()).max()
        lines.append(f"相同case跨方案的碳上限最大差={fmt(cap_spread, 8)}吨，要求GPUh最大差={fmt(work_spread, 8)}。")
        lines.append("上述保障来自合同与预成交条件，不作为PPO超过强规则的证据。\n")
    lines.extend(["训练与选择记录："])
    for source in sources:
        metadata = source["metadata"]
        records = metadata.get("training_records", [])
        joint = metadata.get("joint_records", [])
        actual = sum(row.get("actual_steps", 0) for row in records)
        updates = sum(row.get("optimizer_steps", 0) for row in records)
        added = {}
        for row in joint:
            actor = str(row["actor"])
            added[actor] = added.get(actor, 0) + row.get("actual_actor_steps", 0)
        lines.append(f"{source['label']}: 算法seed={metadata.get('seed', '未记录')}，实际平台报价步数={actual}，"
                     f"优化器更新={updates}，验证选定步数={metadata.get('selected_step', '未记录')}，"
                     f"选定联合轮={metadata.get('selected_joint_cycle', '未记录')}，联合增加各actor步数={added}。")
    if controls is not None:
        lines.append("训练图灰虚线来自同验证集、同初始用户的既定固定矩阵与logit控制评估：" + str(controls["path"]))
        lines.append("这些验证均值仅与固定用户阶段的验证M/S曲线对照；不使用训练选参均值，也不与用户可变化的联合阶段混画。")
    lines.extend(["", "解释：",
        "固定raw仅固定单合同；固定matrix的路由随当前公共M变化；valueShare的回扣随G变化；logit/MAP使用实际报价反馈估计接受概率。",
        "no-history删除显式报价反馈，当前物理状态、现金与已签预约仍可能携带历史影响。同权重清空history是输入敏感性，单独训练no-history是信息集对照。",
        "初始用户和联合用户队列分别比较；更换用户产生的收益不全部归因于平台定价。全部强规则各自列示，不按测试赢家选择有利叙述。",
        "报价多样性仅作描述，不是训练奖励或有效性的判据。完整机制与账务口径见仓库docs/posted-offer-mechanism-notes.txt。"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Default: experiment directory")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    experiment = args.experiment.resolve()
    output = args.output.resolve() if args.output is not None else experiment
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlepad": 12, "savefig.facecolor": "white"})
    cohorts = [load_cohort(experiment / "baselines-initial", "Initial users")]
    joint = experiment / "baselines-joint"
    joint_ready = all((joint / name).exists() for name in
                      ("test_summary.csv", "test_outcomes.csv", "test_private_response_summary.csv"))
    if joint_ready:
        cohorts.append(load_cohort(joint, "Joint users"))
    sources = training_sources(experiment)
    controls = validation_controls(experiment)
    pairs = paired_weather_rows(cohorts)
    patterns = offer_price_patterns(cohorts)
    pairs.to_csv(output / "paired_weather_comparison.csv", index=False, encoding="utf-8-sig")
    patterns.to_csv(output / "offer_price_patterns.csv", index=False, encoding="utf-8-sig")
    text = report_text(cohorts, sources, pairs, patterns, controls)
    if not joint_ready:
        text += "\n联合用户队列未提供完整的同格式评价文件，本次未混入该队列；提供后可重新汇总。\n"
    (output / "报告.txt").write_text(text, encoding="utf-8")
    plot_comparison(cohorts, output, args.dpi)
    plot_training(sources, output, args.dpi, controls)
    print(json.dumps(dict(output=str(output), cohorts=[item["name"] for item in cohorts],
        files=["报告.txt", "comparison.png", "training.png", "paired_weather_comparison.csv",
               "offer_price_patterns.csv"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
