#!/usr/bin/env python3
"""Summarise frozen carbon/cash results; never train, select or simulate.

Private draws are averaged within each public weather day before computing
scenario means and paired t intervals. The one algorithm seed is not treated
as repeated independent training. Run only after the experiment's final test
has completed; importing this module does not open any experiment files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import isqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "experiments" / "carbon-aware" / "seed0"
COHORTS = ("fixed_users", "carbon_joint_users", "cash_joint_users")
SCHEMES = ("original_ppo", "cash_continued", "carbon_ppo", "carbon_rule",
           "cash_response_rule", "carbon_joint_ppo", "cash_joint_ppo")
LABELS = {
    "original_ppo": "Reference PPO",
    "cash_continued": "Cash PPO",
    "carbon_ppo": "Carbon PPO",
    "carbon_rule": "Carbon response rule",
    "cash_response_rule": "Cash response rule",
    "carbon_joint_ppo": "Carbon PPO + joint",
    "cash_joint_ppo": "Cash PPO + joint",
}
SELECTION_COHORT = {
    **{scheme: "fixed_users" for scheme in SCHEMES},
    "carbon_joint_ppo": "carbon_joint_users", "cash_joint_ppo": "cash_joint_users",
}
METRICS = {
    "mean_profit_gain_yuan": "platform_gain_vs_outside_yuan",
    "mean_absolute_profit_yuan": "platform_profit_yuan",
    "mean_outside_profit_yuan": "outside_platform_profit_yuan",
    "mean_grid_emissions_t": "carbon_emissions_t",
    "mean_outside_grid_emissions_t": "outside_carbon_emissions_t",
    "mean_grid_reduction_t": "grid_emissions_reduction_t",
    "mean_dc_footprint_t": "dc_consumption_emissions_t",
    "mean_outside_dc_footprint_t": "outside_dc_consumption_emissions_t",
    "mean_dc_footprint_reduction_t": "dc_footprint_reduction_t",
    "mean_social_saving_yuan": "social_saving_vs_outside_yuan",
    "mean_user_gain_yuan": "user_gain_yuan",
    "mean_user0_gain_yuan": "user_0_gain_yuan",
    "mean_user1_gain_yuan": "user_1_gain_yuan",
    "mean_user2_gain_yuan": "user_2_gain_yuan",
    "mean_buyer_gain_yuan": "buyer_post_settlement_benefit_yuan",
    "mean_M_yuan": "procurement_net_saving_yuan",
    "mean_G_yuan": "flexibility_resource_saving_yuan",
    "mean_R_yuan": "additional_user_rebate_yuan",
    "mean_tasks_resolved": "posted_jobs_resolved",
    "mean_offers": "posted_offers",
    "mean_accepted": "accepted_offers",
    "mean_rejected": "rejected_offers",
    "mean_no_offer": "no_offer_jobs",
    "mean_work_gpuh": "work_gpu_hours",
    "mean_required_work_gpuh": "required_work_cu",
    "mean_completion_rate": "completion_rate",
    "mean_deadline_compliance": "deadline_compliance",
    "mean_same_cap_compliant": "same_cap_compliant",
    "mean_old_profit_floor_satisfied": "old_profit_floor_satisfied",
    "mean_user_ir_rate": "user_ir_rate",
    "mean_carbon_budget_t": "carbon_budget_t",
    "mean_policy_carbon_price_yuan_per_t": "policy_carbon_price_yuan_per_t",
    "mean_cap_shadow_price_yuan_per_t": "cap_shadow_price_yuan_per_t",
}
REQUIRED_METRICS = (
    "platform_gain_vs_outside_yuan", "carbon_emissions_t", "outside_carbon_emissions_t",
    "grid_emissions_reduction_t", "dc_consumption_emissions_t",
    "outside_dc_consumption_emissions_t", "dc_footprint_reduction_t",
    "social_saving_vs_outside_yuan", "user_gain_yuan",
)
PAIRED_METRICS = {
    "profit_gain_yuan": ("platform_gain_vs_outside_yuan", "CNY/day"),
    "grid_reduction_t": ("grid_emissions_reduction_t", "tCO2/day"),
    "grid_emissions_t": ("carbon_emissions_t", "tCO2/day"),
    "dc_footprint_reduction_t": ("dc_footprint_reduction_t", "tCO2/day"),
    "social_saving_yuan": ("social_saving_vs_outside_yuan", "CNY/day"),
    "user_gain_yuan": ("user_gain_yuan", "CNY/day"),
}


def read_json(path, *, optional=False):
    if optional and not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _decode_case(case):
    paired = -int(case) - 1
    if paired < 0:
        raise ValueError("Expected a paired public/private episode ID")
    diagonal = (isqrt(8 * paired + 1) - 1) // 2
    private = paired - diagonal * (diagonal + 1) // 2
    return diagonal - private, private


def weather_means(days, *, expected_weather_days=32, expected_draws=4):
    """Validate paired cells and average private draws, without fitting anything."""
    frame = days.copy()
    if "public_seed" not in frame or "preference_seed" not in frame:
        if "case_id" not in frame:
            raise ValueError("Provide public/preference seeds or exact paired case IDs")
        decoded = [_decode_case(case) for case in frame["case_id"]]
        frame["public_seed"], frame["preference_seed"] = zip(*decoded)
    keys = ["cohort", "scheme", "public_seed", "preference_seed"]
    missing = [column for column in (*keys, *REQUIRED_METRICS) if column not in frame]
    if missing:
        raise ValueError(f"Missing result columns: {missing}")
    if frame.duplicated(keys).any():
        raise ValueError("A private case appears more than once for one cohort/method")
    cells = frame.groupby(["cohort", "scheme", "public_seed"], sort=False).size()
    if not (cells == expected_draws).all():
        raise ValueError("Every method/weather cell must have the declared private draws")
    counts = cells.groupby(level=[0, 1]).size()
    if not (counts == expected_weather_days).all():
        raise ValueError("Every cohort/method must contain the declared weather days")
    reference_cases, reference_methods = None, None
    for _, group in frame.groupby(["cohort", "scheme"], sort=False):
        cases = frozenset(zip(group.public_seed, group.preference_seed))
        if reference_cases is None:
            reference_cases = cases
        elif cases != reference_cases:
            raise ValueError("All comparisons must use identical public/private cases")
    for _, group in frame.groupby("cohort", sort=False):
        methods = frozenset(group.scheme)
        if reference_methods is None:
            reference_methods = methods
        elif methods != reference_methods:
            raise ValueError("Each user cohort must include the same frozen methods")
    columns = list(dict.fromkeys(source for source in METRICS.values() if source in frame))
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[list(REQUIRED_METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("Outcome quantities used in paired statistics must be finite")
    return frame.groupby(["cohort", "scheme", "public_seed"], sort=False)[columns].mean().reset_index()


def mean_interval(values):
    """A t interval across weather means, with n-1 degrees of freedom."""
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("A weather interval needs at least two finite weather means")
    average = float(array.mean())
    half = float(t.ppf(.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array)))
    return average, average - half, average + half


def build_comparison(weather, summary, reference, *, private_draws=4):
    required = float(reference["required_profit_yuan"])
    rows = []
    for (cohort, scheme), group in weather.groupby(["cohort", "scheme"], sort=False):
        row = dict(cohort=cohort, scheme=scheme, label=LABELS.get(scheme, scheme),
                   weather_days=len(group), private_draws_per_weather=private_draws,
                   algorithm_seeds=1, algorithm_root_seed=0,
                   required_profit_yuan=required,
                   train_reference_profit_yuan=float(reference["reference_profit_yuan"]),
                   retained_profit_target=float(reference["q"]),
                   checkpoint_selection_user_cohort=SELECTION_COHORT.get(scheme, "unknown"))
        for output, source in METRICS.items():
            if source in group:
                row[output] = float(group[source].mean())
        for prefix, source in (("profit", "platform_gain_vs_outside_yuan"),
                               ("grid_reduction", "grid_emissions_reduction_t")):
            _, low, high = mean_interval(group[source])
            row[f"{prefix}_weather_ci95_low"] = low
            row[f"{prefix}_weather_ci95_high"] = high
        row["test_profit_qualified"] = row["mean_profit_gain_yuan"] >= required
        row["test_profit_shortfall_yuan"] = max(0., required - row["mean_profit_gain_yuan"])
        row["grid_reduction_percent"] = (100. * row["mean_grid_reduction_t"]
                                          / row["mean_outside_grid_emissions_t"])
        row["dc_footprint_reduction_percent"] = (100. * row["mean_dc_footprint_reduction_t"]
                                                 / row["mean_outside_dc_footprint_t"])
        original = weather[(weather.cohort == cohort) & (weather.scheme == "original_ppo")]
        if original.empty:
            raise ValueError("Each cohort needs its frozen original PPO reference")
        original_cash = float(original.platform_gain_vs_outside_yuan.mean())
        row["same_cohort_original_profit_yuan"] = original_cash
        row["retained_original_profit_fraction"] = (row["mean_profit_gain_yuan"] / original_cash
                                                     if original_cash != 0 else np.nan)
        row["same_test_95pct_profit_qualified"] = (row["retained_original_profit_fraction"]
                                                   >= float(reference["q"]))
        row["same_test_profit_gap_yuan"] = (row["mean_profit_gain_yuan"]
                                            - float(reference["q"]) * original_cash)
        recorded = summary[(summary.cohort == cohort) & (summary.scheme == scheme)]
        if len(recorded) != 1:
            raise ValueError("test_summary must have exactly one matching cohort/method row")
        recorded = recorded.iloc[0]
        for source in REQUIRED_METRICS:
            if source in recorded and not np.isclose(float(recorded[source]), float(group[source].mean()),
                                                      rtol=1e-10, atol=1e-6):
                raise ValueError(f"test_summary and complete day results differ: {cohort}/{scheme}/{source}")
        for name in ("selection_split", "rule_calibration_profit_qualified",
                     "train_profit_qualified", "validation_profit_qualified",
                     "same_test_95pct_profit_qualified", "same_test_profit_gap_yuan"):
            if name in recorded:
                row[name] = recorded[name]
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != len(summary):
        raise ValueError("Summary and day files must contain the same cohort/method rows")
    result["_cohort"] = result.cohort.map({name: index for index, name in enumerate(COHORTS)}).fillna(99)
    result["_scheme"] = result.scheme.map({name: index for index, name in enumerate(SCHEMES)}).fillna(99)
    return result.sort_values(["_cohort", "_scheme"]).drop(columns=["_cohort", "_scheme"]).reset_index(drop=True)


def paired_comparisons(weather, *, private_draws=4):
    """Candidate minus comparator, paired by weather, with cohort labels intact."""
    pairs = []
    for cohort in dict.fromkeys(weather.cohort):
        present = set(weather[weather.cohort == cohort].scheme)
        for comparator in ("original_ppo", "cash_continued", "carbon_rule"):
            if {"carbon_ppo", comparator} <= present:
                pairs.append((cohort, "carbon_ppo", cohort, comparator, "same_user_cohort"))
        if {"carbon_joint_ppo", "cash_joint_ppo"} <= present:
            pairs.append((cohort, "carbon_joint_ppo", cohort, "cash_joint_ppo", "same_user_cohort_joint_platforms"))
    # A secondary system comparison: each jointly trained platform with its
    # own trained users. It is explicitly distinct from holding users fixed.
    own_pairs = (("carbon_joint_users", "carbon_joint_ppo"),
                 ("cash_joint_users", "cash_joint_ppo"))
    if all(((weather.cohort == cohort) & (weather.scheme == scheme)).any()
           for cohort, scheme in own_pairs):
        pairs.append((*own_pairs[0], *own_pairs[1], "each_joint_system_with_own_users"))
    rows = []
    for candidate_cohort, candidate, reference_cohort, comparator, kind in pairs:
        left = weather[(weather.cohort == candidate_cohort) & (weather.scheme == candidate)].set_index("public_seed")
        right = weather[(weather.cohort == reference_cohort) & (weather.scheme == comparator)].set_index("public_seed")
        if set(left.index) != set(right.index):
            raise ValueError("Paired methods must have identical public weather days")
        right = right.reindex(left.index)
        for metric, (source, unit) in PAIRED_METRICS.items():
            average, low, high = mean_interval(left[source].to_numpy() - right[source].to_numpy())
            rows.append(dict(candidate_cohort=candidate_cohort, candidate=candidate,
                reference_cohort=reference_cohort, reference=comparator, comparison_type=kind,
                metric=metric, unit=unit, difference_direction="candidate minus reference",
                mean_difference=average, ci95_low=low, ci95_high=high,
                weather_days=len(left), private_draws_per_weather=private_draws,
                algorithm_seeds=1, interval_scope="weather variation conditional on one frozen trained seed"))
    return pd.DataFrame(rows)


def _phase_metadata(directory):
    result = {}
    for kind in ("cash", "carbon"):
        phase = directory / kind
        result[kind] = dict(fixed_training=read_json(phase / "training_summary.json", optional=True),
                            fixed_selection=read_json(phase / "selection.json", optional=True),
                            joint_training=read_json(phase / "joint" / "summary.json", optional=True),
                            joint_selection=read_json(phase / "joint" / "selection.json", optional=True))
    return result


def make_figure(comparison, input_dir, reference, output_dir):
    """Publication figure: held-out outcomes, then two fixed-user val traces."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"original_ppo": "#444444", "cash_continued": "#3B6FB6", "carbon_ppo": "#D46036",
              "carbon_rule": "#3D9069", "cash_response_rule": "#8A73A6",
              "carbon_joint_ppo": "#D46036", "cash_joint_ppo": "#3B6FB6"}
    markers = {"original_ppo": "s", "cash_continued": "o", "carbon_ppo": "o", "carbon_rule": "D",
               "cash_response_rule": "v", "carbon_joint_ppo": "^", "cash_joint_ppo": "^"}
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
             "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
             "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"}
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), gridspec_kw={"width_ratios": [1.12, 1., 1.]})
        fixed = comparison[comparison.cohort == "fixed_users"]
        if fixed.empty:
            raise ValueError("The primary figure requires the frozen reference-user cohort")
        for row in fixed.itertuples():
            joint = row.scheme.endswith("joint_ppo")
            axes[0].scatter(row.mean_profit_gain_yuan / 1000., row.mean_grid_reduction_t,
                            color=colors.get(row.scheme, "#555555"), marker=markers.get(row.scheme, "o"),
                            s=64 if joint else 50, facecolors="none" if joint else None,
                            linewidths=1.5, zorder=3)
        threshold = float(reference["required_profit_yuan"]) / 1000.
        axes[0].axvline(threshold, color="#777777", linestyle="--", linewidth=1.1)
        axes[0].axhline(0., color="#999999", linewidth=.7)
        axes[0].set(xlabel="Incremental cash ($10^3$ CNY/day)",
                    ylabel="Whole-grid CO$_2$ avoided (t/day)", title="(a) Frozen reference users: test")
        for kind, scheme in (("cash", "cash_continued"), ("carbon", "carbon_ppo")):
            path = input_dir / kind / "validation.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing fixed-user validation trajectory: {path}")
            values = pd.read_csv(path, encoding="utf-8-sig").sort_values("additional_steps")
            selection = read_json(input_dir / kind / "selection.json")
            x = values.additional_steps / 1000.
            for axis, column, scale in ((axes[1], "platform_gain_vs_outside_yuan", 1000.),
                                        (axes[2], "grid_emissions_reduction_t", 1.)):
                axis.plot(x, values[column] / scale, color=colors[scheme], linewidth=1.8,
                          marker="o", markersize=3.5, label=LABELS[scheme])
                chosen = values[values.additional_steps == selection["additional_steps"]]
                if len(chosen) == 1:
                    axis.scatter(chosen.additional_steps / 1000., chosen[column] / scale,
                                 color=colors[scheme], marker="s", s=46, zorder=4,
                                 edgecolors="white", linewidths=.7)
        axes[1].axhline(threshold, color="#777777", linestyle="--", linewidth=1.1)
        axes[1].set(xlabel="Additional platform steps ($10^3$)", ylabel="Incremental cash ($10^3$ CNY/day)",
                    title="(b) Fixed-user validation: cash")
        axes[2].axhline(0., color="#999999", linewidth=.7)
        axes[2].set(xlabel="Additional platform steps ($10^3$)", ylabel="Whole-grid CO$_2$ avoided (t/day)",
                    title="(c) Fixed-user validation: carbon")
        for axis in axes:
            axis.grid(axis="y", color="#e6e6e6", linewidth=.65, zorder=0)
            axis.tick_params(labelsize=9)
        handles = [Line2D([], [], linestyle="none", marker=markers.get(scheme, "o"),
                          markerfacecolor="none" if scheme.endswith("joint_ppo") else colors.get(scheme, "#555555"),
                          markeredgecolor=colors.get(scheme, "#555555"), markersize=6,
                          label=LABELS.get(scheme, scheme)) for scheme in SCHEMES if scheme in set(fixed.scheme)]
        handles.append(Line2D([], [], color="#777777", linestyle="--", label="Fixed TRAIN profit target"))
        figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                      bbox_to_anchor=(.5, .02), fontsize=9, columnspacing=1.8)
        figure.subplots_adjust(left=.065, right=.985, top=.88, bottom=.30, wspace=.34)
        figure.suptitle("Carbon and cash under unchanged physical and settlement rules", fontsize=12, y=.98)
        figure.text(.065, .18, "Test: 32 weather days x 4 private draws. Validation squares mark selected checkpoints. "
                    "Joint-trained policies in panel (a) use the same frozen reference users.", fontsize=8.2)
        figure.savefig(output_dir / "comparison.png", dpi=300)
        figure.savefig(output_dir / "comparison.pdf")
        plt.close(figure)


def write_text(comparison, paired, metadata, output):
    reference, plan = metadata["reference"], metadata["plan"]
    lines = ["碳感知合同：冻结策略比较", "",
             f"算法根种子：{plan.get('seed', 0)}；只含一个训练种子。",
             "先在每个天气日内平均4个私有意愿，再对32个天气日等权统计和配对。",
             "95% t区间反映冻结策略面对天气场景的差异，不代表多种子训练稳定性。",
             f"固定利润门槛：{reference['required_profit_yuan']:.6f} 元/日"
             f"（ALL TRAIN原PPO增量利润 {reference['reference_profit_yuan']:.6f} 的 {100*reference['q']:.2f}%）。",
             "固定TRAIN门槛和同组TEST原PPO的95%门槛分别列示；前者在TEST保持常量，后者仅描述相同测试条件下利润保留比例。",
             "规则128案例校准资格、ALL TRAIN确认、验证资格和TEST资格分别列示。",
             "全部增量相对真实已接受旧合同；全网减排和DC分摊足迹是不同指标，负减排表示增加。",
             "政策碳价与cap影子价分别列示；社会节约S和用户收益为辅助效果指标。",
             "", "用户组\t方案\t增量Π元/日\t同组原PPO利润比\t全网排放t/日\t全网减排t/日\tDC足迹减排t/日\t社会S元/日\t用户增量元/日\t固定TRAIN门槛达标\t同组TEST95%达标"]
    for row in comparison.itertuples():
        lines.append(f"{row.cohort}\t{row.scheme}\t{row.mean_profit_gain_yuan:.2f}\t"
                     f"{row.retained_original_profit_fraction:.4f}\t{row.mean_grid_emissions_t:.6f}\t"
                     f"{row.mean_grid_reduction_t:.6f}\t{row.mean_dc_footprint_reduction_t:.6f}\t"
                     f"{row.mean_social_saving_yuan:.2f}\t{row.mean_user_gain_yuan:.2f}\t{row.test_profit_qualified}\t"
                     f"{row.same_test_95pct_profit_qualified}")
    lines += ["", "配对差值为候选方案减对照方案；利润/全网减排/S为正表示对应指标更高。",
              "同用户组的比较保持三个用户策略相同。each_joint_system_with_own_users另比较各自平台+用户整体，不能归因于平台单独改变。",
              "规则权重从固定参考用户TRAIN选择；转移至联合用户组时仍使用原冻结权重。",
              "checkpoint_selection_user_cohort说明训练/验证资格对应的用户组，不把该资格当作其他用户组的验证结果。"]
    rule = metadata.get("rule_selection", {})
    if rule:
        lines.append(f"碳规则128案例校准候选族是否达到利润门槛：{rule.get('feasible')}；"
                     f"缺口 {rule.get('profit_shortfall_yuan', float('nan')):.6f} 元/日。未达到时不降低目标。")
    full_check = metadata.get("rule_all_train_check", {})
    selected_controller = rule.get("selected", {}).get("controller")
    confirmed = [row for row in full_check.get("results", ())
                 if selected_controller is not None and row.get("controller") == selected_controller]
    if len(confirmed) == 1:
        cash = float(confirmed[0]["mean_platform_gain_yuan"])
        lines.append(f"选定碳规则ALL TRAIN确认：增量Π {cash:.6f} 元/日；"
                     f"固定门槛达标 {cash >= reference['required_profit_yuan']}；"
                     f"缺口 {max(0., reference['required_profit_yuan'] - cash):.6f} 元/日。")
    lines += ["", "配对天气结果见paired_comparison.csv；完整量纲和选择/训练预算见summary_metadata.json。"]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory, output = args.input.resolve(), (args.output or args.input).resolve()
    plan = read_json(directory / "plan.json")
    reference = read_json(directory / "reference.json")
    days = pd.read_csv(directory / "test_days.csv", encoding="utf-8-sig")
    summary = pd.read_csv(directory / "test_summary.csv", encoding="utf-8-sig")
    weather = weather_means(days)
    if set(weather.cohort) != set(COHORTS):
        raise ValueError("Expected complete fixed, carbon-joint and cash-joint user cohorts")
    comparison = build_comparison(weather, summary, reference)
    paired = paired_comparisons(weather)
    metadata = dict(algorithm_seeds=1, algorithm_root_seed=0, weather_days=32,
                    private_draws_per_weather=4,
                    interval="two-sided 95% Student t over 32 paired weather means, df=31",
                    plan=plan, reference=reference,
                    rule_selection=read_json(directory / "rule_selection.json"),
                    rule_validation=read_json(directory / "rule_validation.json", optional=True),
                    rule_all_train_check=read_json(directory / "rule_all_train_check.json", optional=True),
                    phases=_phase_metadata(directory),
                    input_files_sha256={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                        for name in ("test_days.csv", "test_summary.csv", "plan.json", "reference.json")})
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(output / "paired_comparison.csv", index=False, encoding="utf-8-sig")
    weather.to_csv(output / "weather_means.csv", index=False, encoding="utf-8-sig")
    (output / "summary_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                    allow_nan=False), encoding="utf-8")
    write_text(comparison, paired, metadata, output)
    make_figure(comparison, directory, reference, output)
    print(json.dumps(dict(completed=str(output), cohort_methods=len(comparison),
                         paired_metric_rows=len(paired), weather_days=32), ensure_ascii=False))


if __name__ == "__main__":
    main()
