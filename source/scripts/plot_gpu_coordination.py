#!/usr/bin/env python3
"""Plot saved GPU-coordination results without fitting or selecting a model.

The main panel reads comparison/comparison_mean.csv written by
evaluate_gpu_coordination.py.  Fixed-user policies are the primary comparison.
The format-only smoke mode uses conspicuously labelled synthetic values and
must be directed to a scratch folder, never interpreted as research results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = (
    ("cash_only", "ppo_fixed_users", "Cash-objective PPO\n(internal carbon value = 0)"),
    ("electricity_only", "ppo_fixed_users", "Electricity-information PPO\n(internal carbon value = 100)"),
    ("electricity_carbon", "ppo_fixed_users", "Electricity + carbon information PPO\n(internal carbon value = 100)"),
    ("electricity_carbon", "constant_fixed_users", "Train-selected constant menu\n(internal carbon value = 100)"),
)
COLORS = ("#756bb1", "#3182bd", "#238b45", "#636363")


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "savefig.facecolor": "white",
    })


def selected_rows(frame):
    rows = []
    for experiment, phase, label in ORDER:
        selected = frame[(frame.experiment == experiment) & (frame.phase == phase)]
        if len(selected) != 1:
            raise ValueError(f"Need exactly one {experiment}/{phase} mean row; found {len(selected)}")
        row = selected.iloc[0].to_dict()
        row["plot_label"] = label
        rows.append(row)
    return rows


def _signed_labels(ax, values, y, *, digits=0):
    span = max(float(np.max(np.abs(values))), 1e-7)
    ax.set_xlim(-span * .25 if np.all(values >= 0) else -span * 1.40,
                span * .25 if np.all(values <= 0) else span * 1.40)
    offset = span * .045
    for position, value in zip(y, values):
        label = f"{value:+,.{digits}f}" if abs(value) >= 10 ** (-digits) / 2 else f"{0:.{digits}f}"
        ax.text(value + (offset if value >= 0 else -offset), position,
                label, va="center", ha="left" if value >= 0 else "right", fontsize=9)
    ax.axvline(0, color="#333333", linewidth=.8)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)


def plot_comparison(frame, output, *, smoke=False):
    apply_style()
    rows = selected_rows(frame)
    y = np.arange(len(rows))
    panels = (
        ("Operating-cost\nsaving", "Yuan/day", "dispatch_cost_saving_yuan", 1., 0),
        ("All-grid operating\ncost saving", "% of baseline operating cost", "pooled_dispatch_cost_saving_rate", 100., 2),
        ("Full-cost\nsaving", "% of baseline full cost", "pooled_full_cost_saving_rate", 100., 3),
        ("Cash-profit\nchange", "Yuan/day", "cash_profit_change_yuan", 1., 0),
        ("Actual grid-emissions\nreduction", "t CO2/day", "grid_emissions_reduction_t", 1., 2),
        ("Allocated DC-footprint\nreduction", "t CO2/day", "dc_footprint_reduction_t", 1., 2),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(20.8, 5.7), sharey=True)
    for ax, (title, unit, key, multiplier, digits) in zip(axes, panels):
        values = np.asarray([row[key] * multiplier for row in rows], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Cannot plot non-finite values for {key}")
        ax.barh(y, values, height=.57, color=COLORS)
        _signed_labels(ax, values, y, digits=digits)
        ax.set_title(title, fontsize=11, pad=13)
        ax.set_xlabel(unit, fontsize=9, labelpad=10)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_yticks(y, [row["plot_label"] for row in rows], fontsize=10)
    axes[0].invert_yaxis()
    days = sorted({int(row.get("days", 0)) for row in rows})
    day_text = str(days[0]) if len(days) == 1 else "/".join(map(str, days))
    title = ("SYNTHETIC FORMAT CHECK - NOT RESEARCH RESULTS" if smoke else
             f"GPU pricing: fixed identical user policies | {day_text} held-out days | one PPO training seed")
    fig.suptitle(title, fontsize=14, y=.975, weight="bold")
    fig.text(.02, .075,
             "Positive bars mean cost saving, cash gain, or emissions reduction; negative bars mean deterioration. "
             "All changes are relative to the same day's original contracts.", fontsize=10)
    fig.text(.02, .040,
             "Operating cost = all-grid fuel/resources + variable compute/network operations + user inconvenience. "
             "Full cost additionally includes common installed-capacity cost; rates use pooled denominators.", fontsize=9)
    fig.text(.02, .010,
             "Rows 2 and 3 isolate explicit carbon information at the same internal carbon value (100 yuan/t). "
             "Row 1 changes the firm's preference. DC-footprint allocation and actual grid emissions are separate measures.", fontsize=9)
    fig.subplots_adjust(left=.175, right=.985, top=.77, bottom=.22, wspace=.52)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_fixed_cost_sensitivity(frame, output):
    apply_style()
    selected = frame[(frame.experiment == "fixed_capacity_fee") & np.isclose(frame.site_scale, 1.)]
    groups = selected.groupby("fixed_capacity_fee_yuan_per_gpu_capacity_hour").mean(numeric_only=True)
    required = (2., 3., 5.)
    if not all(fee in groups.index for fee in required):
        raise ValueError("Fixed-cost sensitivity needs the 2/3/5 yuan capacity-hour cases")
    contribution = groups.loc[list(required), "platform_contribution_before_fixed_cost_yuan"].to_numpy()
    if not np.allclose(contribution, contribution[0], rtol=1e-10, atol=1e-7):
        raise ValueError("Fixed-cost scenarios changed the operating contribution")
    values = np.r_[contribution[0], groups.loc[list(required), "platform_profit_after_fixed_cost_yuan"]] / 1000
    labels = ("Contribution before\ninstalled-capacity cost", "After fixed fee = 2", "After fixed fee = 3", "After fixed fee = 5")
    fig, ax = plt.subplots(figsize=(10, 5.3))
    y = np.arange(4)
    ax.barh(y, values, height=.58, color=("#3182bd", "#74c476", "#238b45", "#d95f0e"))
    _signed_labels(ax, values, y, digits=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Thousand yuan/day; positive = cash contribution/profit, negative = loss")
    fee_break_even = float(groups["break_even_fixed_fee_yuan_per_gpu_capacity_hour"].mean())
    fig.suptitle("Existing-capacity cost sensitivity | original contracts only", fontsize=13, weight="bold")
    fig.text(.035, .080, f"Fixed fees are yuan per installed GPU-capacity hour; mean break-even fee = {fee_break_even:.3f}. "
             "Same workloads, schedule and dispatch in all cases.", fontsize=9)
    fig.text(.035, .035, "Common capacity cost cancels in paired gains. "
             "Low-utilization batch-only fleet; prices and costs are study assumptions.", fontsize=9)
    fig.subplots_adjust(left=.25, right=.96, top=.84, bottom=.24)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def smoke_frame():
    values = ((3000, .01, .0012, 1800, -1.2, 3.2),
              (2700, .009, .0011, 1600, .5, 4.0),
              (2500, .008, .0010, 1400, 2.0, 5.3),
              (1900, .006, .0008, 900, .8, 2.5))
    keys = ("dispatch_cost_saving_yuan", "pooled_dispatch_cost_saving_rate",
            "pooled_full_cost_saving_rate", "cash_profit_change_yuan",
            "grid_emissions_reduction_t", "dc_footprint_reduction_t")
    return pd.DataFrame([dict(experiment=entry[0], phase=entry[1], days=32,
                              **dict(zip(keys, row))) for entry, row in zip(ORDER, values)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path)
    parser.add_argument("--sensitivity-csv", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Synthetic format check only; use a scratch output directory")
    args = parser.parse_args()
    comparison = args.comparison_dir or args.output_dir / "comparison"
    sensitivity = args.sensitivity_csv or args.output_dir / "尺度与成本敏感性.csv"
    if args.smoke:
        output = plot_comparison(smoke_frame(), args.output_dir / "FORMAT_CHECK_ONLY.png", smoke=True)
        print(output)
        return
    means_path = comparison / "comparison_mean.csv"
    if not means_path.exists():
        raise FileNotFoundError(f"Evaluation results are not ready: {means_path}")
    print(plot_comparison(pd.read_csv(means_path), args.output_dir / "结果对比.png"))
    if sensitivity.exists():
        print(plot_fixed_cost_sensitivity(pd.read_csv(sensitivity), args.output_dir / "固定容量成本与利润.png"))


if __name__ == "__main__":
    main()
