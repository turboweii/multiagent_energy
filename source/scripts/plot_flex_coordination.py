#!/usr/bin/env python3
"""Plot saved flexibility-contract results without fitting or selecting models.

The primary frozen-user panel compares all four contract arms and the main
arm's training-selected constant menu. Signed transfers are plotted separately
from resource cost savings. The optional smoke mode creates only explicitly
labelled format-check images and belongs in a scratch directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


ARMS = (
    ("published_no_share", "Published tariff\nNo resource sharing", "#756bb1"),
    ("conditional_no_share", "Conditional rebate\nNo resource sharing", "#3182bd"),
    ("conditional_share", "Conditional rebate\nResource sharing", "#e88936"),
    ("conditional_green_share", "Conditional rebate\nResource sharing + carbon qualification", "#238b45"),
)


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "savefig.facecolor": "white",
    })


def selected_rows(frame, phase="ppo_fixed_users", include_constant=True):
    selected = []
    for arm, label, color in ARMS:
        values = frame[(frame.experiment == arm) & (frame.phase == phase)]
        if len(values) != 1:
            raise ValueError(f"Expected one mean row for {arm}/{phase}; got {len(values)}")
        row = values.iloc[0].to_dict()
        if phase == "ppo_joint_users":
            cycle = int(row.get("selected_joint_cycle", -1))
            label += ("\nSelected cycle 0 (fixed-stage bundle)" if cycle == 0 else
                      f"\nSelected joint cycle {cycle}" if cycle > 0 else "\nSelected cycle not recorded")
        selected.append(dict(row, label=label, color=color))
    if include_constant:
        values = frame[(frame.experiment == "conditional_green_share") & (frame.phase == "constant_fixed_users")]
        if len(values) != 1:
            raise ValueError("Primary plot needs the train-selected main-arm constant benchmark")
        selected.append(dict(values.iloc[0].to_dict(), label="Train-selected fixed menu\nConditional + green sharing", color="#636363"))
    return selected


def horizontal_panel(ax, rows, key, *, title, xlabel, digits=0, multiplier=1., scientific=False, label_suffix=None):
    y = np.arange(len(rows))
    values = np.asarray([row[key] * multiplier for row in rows], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite figure data in {key}")
    colors = [row["color"] for row in rows]
    ax.barh(y, values, color=colors, height=.57)
    span = max(float(np.max(np.abs(values))), 1e-12 if scientific else 1e-7)
    lower = -span * (1.35 if np.any(values < 0) else .18)
    upper = span * (1.45 if np.any(values > 0) else .18)
    ax.set_xlim(lower, upper)
    for position, (value, row) in enumerate(zip(values, rows)):
        if scientific:
            text = f"{value:.1e}"
        else:
            text = f"{value:+,.{digits}f}" if abs(value) > 10 ** (-digits) / 2 else f"{0:.{digits}f}"
        if label_suffix is not None:
            text += label_suffix(row)
        ax.text(value + np.sign(value if value != 0 else 1) * span * .04, position,
                text, va="center", ha="left" if value >= 0 else "right", fontsize=9)
    ax.axvline(0, color="#333333", linewidth=.8)
    ax.set_title(title, fontsize=11, pad=13)
    ax.set_xlabel(xlabel, fontsize=9, labelpad=10)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8)
    if scientific:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if value == 0 else f"{value:.0e}"))


def make_results(frame, output, *, phase="ppo_fixed_users", smoke=False):
    style()
    primary = phase == "ppo_fixed_users"
    rows = selected_rows(frame, phase, include_constant=primary)
    panels = (
        ("dispatch_cost_saving_yuan", "System operating\ncost saving", "Yuan/day", 0, False, 1.),
        ("cash_profit_change_yuan", "Platform cash-profit\nchange after settlement", "Yuan/day", 0, False, 1.),
        ("user_gain_yuan", "Users' utility\ngain", "Yuan/day", 0, False, 1.),
        ("buyer_post_settlement_benefit_yuan", "Resource procurement\naccount residual", "Yuan/day", 0, False, 1.),
        ("grid_emissions_reduction_t", "Actual all-grid\nemissions reduction", "t CO2/day", 2, False, 1.),
        ("pooled_dispatch_cost_saving_rate", "All-grid operating\ncost saving", "% of baseline operating cost", 2, False, 100.),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(21.8, 6.4), sharey=True)
    for ax, (key, title, unit, digits, scientific, multiplier) in zip(axes, panels):
        horizontal_panel(ax, rows, key, title=title, xlabel=unit, digits=digits,
                         scientific=scientific, multiplier=multiplier)
    axes[0].set_yticks(np.arange(len(rows)), [row["label"] for row in rows], fontsize=10)
    axes[0].invert_yaxis()
    day_counts = sorted({int(row.get("days", 0)) for row in rows})
    day_text = str(day_counts[0]) if len(day_counts) == 1 else "/".join(map(str, day_counts))
    title = ("SYNTHETIC FORMAT CHECK - NOT RESEARCH RESULTS" if smoke else
             f"Conditional compute contracts and resource sharing | {day_text} held-out days | one PPO training seed")
    fig.suptitle(title, fontsize=14, y=.975, weight="bold")
    user_note = ("Primary comparison: identical frozen learned users; constant menu chosen on training days only."
                 if primary else "Validation-selected bundle among cycles 0-4; cycle 0 retains the fixed-stage platform and unchanged initial users. Selected cycles appear in row labels.")
    fig.text(.02, .100, user_note + " All platform actors maximize cash profit; internal carbon valuation = 0.", fontsize=9.5)
    fig.text(.02, .065, "All changes use the same day's original contracts. Positive values mean savings, gains, or avoided emissions. "
             "System operating cost = generation resources + variable compute/network cost + user inconvenience; transfers are excluded.", fontsize=9)
    fig.text(.02, .030, "Procurement-account residual and user utility are separate from platform profit. "
             "Sharing reallocates realised savings; monetary transfers are excluded from system costs.", fontsize=9)
    fig.subplots_adjust(left=.185, right=.985, top=.79, bottom=.24, wspace=.55)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def make_settlement(frame, output, *, phase="ppo_fixed_users", smoke=False):
    style()
    rows = selected_rows(frame, phase, include_constant=phase == "ppo_fixed_users")
    panels = (
        ("cash_profit_before_transfer_change_yuan", "Platform cash-profit change\nbefore procurement adjustment", "Yuan/day; signed", 0),
        ("buyer_payment_to_platform_yuan", "Procurement account\npays platform", "Yuan/day; positive payment", 0),
        ("platform_payment_to_buyer_yuan", "Platform pays\nprocurement account", "Yuan/day; positive payment", 0),
        ("cash_profit_change_yuan", "Platform cash-profit change\nafter procurement adjustment", "Yuan/day; signed", 0),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(16.2, 6.3), sharey=True)
    for ax, (key, title, xlabel, digits) in zip(axes, panels):
        horizontal_panel(ax, rows, key, title=title, xlabel=xlabel, digits=digits)
    axes[0].set_yticks(np.arange(len(rows)), [row["label"] for row in rows], fontsize=10)
    axes[0].invert_yaxis()
    title = ("SYNTHETIC FORMAT CHECK - NOT RESEARCH RESULTS" if smoke else
             "Signed procurement settlement | cash flows shown separately from resource savings")
    fig.suptitle(title, fontsize=13.5, y=.975, weight="bold")
    if phase == "ppo_joint_users":
        fig.text(.025, .135, "Validation cash selects a matched policy bundle among cycles 0-4. Cycle 0 contains no selected joint update; positive cycles include joint updates.", fontsize=9)
    fig.text(.025, .095, "After-settlement profit change = before-settlement profit change + buyer payment - platform payment. "
             "Positive and negative settlement days are preserved; completed jobs and locked invoices are unchanged.", fontsize=9)
    fig.text(.025, .050, "The contract subtracts benefits already captured through electricity bills and variable operations before crediting the resource share. "
             "These transfers redistribute the realised resource saving and do not create additional system savings.", fontsize=9)
    fig.subplots_adjust(left=.245, right=.98, top=.79, bottom=.22, wspace=.58)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def smoke_frame():
    experiments = [row[0] for row in ARMS] + ["conditional_green_share"]
    phases = ["ppo_fixed_users"] * 4 + ["constant_fixed_users"]
    rows = []
    for index, (experiment, phase) in enumerate(zip(experiments, phases)):
        saving = 900. + 500 * index
        user = 250. + 40 * index
        before = 700. - 200 * index
        payment = 0. if index < 2 else 700. + 50 * index
        charge = 0. if index < 2 else 150.
        profit = before + payment - charge
        buyer = saving - user - profit
        rows.append(dict(experiment=experiment, phase=phase, days=32,
            dispatch_cost_saving_yuan=saving, cash_profit_change_yuan=profit, user_gain_yuan=user,
            pooled_dispatch_cost_saving_rate=saving / 300000.,
            buyer_post_settlement_benefit_yuan=buyer, grid_emissions_reduction_t=-2. + 2 * index,
            max_surplus_identity_absolute_error_yuan=1e-10 * (index + 1),
            cash_profit_before_transfer_change_yuan=before, buyer_payment_to_platform_yuan=payment,
            platform_payment_to_buyer_yuan=charge))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path)
    parser.add_argument("--include-joint", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Clearly labelled synthetic format check in a scratch directory")
    args = parser.parse_args(argv)
    if args.smoke:
        frame = smoke_frame()
        print(make_results(frame, args.output_dir / "FORMAT_CHECK_RESULTS_ONLY.png", smoke=True))
        print(make_settlement(frame, args.output_dir / "FORMAT_CHECK_TRANSFERS_ONLY.png", smoke=True))
        return
    source = (args.comparison_dir or args.output_dir / "comparison") / "comparison_mean.csv"
    if not source.exists():
        raise FileNotFoundError(f"Evaluated means are not ready: {source}")
    frame = pd.read_csv(source)
    print(make_results(frame, args.output_dir / "合同机制结果对比.png"))
    print(make_settlement(frame, args.output_dir / "合同结算与平台收益.png"))
    if args.include_joint:
        print(make_results(frame, args.output_dir / "四主体联合训练结果.png", phase="ppo_joint_users"))
        print(make_settlement(frame, args.output_dir / "四主体联合训练结算.png", phase="ppo_joint_users"))


if __name__ == "__main__":
    main()
