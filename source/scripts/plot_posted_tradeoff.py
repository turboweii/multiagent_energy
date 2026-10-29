#!/usr/bin/env python3
"""Plot measured social saving versus platform profit gain in separate cohorts.

Example:
  python scripts/plot_posted_tradeoff.py --experiment /path/to/seed0

Only completed outcome tables are read. Four private draws are averaged within
each weather day before averaging the 32 weather days and computing marginal
95% weather-level t intervals. No model, template or controller is selected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from summarize_posted_offers import (
    ALIASES, MATRIX, RULES, ZERO_HISTORY, decode_case, interval,
    is_no_history, scheme_color,
)


CORE_CONTROLS = ("exit", *RULES, "ppo", ZERO_HISTORY)
LABELS = {
    "exit": "Old contracts / no new offer",
    "strong_fixed_raw": "Train-selected fixed single offer",
    MATRIX: "Fixed rebate matrix + greedy routing",
    "strong_value_share_rule": "Resource-value compensation rule",
    "learned_response_prior": "Logit response prior",
    "learned_response_online_map": "Logit + online MAP",
    "ppo": "PPO / full explicit history",
    ZERO_HISTORY: "PPO / same weights, history zeroed",
}
MARKERS = {
    "exit": "x", "strong_fixed_raw": "s", MATRIX: "D",
    "strong_value_share_rule": "^", "learned_response_prior": "<",
    "learned_response_online_map": ">", "ppo": "*", ZERO_HISTORY: "X",
}


def load_tradeoff_cohort(directory, name, *, expected_weather=32, expected_draws=4,
                         required_schemes=CORE_CONTROLS):
    """Read one frozen-user cohort without mixing its cases with another."""
    directory = Path(directory)
    frame = pd.read_csv(directory / "test_outcomes.csv",dtype={"case_id":"int64"},encoding="utf-8-sig").copy()
    if frame.empty or frame.duplicated(["scheme","case_id"]).any():
        raise ValueError(f"Empty or repeated scheme/case outcomes: {directory}")
    frame["scheme"] = frame.scheme.astype(str)
    missing = set(required_schemes)-set(frame.scheme)
    if missing:
        raise ValueError(f"Missing preset controls in {directory}: {sorted(missing)}")
    addresses = [decode_case(case) for case in frame.case_id]
    frame["public_seed"] = [public for public,_ in addresses]
    frame["preference_seed"] = [private for _,private in addresses]
    for canonical in ("S","profit"):
        source = next((column for column in ALIASES[canonical] if column in frame),None)
        if source is None:
            raise ValueError(f"Missing realised {canonical} outcome in {directory}")
        frame[canonical] = pd.to_numeric(frame[source],errors="raise")
    if not np.isfinite(frame[["S","profit"]].to_numpy()).all():
        raise ValueError(f"Nonfinite realised money account in {directory}")
    schemes = list(dict.fromkeys(frame.scheme))
    case_sets = [set(frame.loc[frame.scheme==scheme,"case_id"]) for scheme in schemes]
    if any(cases!=case_sets[0] for cases in case_sets[1:]):
        raise ValueError(f"Controls must use the same cases within {directory}")
    weather = frame.groupby(["scheme","public_seed"],sort=False)[["S","profit"]].mean()
    weather["private_draws"] = frame.groupby(["scheme","public_seed"]).case_id.nunique()
    points = []
    for scheme in schemes:
        days = weather.xs(scheme,level="scheme")
        if expected_weather is not None and len(days)!=expected_weather:
            raise ValueError(f"Expected {expected_weather} weather days for {scheme} in {directory}")
        if expected_draws is not None and not np.all(days.private_draws==expected_draws):
            raise ValueError(f"Expected {expected_draws} private draws per weather for {scheme} in {directory}")
        social,profit = interval(days.S.to_numpy()),interval(days.profit.to_numpy())
        points.append(dict(cohort=name,scheme=scheme,S_mean_yuan=social["mean"],
            S_ci95_half_yuan=social["half"],profit_mean_yuan=profit["mean"],
            profit_ci95_half_yuan=profit["half"],weather_days=len(days),
            min_private_draws=int(days.private_draws.min()),max_private_draws=int(days.private_draws.max())))
    report = directory / "test_report.json"
    metadata = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
    return dict(name=name,directory=directory,metadata=metadata,weather=weather,
                points=points,schemes=schemes)


def nondominated_observed_means(points,tolerance=1e-9):
    """Maximise both measured coordinates within the supplied finite set."""
    values = np.asarray([[point["S_mean_yuan"],point["profit_mean_yuan"]] for point in points])
    result = []
    for value in values:
        dominated = np.all(values>=value-tolerance,axis=1) & np.any(values>value+tolerance,axis=1)
        result.append(not bool(dominated.any()))
    return result


def control_label(scheme,cohort):
    if is_no_history(scheme,cohort):
        return "PPO / independently trained without history"
    if scheme=="ppo" and (cohort["metadata"].get("model_metadata") or {}).get("history") is False:
        return "PPO / independently trained without history"
    return LABELS.get(scheme,scheme)


def _scheme_order(scheme,cohort):
    preset = ("exit",*RULES,"ppo")
    if scheme in preset:return preset.index(scheme)
    if is_no_history(scheme,cohort):return len(preset)
    if scheme==ZERO_HISTORY:return len(preset)+1
    return len(preset)+2


def _letters(index):
    value = index+1;result = ""
    while value:
        value,remainder = divmod(value-1,26)
        result = chr(65+remainder)+result
    return result


def _overlap(first,second):
    return max(0.,min(first[2],second[2])-max(first[0],second[0]))*max(
        0.,min(first[3],second[3])-max(first[1],second[1]))


def _annotate_letters(axis,points,identifiers,colors):
    """Place short IDs around close points, keeping full text in the legend."""
    factor = axis.figure.dpi/72
    positions = axis.transData.transform(np.asarray([[point["S_mean_yuan"]/1000,
        point["profit_mean_yuan"]/1000] for point in points]))
    radius = 9*factor
    marker_boxes = [(x-radius,y-radius,x+radius,y+radius) for x,y in positions]
    occupied = []
    bounds = axis.get_window_extent()
    offsets = ((12,12),(12,-12),(-12,12),(-12,-12),(24,0),(-24,0),(0,24),(0,-24),
               (24,20),(-24,20),(24,-20),(-24,-20),(38,0),(-38,0),(0,38),(0,-38),
               (40,26),(-40,26),(40,-26),(-40,-26))
    for point,(x,y) in zip(points,positions):
        label = identifiers[point["scheme"]]
        half_width,half_height = (4+3*len(label))*factor,7*factor
        candidates = []
        for dx,dy in offsets:
            center_x,center_y = x+dx*factor,y+dy*factor
            rectangle = (center_x-half_width,center_y-half_height,center_x+half_width,center_y+half_height)
            outside = (rectangle[0]<bounds.x0 or rectangle[1]<bounds.y0
                       or rectangle[2]>bounds.x1 or rectangle[3]>bounds.y1)
            score = (1e8*outside+10*sum(_overlap(rectangle,other) for other in occupied)
                     +sum(_overlap(rectangle,other) for other in marker_boxes)+.01*(dx*dx+dy*dy))
            candidates.append((score,dx,dy,rectangle))
        _,dx,dy,rectangle = min(candidates,key=lambda item:item[0])
        occupied.append(rectangle)
        axis.annotate(label,(point["S_mean_yuan"]/1000,point["profit_mean_yuan"]/1000),
            xytext=(dx,dy),textcoords="offset points",ha="center",va="center",
            fontsize=9,fontweight="bold",color=colors[point["scheme"]],zorder=20,
            bbox=dict(boxstyle="round,pad=.12",facecolor="white",edgecolor="none",alpha=.88),
            arrowprops=dict(arrowstyle="-",color=colors[point["scheme"]],lw=.65,alpha=.7,
                            shrinkA=2,shrinkB=6))


def create_tradeoff_figure(cohorts,*,dpi=200,errorbars=True,highlight_nondominated=True):
    if not cohorts:raise ValueError("Provide at least one completed frozen-user cohort")
    representatives = {}
    for cohort in cohorts:
        for scheme in cohort["schemes"]:representatives.setdefault(scheme,cohort)
    schemes = sorted(representatives,key=lambda scheme:(
        _scheme_order(scheme,representatives[scheme]),list(representatives).index(scheme)))
    identifiers = {scheme:_letters(index) for index,scheme in enumerate(schemes)}
    colors = {scheme:scheme_color(scheme,representatives[scheme]) for scheme in schemes}
    markers = {scheme:("P" if is_no_history(scheme,representatives[scheme]) else MARKERS.get(scheme,"o"))
               for scheme in schemes}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"savefig.facecolor":"white"})
    figure,axes = plt.subplots(1,len(cohorts),figsize=(14.5,8.1),dpi=dpi,sharex=True,sharey=True,squeeze=False)
    axes = axes[0]
    figure.subplots_adjust(left=.08,right=.98,bottom=.33,top=.85,wspace=.14)
    plotted = []
    for axis,cohort in zip(axes,cohorts):
        flags = nondominated_observed_means(cohort["points"])
        for point,nondominated in zip(cohort["points"],flags):
            scheme = point["scheme"];color = colors[scheme]
            x,y = point["S_mean_yuan"]/1000,point["profit_mean_yuan"]/1000
            if errorbars:
                axis.errorbar(x,y,xerr=point["S_ci95_half_yuan"]/1000,
                    yerr=point["profit_ci95_half_yuan"]/1000,fmt="none",ecolor=color,
                    elinewidth=.85,capsize=2.5,alpha=.3,zorder=1)
            size = 12 if scheme=="ppo" else 8
            axis.plot(x,y,marker=markers[scheme],linestyle="none",markersize=size,color=color,
                markeredgecolor=(color if markers[scheme]=="x" else "white"),markeredgewidth=.65,zorder=5)
            if highlight_nondominated and nondominated:
                axis.plot(x,y,marker="o",linestyle="none",markersize=size+5,
                    markerfacecolor="none",markeredgecolor="#292929",markeredgewidth=1.,zorder=6)
            plotted.append(dict(point,control_id=identifiers[scheme],
                                label=control_label(scheme,cohort),nondominated_observed_mean=nondominated))
        first = cohort["points"][0]
        draws = (str(first["min_private_draws"]) if first["min_private_draws"]==first["max_private_draws"]
                 else f"{first['min_private_draws']}-{first['max_private_draws']}")
        axis.set_title(f"{cohort['name']}\n{first['weather_days']} weather days, {draws} private draws/day",fontsize=12,pad=12)
        axis.set_xlabel("Social cost saving S (thousand yuan/day)",labelpad=10)
        axis.grid(True,color="#e7e7e7",linewidth=.7,zorder=0)
        axis.spines[["top","right"]].set_visible(False)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.yaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_ylabel("Platform profit gain ΔΠ (thousand yuan/day)",labelpad=10)
    all_points = [point for cohort in cohorts for point in cohort["points"]]
    for coordinate,half,setter in (("S_mean_yuan","S_ci95_half_yuan",axes[0].set_xlim),
                                 ("profit_mean_yuan","profit_ci95_half_yuan",axes[0].set_ylim)):
        lower = min(point[coordinate]-(point[half] if errorbars and np.isfinite(point[half]) else 0.) for point in all_points)/1000
        upper = max(point[coordinate]+(point[half] if errorbars and np.isfinite(point[half]) else 0.) for point in all_points)/1000
        width = max(upper-lower,.1)
        setter(lower-.10*width,upper+.13*width)
    handles = [Line2D([],[],linestyle="none",marker=markers[scheme],markersize=9,
        color=colors[scheme],label=f"{identifiers[scheme]}  "+textwrap.fill(
            control_label(scheme,representatives[scheme]),width=39)) for scheme in schemes]
    if highlight_nondominated:
        handles.append(Line2D([],[],linestyle="none",marker="o",markersize=12,
            markerfacecolor="none",markeredgecolor="#292929",label="Non-dominated observed means"))
    figure.legend(handles=handles,loc="lower center",bbox_to_anchor=(.53,.115),ncol=3,
                  frameon=False,fontsize=9,columnspacing=2.,handletextpad=.65,labelspacing=.85)
    figure.suptitle("Realised system-saving and platform-profit trade-offs",fontsize=16,y=.975)
    figure.text(.53,.925,"Compare controls within each panel; the frozen user policies differ between panels.",
                ha="center",fontsize=10,color="#454545")
    figure.text(.08,.045,
        "Each point averages private draws within weather first, then weather days. Bars: marginal 95% weather-level t intervals.\n"
        "Rings identify non-dominated means among the evaluated controls in that panel; they do not represent a global frontier.",
        fontsize=9,color="#454545",va="bottom",linespacing=1.5)
    figure.canvas.draw()
    for axis,cohort in zip(axes,cohorts):_annotate_letters(axis,cohort["points"],identifiers,colors)
    return figure,pd.DataFrame(plotted)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,required=True)
    parser.add_argument("--output",type=Path,help="Default: experiment directory")
    parser.add_argument("--dpi",type=int,default=200)
    parser.add_argument("--no-errorbars",action="store_true")
    parser.add_argument("--no-nondominated",action="store_true")
    args = parser.parse_args(argv)
    if args.dpi<72:parser.error("--dpi must be at least 72")
    experiment = args.experiment.resolve()
    output = (args.output or experiment).resolve()
    cohorts = [load_tradeoff_cohort(experiment/"baselines-initial","Initial frozen users"),
               load_tradeoff_cohort(experiment/"baselines-joint","Joint-trained frozen users")]
    figure,points = create_tradeoff_figure(cohorts,dpi=args.dpi,errorbars=not args.no_errorbars,
        highlight_nondominated=not args.no_nondominated)
    output.mkdir(parents=True,exist_ok=True)
    figure.savefig(output/"tradeoff.png",dpi=args.dpi)
    plt.close(figure)
    points.to_csv(output/"tradeoff_points.csv",index=False,encoding="utf-8-sig")
    print(json.dumps(dict(figure=str(output/"tradeoff.png"),points=str(output/"tradeoff_points.csv"),
        cohorts=[cohort["name"] for cohort in cohorts],statistical_unit="weather day after private-draw averaging")))


if __name__=="__main__":main()
