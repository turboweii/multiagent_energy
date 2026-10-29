"""Common physical carbon constraints for the legacy and amendment markets.

The daily allowance is fixed from the original no-shift workload before any
legacy quote or amendment. Both institutions inherit exactly this allowance.
The generation-side carbon price remains part of ordinary OPF clearing.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np

from .cached_grid import MemoizedGridClearing
from .flex_market import FlexMarketConfig, FlexMarketEpisode
from .gpu_scenario import GpuMarketConfig, make_gpu_scenario
from .grid import GridClearing


CAP_TOLERANCE_T = 1e-7


def legacy_capped_config(config=None):
    """Keep supplied physical inputs and quote eligibility; use old settlement.

    The default reproduces published-price legacy contracts. Supplying a
    conditional-rebate configuration permits a second legacy quote benchmark.
    Sharing, carbon-payment eligibility and internal carbon preferences are
    disabled: the carbon constraint is physical and common to every scheme.
    """
    config = config or FlexMarketConfig(conditional_rebates=False)
    if not isinstance(config, FlexMarketConfig):
        raise TypeError("The legacy capped market requires FlexMarketConfig or a subclass")
    return replace(config, sharing_enabled=False, carbon_eligibility=False,
                   internal_carbon_value_yuan_per_t=0.)


def make_capped_scenario(seed=0, config=None):
    """Return the original GPU jobs with a strategy-independent daily cap."""
    config = config or legacy_capped_config()
    if not isinstance(config, GpuMarketConfig):
        raise TypeError("The common capped scenario requires GpuMarketConfig or a subclass")
    scenario = make_gpu_scenario(int(seed), config)
    grid = GridClearing(scenario.grid_spec)
    reference = grid.solve(scenario.baseline_load_mw, None)
    if not reference.feasible:
        raise ValueError("The original GPU contracts must be physically feasible")
    cap = float(reference.emissions_t + CAP_TOLERANCE_T)
    scenario.carbon_budget_t = cap
    constrained = grid.solve(scenario.baseline_load_mw, cap)
    if not constrained.feasible:
        raise ValueError("The original GPU contracts must remain feasible at the common cap")
    background = grid.solve(scenario.background_mw, cap)
    if not background.feasible:
        raise ValueError("The same-cap background counterfactual is infeasible")
    scenario.calibration = dict(scenario.calibration,
        common_carbon_cap_t=cap,
        common_carbon_cap_source="original no-shift workload, before legacy quotes or amendments",
        original_unconstrained_emissions_t=float(reference.emissions_t),
        common_carbon_cap_numerical_margin_t=CAP_TOLERANCE_T,
        original_unconstrained_generation_cost_yuan=float(reference.total_generation_cost),
        original_same_cap_generation_cost_yuan=float(constrained.total_generation_cost),
        background_only_generation_cost_yuan=float(background.total_generation_cost),
        background_only_carbon_budget_t=cap,
        carbon_qualification="identical physical OPF cap in legacy and amendment stages")
    return scenario


def capped_episode_factory(config=None, maxsize=64, grid_cache_size=512):
    """Create independent legacy episodes sharing immutable scenario inputs.

    Configurations are normalised to old settlement while retaining the
    supplied physical configuration and conditional/published quote product.
    The scenario generator itself also accepts the new amendment config.
    """
    config = legacy_capped_config(config)
    if int(maxsize) != maxsize or maxsize < 1:
        raise ValueError("The scenario cache must hold at least one day")

    @lru_cache(maxsize=int(maxsize))
    def scenario(seed):
        return make_capped_scenario(int(seed), config)

    @lru_cache(maxsize=int(maxsize))
    def grid(seed):
        return MemoizedGridClearing(scenario(int(seed)).grid_spec,
                                    maxsize=grid_cache_size)

    def factory(seed):
        day = int(seed)
        return FlexMarketEpisode(scenario(day), grid=grid(day))

    factory.config = config
    factory.scenario_cache_info = scenario.cache_info
    return factory
