"""GPU-hour contracts with separate cash accounts and voluntary carbon value.

The generation-side policy carbon charge is already included in electricity
prices. The firm's internal carbon value is a preference, never a second cash
charge. Consumption footprint and actual grid emissions are reported separately.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

from .carbon import nodal_carbon_intensity, consumption_emissions_t
from .cached_grid import MemoizedGridClearing
from .market import MarketEpisode
from .learning import ActorEnv


class GpuMarketEpisode(MarketEpisode):
    def current_user_observation(self):
        observation = super().current_user_observation()
        # Express monetary option differences in yuan/GPUh rather than a tiny
        # fraction of a whole GPU rental invoice. Original bill is zero here.
        # This is an invertible unit change; private IR and argmin are unchanged.
        options = observation[6:76].reshape(10, 7)
        job = self.jobs[self.position]
        original_bill = job.work_cu * self.config.base_price_yuan_per_cu
        # Derive the differences from known currency values before float32
        # conversion, avoiding cancellation of already-rounded invoice ratios.
        options[:, 2] = [(self.option_bill(job, option) - original_bill) / job.work_cu
                         for option in job.options]
        options[:, 3] = [option.user_loss_yuan / job.work_cu for option in job.options]
        observation[1] = job.work_cu / 20000
        return observation

    def step_user(self, action):
        position = self.position
        reward, done, info = super().step_user(action)
        # Fixed currency per work unit, independent of the firm's rental price.
        reward = info["utility_gain_yuan"] / self.jobs[position].work_cu
        return float(reward), done, info

    def _metrics(self, clearing, bills, losses):
        metrics = super()._metrics(clearing, bills, losses)
        cfg, scenario = self.config, self.scenario
        load = scenario.load_for(self.selected)
        dc_power = load - scenario.background_mw
        ci = nodal_carbon_intensity(scenario.grid_spec, clearing, load)
        footprint = consumption_emissions_t(dc_power, ci, cfg.dt_hours)
        fixed = float(cfg.daily_fixed_compute_cost_yuan)
        eta = float(cfg.internal_carbon_value_yuan_per_t)
        # Parent system cost contains fuel + variable compute/network operations
        # + user inconvenience; common equipment cost is separately disclosed.
        metrics["dispatch_system_cost_yuan"] = metrics["system_cost_yuan"]
        metrics["fixed_compute_cost_yuan"] = fixed
        metrics["full_system_cost_yuan"] = metrics["system_cost_yuan"] + fixed
        metrics["platform_profit_before_fixed_cost_yuan"] = metrics["platform_profit_yuan"]
        metrics["platform_profit_yuan"] -= fixed
        metrics["dc_consumption_emissions_t"] = footprint
        metrics["internal_carbon_value_yuan_per_t"] = eta
        metrics["internal_carbon_valuation_yuan"] = eta * footprint
        metrics["platform_objective_yuan"] = metrics["platform_profit_yuan"] - eta * footprint
        metrics["policy_carbon_price_yuan_per_t"] = clearing.policy_carbon_price_yuan_per_t
        metrics["cap_shadow_price_yuan_per_t"] = clearing.cap_shadow_price_yuan_per_t
        metrics["generator_carbon_payment_yuan"] = clearing.carbon_payment_yuan
        metrics["grid_dispatch_objective_yuan"] = clearing.dispatch_objective_yuan
        metrics["dc_energy_mwh"] = float(np.sum(dc_power) * cfg.dt_hours)
        metrics["grid_energy_mwh"] = float(np.sum(load) * cfg.dt_hours)
        metrics["dc_energy_share"] = metrics["dc_energy_mwh"] / metrics["grid_energy_mwh"]
        bg_cost = float(scenario.calibration["background_only_generation_cost_yuan"])
        metrics["background_only_generation_cost_yuan"] = bg_cost
        metrics["incremental_system_cost_yuan"] = metrics["system_cost_yuan"] - bg_cost
        metrics["work_gpu_hours"] = metrics["work_cu"]
        return metrics

    def outcome(self):
        result = super().outcome()
        metrics = result.metrics
        for name in ("dispatch_system_cost_yuan", "full_system_cost_yuan",
                     "platform_objective_yuan", "dc_consumption_emissions_t", "carbon_emissions_t",
                     "incremental_system_cost_yuan", "dc_energy_mwh"):
            metrics[f"baseline_{name}"] = self.baseline_metrics[name]
            metrics[f"delta_{name}"] = metrics[name] - self.baseline_metrics[name]
        metrics["grid_emissions_reduction_t"] = -metrics["delta_carbon_emissions_t"]
        metrics["dc_footprint_reduction_t"] = -metrics["delta_dc_consumption_emissions_t"]
        metrics["platform_objective_participates"] = metrics["delta_platform_objective_yuan"] >= -1e-6
        return result


class GpuPlatformEnv(ActorEnv):
    """Same firm objective for both information ablations, fixed yuan units."""
    def __init__(self, *args, reward_scale_yuan=1000.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_scale_yuan = float(reward_scale_yuan)

    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)
        reward = info["market_metrics"]["delta_platform_objective_yuan"] / self.reward_scale_yuan
        return observation, float(reward), terminated, truncated, info


def gpu_episode_factory(config=None, maxsize=64, grid_cache_size=512):
    from .gpu_scenario import GpuMarketConfig, make_gpu_scenario
    config = config or GpuMarketConfig()

    @lru_cache(maxsize=maxsize)
    def scenario(seed):
        return make_gpu_scenario(int(seed), config)

    @lru_cache(maxsize=maxsize)
    def grid(seed):
        return MemoizedGridClearing(scenario(seed).grid_spec, maxsize=grid_cache_size)

    def factory(seed):
        return GpuMarketEpisode(scenario(int(seed)), grid(int(seed)))
    return factory
