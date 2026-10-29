"""Conditional flexibility contracts and resource-savings sharing settlement.

The buyer is a modeled cost-recovery system-resource procurement account,
not a claim that a real ISO receives the system's fuel-cost savings as cash.
It contracts before execution, separately records both payment directions,
and offsets benefits already captured through ordinary electricity bills.
Settlement never changes completed jobs, locked user invoices or resource
costs. It can leave the platform with a loss on an unsuccessful offer.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .cached_grid import MemoizedGridClearing
from .gpu_market import GpuMarketEpisode
from .gpu_scenario import GpuMarketConfig, make_gpu_scenario
from .learning import ActorEnv


@dataclass(frozen=True)
class FlexMarketConfig(GpuMarketConfig):
    """A preannounced resource procurement contract around the same GPU jobs."""

    conditional_rebates: bool = True
    sharing_enabled: bool = True
    resource_share: float = .8
    carbon_eligibility: bool = True
    carbon_tolerance_t: float = 1e-7
    # The primary experiment gives the firm ordinary cash-profit preferences;
    # the procurement contract supplies its resource and carbon incentives.
    internal_carbon_value_yuan_per_t: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        for name in ("conditional_rebates", "sharing_enabled", "carbon_eligibility"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        if not np.isfinite(self.resource_share) or not 0 <= self.resource_share <= 1:
            raise ValueError("resource_share must be finite and between zero and one")
        if not np.isfinite(self.carbon_tolerance_t) or self.carbon_tolerance_t < 0:
            raise ValueError("carbon_tolerance_t must be finite and nonnegative")


class FlexMarketEpisode(GpuMarketEpisode):
    """Accepted users choose only their contracts; the buyer settles afterward."""

    def option_bill(self, job, option):
        if (self.config.conditional_rebates
                and not (option.migrated or option.delay_hours > 0)):
            return job.work_cu * self.config.base_price_yuan_per_cu
        return super().option_bill(job, option)

    def _reference_accounts(self):
        """Use immutable original contracts, including during baseline init."""
        cfg, scenario = self.config, self.scenario
        generation = float(self.baseline_clearing.total_generation_cost)
        operations = float(sum(job.options[0].operating_cost_yuan for job in self.jobs))
        dc_power = scenario.baseline_load_mw - scenario.background_mw
        bill = float(np.sum(dc_power * self.baseline_clearing.lmp_yuan_per_mwh) * cfg.dt_hours)
        revenue = float(np.sum(self.baseline_bills))
        profit = revenue - bill - operations - cfg.daily_fixed_compute_cost_yuan
        return generation, operations, bill, revenue, profit

    def _metrics(self, clearing, bills, losses):
        metrics = super()._metrics(clearing, bills, losses)
        cfg = self.config
        base_gen, base_ops, base_bill, base_revenue, base_profit = self._reference_accounts()
        resource_saving = ((base_gen + base_ops)
                           - (metrics["generation_cost_yuan"] + metrics["compute_migration_cost_yuan"]))
        private_pre_rebate_saving = ((base_bill + base_ops)
                                     - (metrics["electricity_bill_yuan"]
                                        + metrics["compute_migration_cost_yuan"]))
        rebate = base_revenue - metrics["service_revenue_yuan"]
        carbon_change = float(clearing.emissions_t - self.baseline_clearing.emissions_t)
        carbon_eligible = bool(not cfg.carbon_eligibility
                               or carbon_change <= cfg.carbon_tolerance_t)
        # With worsening resources the firm bears the complete resource loss.
        # Positive sharing is available only for the contracted carbon service.
        entitlement = min(resource_saving, 0.)
        if resource_saving > 0 and carbon_eligible:
            entitlement = cfg.resource_share * resource_saving
        transfer = (entitlement - private_pre_rebate_saving) if cfg.sharing_enabled else 0.
        buyer_before = resource_saving - private_pre_rebate_saving
        buyer_after = buyer_before - transfer
        cash_before = metrics["platform_profit_yuan"]
        cash_before_fixed = metrics["platform_profit_before_fixed_cost_yuan"]
        metrics["platform_profit_yuan"] = cash_before + transfer
        metrics["platform_profit_before_fixed_cost_yuan"] = cash_before_fixed + transfer
        metrics["platform_objective_yuan"] = (metrics["platform_profit_yuan"]
                                               - metrics["internal_carbon_valuation_yuan"])
        cash_change_before = private_pre_rebate_saving - rebate
        cash_change_after = cash_change_before + transfer
        system_saving = resource_saving - metrics["user_loss_yuan"]
        allocation_sum = cash_change_after + metrics["user_gain_yuan"] + buyer_after
        metrics.update(
            conditional_rebates=bool(cfg.conditional_rebates),
            sharing_enabled=bool(cfg.sharing_enabled),
            resource_share=float(cfg.resource_share),
            carbon_eligibility_required=bool(cfg.carbon_eligibility),
            carbon_eligible=carbon_eligible,
            flexibility_resource_saving_yuan=float(resource_saving),
            platform_pre_rebate_saving_yuan=float(private_pre_rebate_saving),
            user_rebate_yuan=float(rebate),
            shared_resource_entitlement_yuan=float(entitlement if cfg.sharing_enabled else 0.),
            flexibility_settlement_yuan=float(transfer),
            buyer_payment_to_platform_yuan=float(max(transfer, 0.)),
            platform_payment_to_buyer_yuan=float(max(-transfer, 0.)),
            buyer_pre_settlement_benefit_yuan=float(buyer_before),
            buyer_post_settlement_benefit_yuan=float(buyer_after),
            platform_profit_before_transfer_yuan=float(cash_before),
            platform_profit_before_fixed_and_transfer_yuan=float(cash_before_fixed),
            cash_change_before_transfer_yuan=float(cash_change_before),
            cash_change_after_transfer_yuan=float(cash_change_after),
            baseline_cash_profit_before_transfer_yuan=float(base_profit),
            contract_system_cost_saving_yuan=float(system_saving),
            surplus_allocation_sum_yuan=float(allocation_sum),
            surplus_identity_error_yuan=float(allocation_sum - system_saving),
            settlement_grid_emissions_change_t=carbon_change,
        )
        return metrics


class FlexPlatformEnv(ActorEnv):
    """The private cash return includes the actual preannounced settlement."""

    def __init__(self, *args, reward_scale_yuan=1000., **kwargs):
        super().__init__(*args, **kwargs)
        if self.actor != "platform" or not np.isfinite(reward_scale_yuan) or reward_scale_yuan <= 0:
            raise ValueError("FlexPlatformEnv needs the platform actor and a positive fixed reward scale")
        self.reward_scale_yuan = float(reward_scale_yuan)

    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)
        reward = info["market_metrics"]["delta_platform_profit_yuan"] / self.reward_scale_yuan
        return observation, float(reward), terminated, truncated, info


def flex_episode_factory(config=None, maxsize=64, grid_cache_size=512):
    """Same scenarios and OPF as the GPU study, independent booking state."""
    config = config or FlexMarketConfig()
    if not isinstance(config, FlexMarketConfig):
        raise TypeError("flex_episode_factory requires FlexMarketConfig")

    @lru_cache(maxsize=maxsize)
    def scenario(seed):
        return make_gpu_scenario(int(seed), config)

    @lru_cache(maxsize=maxsize)
    def grid(seed):
        return MemoizedGridClearing(scenario(seed).grid_spec, maxsize=grid_cache_size)

    def factory(seed):
        return FlexMarketEpisode(scenario(int(seed)), grid=grid(int(seed)))
    return factory
