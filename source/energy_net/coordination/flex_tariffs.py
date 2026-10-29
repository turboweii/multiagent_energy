"""Public contract forecasts for conditional rebates and resource sharing.

The 77 raw price menus remain signal-independent. A forecast changes one job
while every other job keeps its original contract, then clears that public
counterfactual. It never evaluates a menu with simulated users or reads their
delay/migration preferences. Forecasts are not realised menu returns: capacity,
prices, emissions and settlement can change when several users respond.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .carbon import nodal_carbon_intensity
from .gpu_tariffs import (
    FixedGpuTariffCatalogue, candidate_gpu_tariff_observation,
    feature_names as gpu_feature_names,
)


FlexTariffCatalogue = FixedGpuTariffCatalogue


def quote_bill_for_option(job, option, tariff_t, base_price, conditional_rebates):
    """Known contract invoice, with ``tariff_t`` in yuan/GPUh and [T,region]."""
    unchanged = not (option.migrated or option.delay_hours > 0)
    if option.is_baseline or (conditional_rebates and unchanged):
        return float(job.work_cu * base_price)
    return float(np.sum(option.allocation_cu * tariff_t))


def feature_names():
    names = list(gpu_feature_names()[:-1])
    names.extend(("conditional_rebates", "sharing_enabled", "resource_share",
                  "carbon_eligibility", "carbon_tolerance_microtonnes"))
    for prefix in (
        "owner_best_forecast_resource_saving_yuan_per_gpuh",
        "owner_mean_forecast_resource_saving_yuan_per_gpuh",
        "owner_worst_forecast_resource_saving_yuan_per_gpuh",
        "owner_resource_positive_offer_fraction",
        "owner_best_forecast_grid_emission_reduction_kg_per_gpuh",
        "owner_mean_forecast_grid_emission_reduction_kg_per_gpuh",
        "owner_grid_nonincreasing_offer_fraction",
        "owner_best_forecast_private_margin_yuan_per_gpuh",
        "owner_best_forecast_settled_margin_yuan_per_gpuh",
        "owner_mean_forecast_settled_margin_yuan_per_gpuh",
        "owner_worst_forecast_settled_margin_yuan_per_gpuh",
        "owner_best_forecast_transfer_yuan_per_gpuh",
        "owner_mean_forecast_transfer_yuan_per_gpuh",
        "owner_best_forecast_buyer_residual_yuan_per_gpuh",
        "owner_feasible_discounted_offer_fraction",
    ):
        names.extend(f"{prefix}_{owner}" for owner in range(3))
    names.append("unique_raw_action")
    return tuple(names)


flex_tariff_feature_names = feature_names


@dataclass(frozen=True)
class PublicOptionForecast:
    """One-job response of the rule-cleared network; no user choice forecast."""

    option: object
    allocation: np.ndarray
    feasible: bool
    reference_bill_saving_yuan: float
    reference_footprint_saving_t: float
    operating_increase_yuan: float
    resource_saving_yuan: float
    private_pre_rebate_saving_yuan: float
    grid_emission_reduction_t: float


def public_option_forecasts(episode):
    """Clear public one-job alternatives against immutable original contracts."""
    config, scenario, reference = episode.config, episode.scenario, episode.baseline_clearing
    dt = config.dt_hours
    baseline_load = scenario.baseline_load_mw
    baseline_dc = baseline_load - scenario.background_mw
    baseline_bill = float(np.sum(baseline_dc * reference.lmp_yuan_per_mwh) * dt)
    intensity = nodal_carbon_intensity(scenario.grid_spec, reference, baseline_load)
    original_alloc = sum((job.options[0].allocation_cu for job in scenario.jobs),
                         np.zeros_like(baseline_load))
    capacity = np.asarray(config.capacity_cu_per_hour) * dt
    public = []
    for job in scenario.jobs:
        original = job.options[0]
        original_power = scenario.option_power_mw(original)
        forecasts = []
        for option in job.options[1:]:
            allocation = original_alloc - original.allocation_cu + option.allocation_cu
            possible = bool(np.all(allocation <= capacity + 1e-7)
                            and np.all(option.transfer_gb <= config.bandwidth_gb_per_hour * dt + 1e-7))
            new_power = scenario.option_power_mw(option)
            power_change = original_power - new_power
            operating_increase = float(option.operating_cost_yuan - original.operating_cost_yuan)
            # Same physical contract (option 1) reuses the actual baseline solve.
            result = reference
            changed = bool(option.migrated or option.delay_hours > 0)
            if possible and changed:
                result = episode.grid.solve(baseline_load - power_change, scenario.carbon_budget_t)
                possible = bool(result.feasible)
            if possible:
                load = baseline_load - power_change
                bill = float(np.sum((load - scenario.background_mw) * result.lmp_yuan_per_mwh) * dt)
                resource = float(reference.total_generation_cost - result.total_generation_cost - operating_increase)
                private = float(baseline_bill - bill - operating_increase)
                reduction = float(reference.emissions_t - result.emissions_t)
            else:
                resource = private = reduction = 0.
            forecasts.append(PublicOptionForecast(
                option=option, allocation=allocation, feasible=possible,
                reference_bill_saving_yuan=float(np.sum(power_change * reference.lmp_yuan_per_mwh) * dt),
                reference_footprint_saving_t=float(np.sum(power_change * intensity) * dt),
                operating_increase_yuan=operating_increase,
                resource_saving_yuan=resource,
                private_pre_rebate_saving_yuan=private,
                grid_emission_reduction_t=reduction,
            ))
        public.append((job, tuple(forecasts)))
    return tuple(public), original_alloc


def _entitlement(config, resource_saving, emissions_reduction):
    eligible = (not config.carbon_eligibility
                or emissions_reduction >= -config.carbon_tolerance_t)
    if resource_saving <= 0:
        return float(resource_saving)
    return float(config.resource_share * resource_saving if eligible else 0.)


def candidate_flex_tariff_observation(episode, catalogue=None, *, forecasts=None):
    """Same action rows across contracts, with rule-consistent public forecasts."""
    config = episode.config
    catalogue = catalogue or FixedGpuTariffCatalogue(config)
    # Preserve the existing public context, units and explicit carbon signals.
    # All original proposal statistics are recomputed below using eligible bills.
    base = candidate_gpu_tariff_observation(episode, catalogue, carbon_aware=True)
    base_names = gpu_feature_names()
    old_prefixes = (
        "owner_original_windfall_yuan_per_gpuh", "owner_best_price_cut_yuan_per_gpuh",
        "owner_best_forecast_margin_yuan_per_gpuh", "owner_mean_forecast_margin_yuan_per_gpuh",
        "owner_worst_forecast_margin_yuan_per_gpuh", "owner_best_energy_saving_yuan_per_gpuh",
        "owner_mean_energy_saving_yuan_per_gpuh", "owner_profitable_offer_fraction",
        "owner_discounted_offer_fraction", "carbon_owner_best_saving_kg_per_gpuh",
        "carbon_owner_mean_saving_kg_per_gpuh", "carbon_owner_best_internal_gain_yuan_per_gpuh",
        "carbon_owner_best_adjusted_margin_yuan_per_gpuh",
        "carbon_owner_mean_adjusted_margin_yuan_per_gpuh",
        "carbon_owner_worst_adjusted_margin_yuan_per_gpuh",
    )
    old_columns = np.asarray([[base_names.index(f"{prefix}_{owner}") for owner in range(3)]
                              for prefix in old_prefixes])
    public, original_alloc = public_option_forecasts(episode) if forecasts is None else forecasts
    capacity = np.asarray(config.capacity_cu_per_hour) * config.dt_hours
    blocks = np.minimum(np.arange(config.n_steps) * config.n_blocks // config.n_steps,
                        config.n_blocks - 1)
    eta = float(config.internal_carbon_value_yuan_per_t)
    flags = np.asarray([float(config.conditional_rebates), float(config.sharing_enabled),
                        float(config.resource_share), float(config.carbon_eligibility),
                        float(config.carbon_tolerance_t) * 1e6])
    rows = []
    for index in range(catalogue.n_actions):
        raw = catalogue.decode(episode, index)
        discounts = np.asarray(config.discount_levels)[raw[:12]].reshape(3, 4)
        quota = np.asarray(config.quota_levels)[raw[12:]]
        limit = np.maximum(original_alloc, capacity * quota)
        tariff_t = (config.base_price_yuan_per_cu * (1 - discounts))[:, blocks].T
        old_stats, new_stats, owner_count = np.zeros((15, 3)), np.zeros((15, 3)), np.zeros(3)
        for job, alternatives in public:
            owner, quantity = job.user_id, job.work_cu
            owner_count[owner] += 1
            original_invoice = quantity * config.base_price_yuan_per_cu
            # Option 1 has original physics and is the no-movement price option.
            old_stats[0, owner] += (original_invoice - quote_bill_for_option(
                job, job.options[1], tariff_t, config.base_price_yuan_per_cu,
                config.conditional_rebates)) / quantity
            cuts, stale_margins, stale_bill, footprints = [], [], [], []
            resources, reductions, private_margins, settled_margins, transfers, residuals = [], [], [], [], [], []
            for forecast in alternatives:
                if not forecast.feasible or np.any(forecast.allocation > limit + 1e-7):
                    continue
                bill = quote_bill_for_option(job, forecast.option, tariff_t,
                                             config.base_price_yuan_per_cu, config.conditional_rebates)
                rebate = original_invoice - bill
                if rebate <= 1e-9:
                    continue
                resource = forecast.resource_saving_yuan
                private = forecast.private_pre_rebate_saving_yuan
                entitlement = _entitlement(config, resource, forecast.grid_emission_reduction_t)
                transfer = entitlement - private if config.sharing_enabled else 0.
                residual = resource - private - transfer
                cuts.append(rebate / quantity)
                stale_margins.append((forecast.reference_bill_saving_yuan - forecast.operating_increase_yuan - rebate) / quantity)
                stale_bill.append(forecast.reference_bill_saving_yuan / quantity)
                footprints.append(forecast.reference_footprint_saving_t / quantity)
                resources.append(resource / quantity)
                reductions.append(forecast.grid_emission_reduction_t * 1000 / quantity)
                private_margins.append((private - rebate) / quantity)
                settled_margins.append((private + transfer - rebate) / quantity)
                transfers.append(transfer / quantity)
                residuals.append(residual / quantity)
            if cuts:
                stale_margins, footprints = np.asarray(stale_margins), np.asarray(footprints)
                adjusted = stale_margins + eta * footprints
                old_stats[1:, owner] += [max(cuts), stale_margins.max(), stale_margins.mean(), stale_margins.min(),
                    max(stale_bill), np.mean(stale_bill), np.mean(stale_margins > 0),
                    len(cuts) / max(1, len(alternatives)), footprints.max() * 1000,
                    footprints.mean() * 1000, eta * footprints.max(), adjusted.max(), adjusted.mean(), adjusted.min()]
                resources, reductions = np.asarray(resources), np.asarray(reductions)
                settled_margins = np.asarray(settled_margins)
                new_stats[:, owner] += [resources.max(), resources.mean(), resources.min(),
                    np.mean(resources > 0), reductions.max(), reductions.mean(),
                    np.mean(reductions >= -config.carbon_tolerance_t * 1000 / quantity),
                    max(private_margins), settled_margins.max(), settled_margins.mean(), settled_margins.min(),
                    max(transfers), np.mean(transfers), max(residuals), len(cuts) / max(1, len(alternatives))]
        old_stats /= np.maximum(owner_count, 1)[None, :]
        new_stats /= np.maximum(owner_count, 1)[None, :]
        row = base[index].copy()
        row[old_columns] = old_stats
        rows.append(np.r_[row[:-1], flags, new_stats.ravel(), row[-1]])
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (catalogue.n_actions, len(feature_names())) or not np.isfinite(result).all():
        raise ValueError("Flex observations need finite equal-size public rows for every menu")
    return result


class FlexTariffObserver:
    """Cache public counterfactuals and quote rows once per immutable scenario."""

    def __init__(self, catalogue=None, maxsize=512):
        self.catalogue = catalogue or FixedGpuTariffCatalogue()
        self.maxsize = int(maxsize)
        if self.maxsize < 1:
            raise ValueError("Flex observer cache must hold at least one scenario")
        self.training_max_discount = .03
        self.feature_names = feature_names()
        self._cache = OrderedDict()
        self._forecast_cache = OrderedDict()

    def _forecasts(self, episode):
        key = int(episode.scenario.seed)
        entry = self._forecast_cache.get(key)
        if entry is None or entry[0] is not episode.scenario:
            entry = (episode.scenario, public_option_forecasts(episode))
            self._forecast_cache[key] = entry
            if len(self._forecast_cache) > self.maxsize:
                self._forecast_cache.popitem(last=False)
        self._forecast_cache.move_to_end(key)
        return entry[1]

    def raw(self, episode):
        key = int(episode.scenario.seed)
        entry = self._cache.get(key)
        if entry is None or entry[0] is not episode.scenario:
            values = candidate_flex_tariff_observation(episode, self.catalogue, forecasts=self._forecasts(episode))
            entry = (episode.scenario, values)
            self._cache[key] = entry
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return entry[1].copy()

    def __call__(self, episode, *, training=False):
        values = self.raw(episode)
        if training:
            for index, entry in enumerate(self.catalogue.entries):
                if entry is not None and entry[1] > self.training_max_discount + 1e-12:
                    values[index, -1] = 0.
        return values

    def fit_normalization(self, episode_factory, seeds):
        rows = np.concatenate([self.raw(episode_factory(int(seed)))[:, :-1] for seed in seeds], axis=0)
        mean = rows.mean(axis=0, dtype=np.float64)
        scale = np.maximum(rows.std(axis=0, dtype=np.float64), .01)
        return mean.astype(np.float32), scale.astype(np.float32)
