"""Public quotes and forecasts for amendments to already accepted bookings.

The outside booking is established before the amendment quote. Forecasts
change one public execution option while all other bookings remain fixed;
they do not simulate user responses or use private inconvenience values.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .carbon import nodal_carbon_intensity
from .gpu_tariffs import FixedGpuTariffCatalogue
from .market import encode_platform_action


def same_execution(first, second):
    """Execution equality is physical, independent of loss or old option labels."""
    return all(np.array_equal(getattr(first, name), getattr(second, name))
               for name in ("allocation_cu", "extra_energy_mwh", "transfer_gb"))


def quote_bill_for_option(job, option, discounts_t, outside_bill, *, amendment_open=True):
    """A changed execution gets an additional discount on its accepted invoice.

    ``discounts_t`` has shape [time, region], not currency units. The accepted
    outside unit price may already differ from the original GPU rental price.
    """
    if not amendment_open or same_execution(option, job.options[0]):
        return float(outside_bill)
    rebate_fraction = float(np.sum(option.allocation_cu * discounts_t) / job.work_cu)
    return float(outside_bill * (1 - rebate_fraction))


class AmendmentTariffCatalogue:
    """Exit, free amendments, then the 76 stable nonzero GPU quote templates."""

    n_actions = 78
    patterns = FixedGpuTariffCatalogue.patterns
    amplitudes = FixedGpuTariffCatalogue.amplitudes
    quotas = FixedGpuTariffCatalogue.quotas

    def __init__(self, config=None):
        self.config = config
        self.base = FixedGpuTariffCatalogue(config)
        self.entries = (None, ("zero_rebate_amendment", 0., 1.), *self.base.entries[1:])
        self.labels = ("outside_bookings", "zero_rebate_amendment", *self.base.labels[1:])

    def decode(self, episode, index):
        value = np.asarray(index)
        if value.size != 1:
            raise ValueError("One scalar amendment menu index is required")
        scalar = value.item()
        if not np.isfinite(scalar) or int(scalar) != scalar or not 0 <= scalar < self.n_actions:
            raise ValueError("Amendment menu index must be an integer in [0, 78)")
        index = int(scalar)
        raw = self.base.decode(episode, 0 if index <= 1 else index - 1)
        return np.r_[raw, int(index != 0)].astype(np.int64)

    def sample(self, episode, rng):
        branch = float(rng.random())
        if branch < .60:
            return self.decode(episode, int(rng.integers(self.n_actions)))
        config = episode.config
        if branch < .80:
            discounts = rng.choice(config.discount_levels, size=(3, 4))
        else:
            discounts = np.full((3, 4), rng.choice(config.discount_levels))
        return np.r_[encode_platform_action(discounts, np.ones(3), config), 1].astype(np.int64)


def feature_names():
    names = []
    for prefix in ("outside_energy_yuan_per_gpuh", "outside_capacity_fraction",
                   "renewable_available_mw", "renewable_curtailed_mw",
                   "outside_carbon_t_per_mwh"):
        names.extend(f"{prefix}_r{region}_b{block}" for region in range(3) for block in range(4))
    names.extend(f"site_capacity_thousand_gpu_{region}" for region in range(3))
    names.extend(f"line_mean_utilization_{line}" for line in range(3))
    names.extend(f"line_peak_utilization_{line}" for line in range(3))
    for prefix in ("owner_work_thousand_gpuh", "owner_outside_invoice_yuan_per_gpuh",
                   "owner_outside_start_hour"):
        names.extend(f"{prefix}_{owner}" for owner in range(3))
    names.extend(("outside_dc_bill_yuan_per_gpuh", "accepted_outside_profit_yuan_per_gpuh",
                  "carbon_budget_thousand_t", "outside_grid_emissions_thousand_t",
                  "generator_carbon_price_thousand_yuan_per_t", "resource_share", "net_before_share",
                  "direct_no_settlement"))
    names.extend(f"extra_discount_r{region}_b{block}" for region in range(3) for block in range(4))
    names.extend(("mean_extra_discount", "maximum_extra_discount", "amendment_open"))
    for prefix in (
        "owner_best_resource_saving_yuan_per_gpuh", "owner_mean_resource_saving_yuan_per_gpuh",
        "owner_worst_resource_saving_yuan_per_gpuh", "owner_best_extra_rebate_yuan_per_gpuh",
        "owner_mean_extra_rebate_yuan_per_gpuh", "owner_best_net_surplus_yuan_per_gpuh",
        "owner_mean_net_surplus_yuan_per_gpuh", "owner_worst_net_surplus_yuan_per_gpuh",
        "owner_best_proposed_platform_gain_yuan_per_gpuh", "owner_mean_proposed_platform_gain_yuan_per_gpuh",
        "owner_worst_proposed_platform_gain_yuan_per_gpuh", "owner_best_admitted_platform_gain_yuan_per_gpuh",
        "owner_mean_admitted_platform_gain_yuan_per_gpuh", "owner_best_buyer_gain_yuan_per_gpuh",
        "owner_mean_buyer_gain_yuan_per_gpuh", "owner_best_grid_emission_reduction_kg_per_gpuh",
        "owner_mean_grid_emission_reduction_kg_per_gpuh", "owner_physically_feasible_fraction",
        "owner_admissible_fraction", "owner_best_settlement_yuan_per_gpuh",
        "owner_mean_settlement_yuan_per_gpuh",
    ):
        names.extend(f"{prefix}_{owner}" for owner in range(3))
    names.append("unique_raw_action")
    return tuple(names)


amendment_feature_names = feature_names


@dataclass(frozen=True)
class AmendmentOptionForecast:
    option: object
    feasible: bool
    resource_saving_yuan: float
    private_pre_rebate_saving_yuan: float
    grid_emission_reduction_t: float


def _available_options(episode, job_index):
    available = getattr(episode, "option_available", None)
    if available is None:
        return (True,) * len(episode.jobs[job_index].options)
    return available[job_index]


def public_amendment_forecasts(episode):
    """One-job public OPF forecasts against the accepted outside bookings."""
    config, scenario, reference = episode.config, episode.scenario, episode.baseline_clearing
    outside_load = scenario.baseline_load_mw
    dt = config.dt_hours
    outside_dc = outside_load - scenario.background_mw
    outside_bill = float(np.sum(outside_dc * reference.lmp_yuan_per_mwh) * dt)
    original_alloc = sum((job.options[0].allocation_cu for job in episode.jobs), np.zeros_like(outside_load))
    original_transfer = sum((job.options[0].transfer_gb for job in episode.jobs),
                            np.zeros((config.n_steps, 3, 3)))
    capacity = np.asarray(config.capacity_cu_per_hour) * dt
    public = []
    for job_index, job in enumerate(episode.jobs):
        outside = job.options[0]
        outside_power = scenario.option_power_mw(outside)
        forecasts = []
        availability = _available_options(episode, job_index)
        for index, option in enumerate(job.options[1:], start=1):
            if index >= len(availability) or not availability[index] or same_execution(option, outside):
                continue
            allocation = original_alloc - outside.allocation_cu + option.allocation_cu
            transfer = original_transfer - outside.transfer_gb + option.transfer_gb
            possible = bool(np.all(allocation <= capacity + 1e-7)
                            and np.all(transfer <= config.bandwidth_gb_per_hour * dt + 1e-7))
            load = outside_load - outside_power + scenario.option_power_mw(option)
            result = episode.grid.solve(load, scenario.carbon_budget_t) if possible else None
            possible = bool(possible and result.feasible)
            if possible:
                operating_increase = option.operating_cost_yuan - outside.operating_cost_yuan
                candidate_bill = float(np.sum((load - scenario.background_mw)
                                               * result.lmp_yuan_per_mwh) * dt)
                resource = float(reference.total_generation_cost - result.total_generation_cost - operating_increase)
                private = float(outside_bill - candidate_bill - operating_increase)
                reduction = float(reference.emissions_t - result.emissions_t)
            else:
                resource = private = reduction = 0.
            forecasts.append(AmendmentOptionForecast(option, possible, resource, private, reduction))
        public.append((job, tuple(forecasts)))
    return tuple(public), original_alloc


def _reference_context(episode, allocation):
    config, scenario, reference = episode.config, episode.scenario, episode.baseline_clearing
    blocks = np.minimum(np.arange(config.n_steps) * config.n_blocks // config.n_steps, config.n_blocks - 1)
    by_block = lambda values: np.column_stack([values[blocks == block].mean(axis=0)
                                               for block in range(config.n_blocks)])
    outside_load = scenario.baseline_load_mw
    dc_power = outside_load - scenario.background_mw
    available, curtailed = np.zeros_like(outside_load), np.zeros_like(outside_load)
    spec = scenario.grid_spec
    for generator, region in enumerate(spec.generator_bus):
        if spec.generator_emissions_t_per_mwh[generator] == 0:
            available[:, region] += spec.generator_capacity_mw[:, generator]
            curtailed[:, region] += np.maximum(0, spec.generator_capacity_mw[:, generator]
                                               - reference.generation_mw[:, generator])
    intensity = nodal_carbon_intensity(spec, reference, outside_load)
    utilization = np.divide(np.abs(reference.flows_mw), spec.line_capacity_mw,
                            out=np.zeros_like(reference.flows_mw), where=spec.line_capacity_mw > 0)
    quantities = np.array([sum(job.work_cu for job in episode.jobs if job.user_id == owner)
                           for owner in range(3)])
    bills = np.asarray(episode.baseline_bills, dtype=float)
    owner_bill = [sum(bill for bill, job in zip(bills, episode.jobs) if job.user_id == owner)
                  / max(quantities[owner], 1e-12) for owner in range(3)]
    start_hour = [np.mean([job.options[0].start_step * config.dt_hours for job in episode.jobs
                          if job.user_id == owner]) if quantities[owner] > 0 else 0. for owner in range(3)]
    work = float(quantities.sum())
    dc_bill = float(np.sum(dc_power * reference.lmp_yuan_per_mwh) * config.dt_hours)
    # Cash profit follows accepted invoices and public grid/operation costs;
    # no aggregate user loss or utility is included in the platform state.
    operations = sum(job.options[0].operating_cost_yuan for job in episode.jobs)
    outside_profit = float(bills.sum() - dc_bill - operations - config.daily_fixed_compute_cost_yuan)
    context = np.r_[by_block(reference.lmp_yuan_per_mwh * np.asarray(config.energy_mwh_per_cu)).ravel(),
                    by_block(allocation / (np.asarray(config.capacity_cu_per_hour) * config.dt_hours)).ravel(),
                    by_block(available).ravel(), by_block(curtailed).ravel(), by_block(intensity).ravel(),
                    np.asarray(config.capacity_cu_per_hour) / 1000,
                    utilization.mean(axis=0), utilization.max(axis=0), quantities / 1000,
                    owner_bill, start_hour, dc_bill / max(work, 1e-12), outside_profit / max(work, 1e-12),
                    (scenario.carbon_budget_t or 0.) / 1000, reference.emissions_t / 1000,
                    spec.carbon_price_yuan_per_t / 1000, config.resource_share,
                    float(config.settlement_order == "net"), float(config.settlement_order == "direct")]
    return context, blocks


def candidate_amendment_observation(episode, catalogue=None, *, forecasts=None):
    catalogue = catalogue or AmendmentTariffCatalogue(episode.config)
    config = episode.config
    public, outside_alloc = public_amendment_forecasts(episode) if forecasts is None else forecasts
    context, blocks = _reference_context(episode, outside_alloc)
    rows, seen = [], set()
    for menu in range(catalogue.n_actions):
        raw = catalogue.decode(episode, menu)
        key = tuple(int(value) for value in raw)
        unique = key not in seen
        seen.add(key)
        opened = bool(raw[15])
        discounts = np.asarray(config.discount_levels)[raw[:12]].reshape(3, 4)
        discounts_t = discounts[:, blocks].T
        stats, counts = np.zeros((21, 3)), np.zeros(3)
        if opened:
            for job_index, (job, alternatives) in enumerate(public):
                owner, quantity = job.user_id, job.work_cu
                counts[owner] += 1
                resources, rebates, net_surplus, margins, admitted, buyer, reductions, settlements = [], [], [], [], [], [], [], []
                for forecast in alternatives:
                    if not forecast.feasible:
                        continue
                    bill = quote_bill_for_option(job, forecast.option, discounts_t,
                                                 episode.baseline_bills[job_index])
                    rebate = float(episode.baseline_bills[job_index] - bill)
                    resource = forecast.resource_saving_yuan
                    net = resource - rebate
                    tolerance = getattr(config, "admission_tolerance_yuan", 0.)
                    if config.settlement_order == "direct":
                        gain = forecast.private_pre_rebate_saving_yuan - rebate
                        qualifies = bool(gain >= -tolerance)
                    elif config.settlement_order == "net":
                        gain = config.resource_share * net
                        qualifies = bool(net >= -tolerance)
                    elif config.settlement_order == "gross":
                        gain = config.resource_share * resource - rebate
                        qualifies = bool(gain >= -tolerance and net >= -tolerance)
                    else:
                        raise ValueError("Unknown amendment settlement order")
                    buyer_gain = net - gain if qualifies else 0.
                    transfer = gain - (forecast.private_pre_rebate_saving_yuan - rebate) if qualifies else 0.
                    resources.append(resource / quantity); rebates.append(rebate / quantity)
                    net_surplus.append(net / quantity); margins.append(gain / quantity)
                    admitted.append(qualifies); buyer.append(buyer_gain / quantity)
                    reductions.append(forecast.grid_emission_reduction_t * 1000 / quantity)
                    settlements.append(transfer / quantity)
                if resources:
                    margins, admitted = np.asarray(margins), np.asarray(admitted)
                    realised = np.where(admitted, margins, 0.)
                    stats[:, owner] += [max(resources), np.mean(resources), min(resources),
                        max(rebates), np.mean(rebates), max(net_surplus), np.mean(net_surplus), min(net_surplus),
                        margins.max(), margins.mean(), margins.min(), realised.max(), realised.mean(),
                        max(buyer), np.mean(buyer), max(reductions), np.mean(reductions),
                        len(resources) / max(1, len(alternatives)), admitted.mean(),
                        max(settlements), np.mean(settlements)]
        stats /= np.maximum(counts, 1)[None, :]
        rows.append(np.r_[context, discounts.ravel(), discounts.mean(), discounts.max(),
                          float(opened), stats.ravel(), float(unique)])
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (catalogue.n_actions, len(feature_names())) or not np.isfinite(result).all():
        raise ValueError("Expected finite public forecast rows for all amendment menus")
    return result


class AmendmentTariffObserver:
    """Cache forecasts by immutable case and accepted public booking contents."""
    def __init__(self, catalogue=None, maxsize=512):
        self.catalogue = catalogue or AmendmentTariffCatalogue()
        self.maxsize = int(maxsize)
        if self.maxsize < 1:
            raise ValueError("Observer cache must hold at least one booking")
        self.feature_names = feature_names()
        self.training_max_discount = .03
        self._cache, self._forecast_cache = OrderedDict(), OrderedDict()

    @staticmethod
    def _identity(episode):
        booking = getattr(episode, "booking", None)
        original = getattr(episode, "original_scenario", getattr(booking, "scenario", episode.scenario))
        signature = (tuple(getattr(episode, "outside_selected_options", getattr(booking, "selected_options", ()))),
                     tuple(float(value) for value in episode.baseline_bills),
                     tuple(getattr(booking, "platform_action", ())))
        return original, episode.config, signature

    def _forecasts(self, episode):
        key, (reference, config, signature) = int(episode.scenario.seed), self._identity(episode)
        entry = self._forecast_cache.get(key)
        if entry is None or entry[0] is not reference or entry[1] is not config or entry[2] != signature:
            entry = (reference, config, signature, public_amendment_forecasts(episode))
            self._forecast_cache[key] = entry
            if len(self._forecast_cache) > self.maxsize:
                self._forecast_cache.popitem(last=False)
        self._forecast_cache.move_to_end(key)
        return entry[3]

    def raw(self, episode):
        key, (reference, config, signature) = int(episode.scenario.seed), self._identity(episode)
        entry = self._cache.get(key)
        if entry is None or entry[0] is not reference or entry[1] is not config or entry[2] != signature:
            values = candidate_amendment_observation(episode, self.catalogue,
                                                    forecasts=self._forecasts(episode))
            entry = (reference, config, signature, values)
            self._cache[key] = entry
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return entry[3].copy()

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
