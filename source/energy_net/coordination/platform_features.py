"""Public, action-aware features for the platform's daily tariff choice.

No private preference, user policy or candidate dispatch is evaluated here.
Potential margins use reference LMPs and hypothetical task placements; they
are forecasts for learning, not realised or screened tariff profits.
"""
from __future__ import annotations

import numpy as np

from .tariffs import TariffCatalogue


def feature_names():
    names = []
    for prefix in ("reference_energy_cost", "original_work_share", "curtailed_energy_ratio"):
        names.extend(f"{prefix}_r{r}_b{b}" for r in range(3) for b in range(4))
    names.extend(f"line_mean_utilization_{i}" for i in range(3))
    names.extend(f"line_peak_utilization_{i}" for i in range(3))
    names.extend(("carbon_price_scaled", "carbon_cap_scaled", "reference_dc_bill_ratio"))
    names.extend(f"owner_work_share_{u}" for u in range(3))
    names.extend(f"discount_r{r}_b{b}" for r in range(3) for b in range(4))
    names.extend(f"quota_r{r}" for r in range(3))
    names.extend(("mean_discount", "maximum_discount", "original_invoice_discount_exposure"))
    for prefix in ("owner_original_invoice_discount", "owner_possible_price_cut",
                   "owner_best_reference_margin", "owner_average_reference_margin",
                   "owner_profitable_offer_share", "owner_clean_fit", "owner_feasible_offer_share"):
        names.extend(f"{prefix}_{u}" for u in range(3))
    names.append("unique_raw_action")
    return tuple(names)


tariff_feature_names = feature_names


def _public_context(episode):
    cfg, scenario = episode.config, episode.scenario
    blocks = np.minimum(np.arange(cfg.n_steps) * cfg.n_blocks // cfg.n_steps, cfg.n_blocks - 1)
    baseline = episode.baseline_clearing
    prices = baseline.lmp_yuan_per_mwh
    energy = np.asarray(cfg.energy_mwh_per_cu)
    allocation = sum((job.options[0].allocation_cu for job in scenario.jobs), np.zeros((48, 3)))
    total_work = scenario.total_work_cu
    revenue = total_work * cfg.base_price_yuan_per_cu
    dc_energy = float(np.sum((scenario.baseline_load_mw - scenario.background_mw) * cfg.dt_hours))
    clean_spare = np.zeros((48, 3))
    spec = scenario.grid_spec
    for generator in range(spec.n_generators):
        if spec.generator_emissions_t_per_mwh[generator] == 0:
            clean_spare[:, spec.generator_bus[generator]] += np.maximum(
                0, spec.generator_capacity_mw[:, generator] - baseline.generation_mw[:, generator])
    costs = np.column_stack([
        (prices[blocks == b] * energy).mean(axis=0) / cfg.base_price_yuan_per_cu
        for b in range(4)])
    work = np.column_stack([allocation[blocks == b].sum(axis=0) / total_work for b in range(4)])
    spare = np.column_stack([clean_spare[blocks == b].sum(axis=0) * cfg.dt_hours
                            / max(dc_energy, 1e-9) for b in range(4)])
    utilization = np.abs(baseline.flows_mw) / spec.line_capacity_mw
    dc_bill = np.sum(prices * (scenario.baseline_load_mw - scenario.background_mw)) * cfg.dt_hours
    context = np.r_[costs.ravel(), work.ravel(), spare.ravel(),
                    utilization.mean(axis=0), utilization.max(axis=0),
                    baseline.carbon_price_yuan_per_t / 1000,
                    scenario.carbon_budget_t / 2000, dc_bill / revenue,
                    [sum(j.work_cu for j in scenario.jobs if j.user_id == u) / total_work
                     for u in range(3)]]
    return context, allocation, clean_spare, blocks, revenue


def candidate_tariff_observation(episode, catalogue=None):
    """Return [55, D] float32 features without changing published contracts.

    The last column masks only exactly identical raw 15-component market
    actions. It never screens a price using user utility or realised profit.
    Feature rows are invariant to beta, kappa and option user-loss values.
    Because these features depend only on immutable public scenario inputs,
    callers may cache them by scenario before collecting platform transitions.
    """
    catalogue = catalogue or TariffCatalogue(episode.config)
    cfg, scenario = episode.config, episode.scenario
    context, original_alloc, clean_spare, blocks, revenue = _public_context(episode)
    lmp = episode.baseline_clearing.lmp_yuan_per_mwh
    capacity = np.asarray(cfg.capacity_cu_per_hour) * cfg.dt_hours
    public = []
    for job in scenario.jobs:
        original = job.options[0]
        original_power = scenario.option_power_mw(original)
        original_bill = float(np.sum(original_power * lmp) * cfg.dt_hours)
        options = []
        for option in job.options[1:]:
            allocation = original_alloc - original.allocation_cu + option.allocation_cu
            physical = (np.all(allocation <= capacity + 1e-7)
                        and np.all(option.transfer_gb <= cfg.bandwidth_gb_per_hour * cfg.dt_hours + 1e-7))
            power = scenario.option_power_mw(option)
            forecast_saving = original_bill - float(np.sum(power * lmp) * cfg.dt_hours)
            incremental_power = np.maximum(0, power - original_power)
            incremental_energy = float(incremental_power.sum() * cfg.dt_hours)
            clean_fit = (float(np.minimum(incremental_power, clean_spare).sum() * cfg.dt_hours)
                         / incremental_energy if incremental_energy > 1e-12 else 0.0)
            opex_increase = option.operating_cost_yuan - original.operating_cost_yuan
            options.append((option, allocation, physical, forecast_saving - opex_increase, clean_fit))
        public.append((job, options))

    rows, seen = [], set()
    for index in range(catalogue.n_actions):
        raw = catalogue.decode(episode, index)
        key = tuple(int(v) for v in raw)
        unique = key not in seen
        seen.add(key)
        discounts = np.asarray(cfg.discount_levels)[raw[:12]].reshape(3, 4)
        quotas = np.asarray(cfg.quota_levels)[raw[12:]]
        tariff_t = (cfg.base_price_yuan_per_cu * (1 - discounts))[:, blocks].T
        quota_limit = np.maximum(original_alloc, capacity * quotas)
        stats = np.zeros((7, 3))
        owner_count = np.zeros(3)
        exposure = 0.0
        for job, options in public:
            owner = job.user_id
            owner_count[owner] += 1
            base_invoice = job.work_cu * cfg.base_price_yuan_per_cu
            windfall = base_invoice - float(np.sum(job.options[0].allocation_cu * tariff_t))
            exposure += windfall
            stats[0, owner] += windfall / revenue
            cuts, margins, fits = [], [], []
            for option, allocation, physical, cost_saving, clean_fit in options:
                if not physical or np.any(allocation > quota_limit + 1e-7):
                    continue
                cut = base_invoice - float(np.sum(option.allocation_cu * tariff_t))
                if cut <= 1e-9:
                    continue
                cuts.append(cut)
                margins.append(cost_saving - cut)
                fits.append(clean_fit)
            if cuts:
                stats[1, owner] += max(cuts) / revenue
                stats[2, owner] += max(margins) / revenue
                stats[3, owner] += float(np.mean(margins)) / revenue
                stats[4, owner] += max(margins) > 0
                stats[5, owner] += max(fits)
                stats[6, owner] += len(cuts) / max(1, len(options))
        stats[4:] /= np.maximum(owner_count, 1)[None, :]
        proposal = np.r_[discounts.ravel(), quotas, discounts.mean(), discounts.max(),
                         exposure / revenue, stats.ravel(), float(unique)]
        rows.append(np.r_[context, proposal])
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (catalogue.n_actions, len(feature_names())) or not np.isfinite(result).all():
        raise ValueError("Invalid public candidate-tariff features.")
    return result
