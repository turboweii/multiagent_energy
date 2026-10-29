"""Signal-independent GPU tariff choices and public information ablations.

The same 77 raw menus are available with and without carbon observations.
Templates never rank electricity or carbon signals, and observations contain
no private delay/migration preferences or simulated user responses. Candidate
margins and footprints use the original-contract reference dispatch throughout;
they are forecasts, not realised returns or feasibility certificates.
"""
from __future__ import annotations

from collections import OrderedDict
from itertools import product

import numpy as np

from .carbon import consumption_emissions_t, nodal_carbon_intensity
from .market import encode_platform_action


class FixedGpuTariffCatalogue:
    """Original plus 19 fixed masks times four discount magnitudes.

    Four masks select one time block, three select one region, and twelve
    select one region/block cell. Quotas are always the full physical capacity.
    Every magnitude must exist exactly in the episode configuration; rounding
    a small discount to a legacy zero-discount level is rejected.
    """

    patterns = (*(f"time_{block}" for block in range(4)),
                *(f"region_{region}" for region in range(3)),
                *(f"cell_{region}_{block}" for region in range(3) for block in range(4)))
    amplitudes = (.0025, .005, .01, .02)
    quotas = (1.0,)
    n_actions = 77

    def __init__(self, config=None):
        self.config = config
        self.entries = (None, *((pattern, amplitude, 1.)
                                for pattern, amplitude in product(self.patterns, self.amplitudes)))
        self.labels = ("original", *(f"{pattern}_discount_{amplitude:.4f}_quota_1.00"
                                     for pattern, amplitude, _ in self.entries[1:]))
        if config is not None:
            self._validate(config)

    @classmethod
    def _validate(cls, config):
        if (config.n_regions, config.n_blocks) != (3, 4):
            raise ValueError("GPU tariff masks require three regions and four time blocks")
        if any(not np.any(np.isclose(amplitude, config.discount_levels, rtol=0, atol=1e-12))
               for amplitude in (0., *cls.amplitudes)):
            raise ValueError("GPU discount levels must contain zero and all exact catalogue magnitudes")
        if not np.any(np.isclose(1., config.quota_levels, rtol=0, atol=1e-12)):
            raise ValueError("GPU tariff catalogue requires full quota 1.0")

    def decode(self, episode, index):
        value = np.asarray(index)
        if value.size != 1:
            raise ValueError("One scalar GPU tariff index is required")
        scalar = value.item()
        if not np.isfinite(scalar) or int(scalar) != scalar or not 0 <= scalar < self.n_actions:
            raise ValueError("GPU tariff index must be an integer in [0, 77)")
        config = episode.config
        self._validate(config)
        index = int(scalar)
        if index == 0:
            return encode_platform_action(np.zeros((3, 4)), np.ones(3), config)
        pattern, amplitude, _ = self.entries[index]
        mask = np.zeros((3, 4))
        parts = pattern.split("_")
        if parts[0] == "time":
            mask[:, int(parts[1])] = 1
        elif parts[0] == "region":
            mask[int(parts[1]), :] = 1
        else:
            mask[int(parts[1]), int(parts[2])] = 1
        return encode_platform_action(amplitude * mask, np.ones(3), config)

    def sample(self, episode, rng):
        """Mix catalogue, independent-cell and uniform public training menus."""
        config = episode.config
        self._validate(config)
        branch = rng.random()
        if branch < .60:
            return self.decode(episode, int(rng.integers(self.n_actions)))
        if branch < .80:
            discounts = rng.choice(config.discount_levels, size=(3, 4))
        else:
            discounts = np.full((3, 4), rng.choice(config.discount_levels))
        return encode_platform_action(discounts, np.ones(3), config)


def feature_names():
    names = []
    for prefix in ("reference_energy_yuan_per_gpuh", "original_capacity_fraction",
                   "renewable_available_mw", "renewable_curtailed_mw"):
        names.extend(f"{prefix}_r{r}_b{b}" for r in range(3) for b in range(4))
    names.extend(f"site_capacity_gpu_{r}" for r in range(3))
    names.extend(f"line_mean_utilization_{line}" for line in range(3))
    names.extend(f"line_peak_utilization_{line}" for line in range(3))
    names.extend(f"owner_work_thousand_gpuh_{owner}" for owner in range(3))
    names.append("reference_dc_electricity_yuan_per_gpuh")
    names.extend(f"carbon_reference_t_per_mwh_r{r}_b{b}" for r in range(3) for b in range(4))
    names.extend(("carbon_external_price_yuan_per_t", "carbon_internal_value_yuan_per_t",
                  "carbon_reference_dc_kg_per_gpuh"))
    names.extend(f"discount_r{r}_b{b}" for r in range(3) for b in range(4))
    names.extend(("mean_discount", "maximum_discount"))
    for prefix in ("owner_original_windfall_yuan_per_gpuh", "owner_best_price_cut_yuan_per_gpuh",
                   "owner_best_forecast_margin_yuan_per_gpuh", "owner_mean_forecast_margin_yuan_per_gpuh",
                   "owner_worst_forecast_margin_yuan_per_gpuh", "owner_best_energy_saving_yuan_per_gpuh",
                   "owner_mean_energy_saving_yuan_per_gpuh", "owner_profitable_offer_fraction",
                   "owner_discounted_offer_fraction", "carbon_owner_best_saving_kg_per_gpuh",
                   "carbon_owner_mean_saving_kg_per_gpuh", "carbon_owner_best_internal_gain_yuan_per_gpuh",
                   "carbon_owner_best_adjusted_margin_yuan_per_gpuh",
                   "carbon_owner_mean_adjusted_margin_yuan_per_gpuh",
                   "carbon_owner_worst_adjusted_margin_yuan_per_gpuh"):
        names.extend(f"{prefix}_{owner}" for owner in range(3))
    names.append("unique_raw_action")
    return tuple(names)


gpu_tariff_feature_names = feature_names


def _internal_value(episode):
    for source in (episode, episode.config):
        for name in ("internal_carbon_value_yuan_per_t", "internal_carbon_price_yuan_per_t"):
            if hasattr(source, name):
                return float(getattr(source, name))
    return 0.


def candidate_gpu_tariff_observation(episode, catalogue=None, *, carbon_aware=True):
    """Build equal-shape public candidate rows; carbon columns alone are ablated."""
    catalogue = catalogue or FixedGpuTariffCatalogue(episode.config)
    config, scenario = episode.config, episode.scenario
    reference, spec = episode.baseline_clearing, scenario.grid_spec
    blocks = np.minimum(np.arange(config.n_steps) * config.n_blocks // config.n_steps,
                        config.n_blocks - 1)
    lmp = reference.lmp_yuan_per_mwh
    physical = np.asarray(config.capacity_cu_per_hour) * config.dt_hours
    energy = np.asarray(config.energy_mwh_per_cu)
    original_alloc = sum((job.options[0].allocation_cu for job in scenario.jobs),
                         np.zeros((config.n_steps, config.n_regions)))
    work = scenario.total_work_cu
    dc_power = scenario.baseline_load_mw - scenario.background_mw
    clean_available, clean_spare = np.zeros_like(dc_power), np.zeros_like(dc_power)
    for generator, region in enumerate(spec.generator_bus):
        if spec.generator_emissions_t_per_mwh[generator] == 0:
            clean_available[:, region] += spec.generator_capacity_mw[:, generator]
            clean_spare[:, region] += np.maximum(0, spec.generator_capacity_mw[:, generator]
                                                 - reference.generation_mw[:, generator])
    by_block = lambda values: np.column_stack([values[blocks == block].mean(axis=0)
                                               for block in range(config.n_blocks)])
    utilization = np.divide(np.abs(reference.flows_mw), spec.line_capacity_mw,
                            out=np.zeros_like(reference.flows_mw), where=spec.line_capacity_mw > 0)
    context = np.r_[by_block(lmp * energy).ravel(), by_block(original_alloc / physical).ravel(),
                    by_block(clean_available).ravel(), by_block(clean_spare).ravel(),
                    config.capacity_cu_per_hour, utilization.mean(axis=0), utilization.max(axis=0),
                    [sum(job.work_cu for job in scenario.jobs if job.user_id == owner) / 1000
                     for owner in range(3)], np.sum(dc_power * lmp) * config.dt_hours / work]
    intensity = (nodal_carbon_intensity(spec, reference, scenario.baseline_load_mw)
                 if carbon_aware else np.zeros_like(dc_power))
    internal_value = _internal_value(episode) if carbon_aware else 0.
    carbon_context = np.r_[by_block(intensity).ravel(),
                           getattr(spec, "carbon_price_yuan_per_t", 0.) if carbon_aware else 0.,
                           internal_value,
                           consumption_emissions_t(dc_power, intensity, config.dt_hours) * 1000 / work]
    public = []
    for job in scenario.jobs:
        original = job.options[0]
        original_power = scenario.option_power_mw(original)
        options = []
        for option in job.options[1:]:
            allocation = original_alloc - original.allocation_cu + option.allocation_cu
            physically_possible = (np.all(allocation <= physical + 1e-7)
                                   and np.all(option.transfer_gb <= config.bandwidth_gb_per_hour
                                              * config.dt_hours + 1e-7))
            power_change = original_power - scenario.option_power_mw(option)
            energy_saving = float(np.sum(power_change * lmp) * config.dt_hours)
            carbon_saving = float(np.sum(power_change * intensity) * config.dt_hours)
            operating_increase = option.operating_cost_yuan - original.operating_cost_yuan
            options.append((option, allocation, physically_possible, energy_saving,
                            carbon_saving, operating_increase))
        public.append((job, options))
    rows, seen = [], set()
    for index in range(catalogue.n_actions):
        raw = catalogue.decode(episode, index)
        key = tuple(int(value) for value in raw)
        unique = key not in seen
        seen.add(key)
        discounts = np.asarray(config.discount_levels)[raw[:12]].reshape(3, 4)
        quotas = np.asarray(config.quota_levels)[raw[12:]]
        quota_limit = np.maximum(original_alloc, physical * quotas)
        tariff = (config.base_price_yuan_per_cu * (1 - discounts))[:, blocks].T
        stats, owner_count = np.zeros((15, 3)), np.zeros(3)
        for job, options in public:
            owner, quantity = job.user_id, job.work_cu
            owner_count[owner] += 1
            original_invoice = quantity * config.base_price_yuan_per_cu
            stats[0, owner] += (original_invoice - np.sum(job.options[0].allocation_cu * tariff)) / quantity
            cuts, margins, savings, carbon_savings = [], [], [], []
            for option, allocation, possible, energy_saving, carbon_saving, opex_increase in options:
                if not possible or np.any(allocation > quota_limit + 1e-7):
                    continue
                cut = original_invoice - float(np.sum(option.allocation_cu * tariff))
                if cut <= 1e-9:
                    continue
                cuts.append(cut / quantity)
                margins.append((energy_saving - opex_increase - cut) / quantity)
                savings.append(energy_saving / quantity)
                carbon_savings.append(carbon_saving / quantity)
            if cuts:
                margins, carbon_savings = np.asarray(margins), np.asarray(carbon_savings)
                stats[1:9, owner] += [max(cuts), margins.max(), margins.mean(), margins.min(),
                                      max(savings), np.mean(savings), np.mean(margins > 0),
                                      len(cuts) / max(1, len(options))]
                if carbon_aware:
                    adjusted = margins + internal_value * carbon_savings
                    stats[9:, owner] += [carbon_savings.max() * 1000, carbon_savings.mean() * 1000,
                                         internal_value * carbon_savings.max(), adjusted.max(),
                                         adjusted.mean(), adjusted.min()]
        stats /= np.maximum(owner_count, 1)[None, :]
        proposal = np.r_[discounts.ravel(), discounts.mean(), discounts.max(), stats.ravel(), float(unique)]
        rows.append(np.r_[context, carbon_context, proposal])
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (catalogue.n_actions, len(feature_names())) or not np.all(np.isfinite(result)):
        raise ValueError("GPU tariff observations need one finite feature row per fixed menu")
    return result


class GpuTariffObserver:
    """Cache immutable scenario inputs and fit normalization on training seeds."""

    def __init__(self, catalogue=None, carbon_aware=True, maxsize=512):
        self.catalogue = catalogue or FixedGpuTariffCatalogue()
        self.carbon_aware = bool(carbon_aware)
        self.maxsize = int(maxsize)
        self.training_max_discount = .03
        self.feature_names = feature_names()
        self.carbon_feature_indices = tuple(index for index, name in enumerate(self.feature_names)
                                            if name.startswith("carbon_"))
        self._cache = OrderedDict()

    def raw(self, episode):
        key = (int(episode.scenario.seed), _internal_value(episode))
        entry = self._cache.get(key)
        if entry is None or entry[0] is not episode.scenario:
            entry = (episode.scenario, candidate_gpu_tariff_observation(
                episode, self.catalogue, carbon_aware=self.carbon_aware))
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
                    values[index, -1] = 0
        return values

    def fit_normalization(self, episode_factory, seeds):
        rows = np.concatenate([self.raw(episode_factory(int(seed)))[:, :-1] for seed in seeds], axis=0)
        mean = rows.mean(axis=0, dtype=np.float64)
        scale = np.maximum(rows.std(axis=0, dtype=np.float64), .01)
        return mean.astype(np.float32), scale.astype(np.float32)
