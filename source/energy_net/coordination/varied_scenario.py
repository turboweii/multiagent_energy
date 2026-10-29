"""Regional weather and task origins with fixed hardware and generation costs.

This scenario is separate from the original GPU benchmark. Its physical day
is sampled before any quote, user response or reward. A declared infeasible
day raises an error; weather and capacities are never redrawn after clearing.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .amendment_market import AmendmentConfig
from .capped_market import CAP_TOLERANCE_T
from .gpu_scenario import GpuMarketConfig, make_gpu_scenario
from .grid import GridClearing
from .scenario import Scenario, _original_shapes


@dataclass(frozen=True)
class VariedMarketConfig(AmendmentConfig):
    """Fixed regional installations and predeclared weather distributions."""

    solar_capacity_mw: tuple = (12., 42., 38.)
    wind_capacity_mw: tuple = (28., 12., 16.)
    cloud_factor_bounds: tuple = (.25, 1.)
    wind_daily_capacity_factor_bounds: tuple = (.15, .65)
    wind_intraday_amplitude: float = .18
    background_daily_multiplier_bounds: tuple = (.85, 1.15)
    background_peak_shift_steps: int = 6
    release_jitter_steps: tuple = (-1, 0, 1)

    def __post_init__(self):
        super().__post_init__()
        for name in ("solar_capacity_mw", "wind_capacity_mw"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{name} must contain three finite nonnegative installations")
        for name in ("cloud_factor_bounds", "wind_daily_capacity_factor_bounds"):
            values = np.asarray(getattr(self, name), dtype=float)
            if (values.shape != (2,) or not np.isfinite(values).all()
                    or not 0 <= values[0] <= values[1] <= 1):
                raise ValueError(f"{name} must be an ordered pair within [0,1]")
        background = np.asarray(self.background_daily_multiplier_bounds, dtype=float)
        if (background.shape != (2,) or not np.isfinite(background).all()
                or not 0 < background[0] <= background[1]):
            raise ValueError("Background daily multipliers must be finite, ordered and positive")
        if not np.isfinite(self.wind_intraday_amplitude) or not 0 <= self.wind_intraday_amplitude <= 1:
            raise ValueError("Wind intraday amplitude must lie in [0,1]")
        if (int(self.background_peak_shift_steps) != self.background_peak_shift_steps
                or self.background_peak_shift_steps < 0):
            raise ValueError("Background peak shift must be a nonnegative integer")
        jitter = np.asarray(self.release_jitter_steps)
        if (jitter.ndim != 1 or not jitter.size or not np.isfinite(jitter).all()
                or np.any(jitter != jitter.astype(int))):
            raise ValueError("Provide a nonempty sequence of integer release shifts")


def _smooth_blocks(values, n_steps):
    """Continuous, periodic linear interpolation between six-hour controls."""
    controls = np.asarray(values, dtype=float)
    coordinates = np.linspace(0, n_steps, len(controls) + 1)
    return np.interp(np.arange(n_steps), coordinates, np.r_[controls, controls[0]])


def _physical_options(config):
    """Legacy settlement normalization can retain or omit the extra fields."""
    defaults = VariedMarketConfig()
    names = ("solar_capacity_mw", "wind_capacity_mw", "cloud_factor_bounds",
             "wind_daily_capacity_factor_bounds", "wind_intraday_amplitude",
             "background_daily_multiplier_bounds", "background_peak_shift_steps",
             "release_jitter_steps")
    return {name: getattr(config, name, getattr(defaults, name)) for name in names}


def _independent_background(rng, config, physics):
    shapes = _original_shapes()
    indices = rng.integers(len(shapes), size=3)
    shift_limit = int(physics["background_peak_shift_steps"])
    shifts = rng.integers(-shift_limit, shift_limit + 1, size=3)
    multipliers = rng.uniform(*physics["background_daily_multiplier_bounds"], size=3)
    background = np.column_stack([
        base * (.72 + .28 * np.roll(np.clip(shapes[index], .4, 2.), shift)) * multiplier
        for base, index, shift, multiplier in zip((22., 20., 18.), indices, shifts, multipliers)])
    metadata = dict(background_shape_indices=indices.tolist(),
                    background_peak_shifts_steps=shifts.tolist(),
                    background_daily_multipliers=multipliers.tolist())
    return background, metadata


def _regional_weather(rng, config, physics):
    hour = np.arange(config.n_steps) * config.dt_hours
    clear_sky = np.maximum(0., np.sin(np.pi * (hour - 6.) / 12.))
    daily_cloud = rng.uniform(*physics["cloud_factor_bounds"], size=3)
    block_cloud = rng.uniform(*physics["cloud_factor_bounds"], size=(3, 4))
    cloud = np.column_stack([.5 * daily_cloud[region] + .5 * _smooth_blocks(block_cloud[region], config.n_steps)
                             for region in range(3)])
    solar = clear_sky[:, None] * cloud * np.asarray(physics["solar_capacity_mw"])
    wind_means = rng.uniform(*physics["wind_daily_capacity_factor_bounds"], size=3)
    wind_cf = np.zeros_like(cloud)
    for region, mean in enumerate(wind_means):
        fluctuation = _smooth_blocks(rng.uniform(-1., 1., size=4), config.n_steps)
        fluctuation -= fluctuation.mean()
        magnitude = max(float(np.max(np.abs(fluctuation))), 1e-12)
        amplitude = min(physics["wind_intraday_amplitude"], .95 * mean, .95 * (1. - mean))
        wind_cf[:, region] = mean + amplitude * fluctuation / magnitude
    wind = wind_cf * np.asarray(physics["wind_capacity_mw"])
    metadata = dict(cloud_daily_factors=daily_cloud.tolist(),
                    cloud_block_factors=block_cloud.tolist(),
                    effective_cloud_factor_min=cloud.min(axis=0).tolist(),
                    effective_cloud_factor_max=cloud.max(axis=0).tolist(),
                    wind_daily_capacity_factors=wind_means.tolist(),
                    wind_realized_mean_capacity_factors=wind_cf.mean(axis=0).tolist(),
                    wind_min_capacity_factors=wind_cf.min(axis=0).tolist(),
                    wind_max_capacity_factors=wind_cf.max(axis=0).tolist(),
                    solar_available_mw=solar.tolist(), wind_available_mw=wind.tolist())
    return solar, wind, metadata


def _relocate_jobs(rng, original_jobs, config, physics):
    duration = int(config.batch_duration_hours / config.dt_hours)
    valid_jitter = tuple(int(shift) for shift in physics["release_jitter_steps"]
                        if min(config.batch_release_steps) + shift >= 0
                        and max(config.batch_release_steps) + shift + duration + 6 <= config.n_steps)
    if not valid_jitter:
        raise ValueError("Declared release shifts leave no complete execution and delay window")
    shift = int(rng.choice(valid_jitter))
    permutations = {release: rng.permutation(3) for release in config.batch_release_steps}
    jobs = []
    for job in original_jobs:
        permutation = permutations[job.release_step]
        inverse = np.argsort(permutation)
        options = []
        for option in job.options:
            allocation = np.roll(option.allocation_cu[:, inverse], shift, axis=0).copy()
            extra = np.roll(option.extra_energy_mwh[:, inverse], shift, axis=0).copy()
            transfer = np.roll(option.transfer_gb[:, inverse][:, :, inverse], shift, axis=0).copy()
            options.append(replace(option, region=int(permutation[option.region]),
                start_step=option.start_step + shift, end_step=option.end_step + shift,
                allocation_cu=allocation, extra_energy_mwh=extra, transfer_gb=transfer))
        jobs.append(replace(job, home_region=int(permutation[job.home_region]),
                            release_step=job.release_step + shift,
                            deadline_step=job.deadline_step + shift, options=tuple(options)))
    metadata = dict(common_release_jitter_steps=shift,
                    release_jitter_sampling_support=list(valid_jitter),
                    home_permutations_by_original_release={str(release): permutation.tolist()
                                                           for release, permutation in permutations.items()})
    return tuple(jobs), metadata


def make_varied_scenario(seed=0, config=None):
    """Generate one varied physical day and its strategy-independent cap.

    Passing a legacy-normalized GPU/Flex config keeps its economic rules.
    Extra weather fields default to the same VariedMarketConfig values when
    that legacy config does not carry them. A subclass-preserving dataclass
    normalization retains explicitly supplied varied-physics parameters.
    """
    config = config or VariedMarketConfig()
    if not isinstance(config, GpuMarketConfig):
        raise TypeError("The varied market requires a GPU config or subclass")
    seed = int(seed)
    physics = _physical_options(config)
    # Reuse the original job dimensions and private preferences unchanged.
    original = make_gpu_scenario(seed, config)
    sequences = np.random.SeedSequence([seed, 5260429]).spawn(3)
    background, load_metadata = _independent_background(np.random.default_rng(sequences[0]), config, physics)
    solar, wind, weather_metadata = _regional_weather(np.random.default_rng(sequences[1]), config, physics)
    jobs, placement_metadata = _relocate_jobs(np.random.default_rng(sequences[2]), original.jobs, config, physics)
    renewable = solar + wind
    # Original coal0, gas1 and backup2 retain their costs, capacities and buses.
    # The old single renewable generator is replaced by fixed regional fleets.
    thermal = np.flatnonzero(original.grid_spec.generator_emissions_t_per_mwh > 0)
    spec = replace(original.grid_spec,
        generator_bus=np.r_[original.grid_spec.generator_bus[thermal], np.arange(3)],
        generator_capacity_mw=np.column_stack([original.grid_spec.generator_capacity_mw[:, thermal], renewable]),
        generator_cost_yuan_per_mwh=np.r_[original.grid_spec.generator_cost_yuan_per_mwh[thermal], [15.] * 3],
        generator_emissions_t_per_mwh=np.r_[original.grid_spec.generator_emissions_t_per_mwh[thermal], np.zeros(3)])
    scenario = Scenario(config, seed, background, renewable.sum(axis=1), spec, None, jobs)
    grid = GridClearing(spec)
    reference = grid.solve(scenario.baseline_load_mw, None)
    if not reference.feasible:
        raise ValueError(f"Varied physical day {seed} has an infeasible original no-shift dispatch: {reference.status}")
    scenario.carbon_budget_t = float(reference.emissions_t + CAP_TOLERANCE_T)
    constrained = grid.solve(scenario.baseline_load_mw, scenario.carbon_budget_t)
    if not constrained.feasible:
        raise ValueError(f"Varied physical day {seed} is infeasible at its original common carbon cap")
    background_only = grid.solve(background, scenario.carbon_budget_t)
    if not background_only.feasible:
        raise ValueError(f"Varied physical day {seed} has an infeasible same-cap background dispatch")
    allocation = sum((job.options[0].allocation_cu for job in jobs), np.zeros_like(background))
    capacity = np.asarray(config.capacity_cu_per_hour) * config.dt_hours
    if np.any(allocation > capacity + 1e-7):
        raise ValueError(f"Varied physical day {seed} exceeds the installed no-shift compute capacity")
    remote_counts = []
    for job in jobs:
        remote_counts.append(sum(option.migrated
            and np.all(allocation - job.options[0].allocation_cu + option.allocation_cu <= capacity + 1e-7)
            and np.all(option.transfer_gb <= config.bandwidth_gb_per_hour * config.dt_hours + 1e-7)
            for option in job.options))
    dc_energy = float((scenario.baseline_load_mw - background).sum() * config.dt_hours)
    background_energy = float(background.sum() * config.dt_hours)
    scenario.calibration = dict(original.calibration,
        scenario_family="varied_regional_weather_and_origins_v1",
        scenario_sampling="independent seeded input streams before prices; no rejection sampling or reward-based selection",
        regional_solar_capacity_mw=list(physics["solar_capacity_mw"]),
        regional_wind_capacity_mw=list(physics["wind_capacity_mw"]),
        cloud_factor_combination="equal mixture of daily factor and continuous periodic block factor; effective value stays in declared bounds",
        cloud_factor_bounds=list(physics["cloud_factor_bounds"]),
        wind_daily_capacity_factor_bounds=list(physics["wind_daily_capacity_factor_bounds"]),
        background_daily_multiplier_bounds=list(physics["background_daily_multiplier_bounds"]),
        fixed_physics="original PUE, thermal plant buses/capacities/fuel/emission coefficients, network and quote-independent compute capacity",
        task_origins="independent per-batch permutation maps user home and all option/transfer locations; user IDs and preferences stay attached to original user",
        carbon_policy="exogenous generation carbon cost and common original no-shift E0 allowance",
        common_carbon_cap_t=scenario.carbon_budget_t,
        common_carbon_cap_source="varied original no-shift workload, before legacy quotes or amendments",
        original_unconstrained_emissions_t=float(reference.emissions_t),
        common_carbon_cap_numerical_margin_t=CAP_TOLERANCE_T,
        original_unconstrained_generation_cost_yuan=float(reference.total_generation_cost),
        original_same_cap_generation_cost_yuan=float(constrained.total_generation_cost),
        background_only_generation_cost_yuan=float(background_only.total_generation_cost),
        background_only_carbon_budget_t=scenario.carbon_budget_t,
        carbon_qualification="identical physical OPF cap in legacy and amendment stages",
        dc_energy_mwh=dc_energy, background_energy_mwh=background_energy,
        dc_energy_share=dc_energy / (dc_energy + background_energy),
        physically_feasible_remote_options_per_job=[int(value) for value in remote_counts],
        load_shape="independent regional original EnergyNet normalized household shapes, phases and daily multipliers",
        **load_metadata, **weather_metadata, **placement_metadata)
    return scenario
