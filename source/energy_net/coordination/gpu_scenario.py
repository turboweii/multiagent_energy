"""GPU-hour scenario with explicitly separated hardware and study assumptions.

One work unit is one H100-equivalent GPU-hour on homogeneous eight-GPU
nodes.  NVIDIA specifies eight H100s and 10.2 kW maximum for a DGX H100;
Latif et al. measured up to 8.4 kW on an eight-H100 HGX training node.
The study's 8.0 kW active and 0.8 kW idle node powers are assumptions,
not claimed measurements.  PUE, yuan prices, user preferences, checkpoint
sizes, and hardware-capacity costs are also explicit study assumptions.

Sources accessed 2026-10-29:
https://docs.nvidia.com/dgx-superpod/design-guides/dgx-superpod-data-center-design-h100/latest/planning.html
https://arxiv.org/abs/2412.08602
https://datacenters.google/efficiency/
https://lambda.ai/instances

Lambda lists USD 3.99/GPU-hour for an eight-H100 instance.  This only
anchors the order of magnitude: 18 yuan/GPU-hour below is a synthetic
study price, not that quotation converted using an implicit exchange rate.
The carbon price is an exogenous generation-side cost.  This scenario
does not impose the original scenario's always-binding system-wide cap.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .grid import GridClearing, default_grid
from .scenario import ExecutionOption, Job, MarketConfig, Scenario, _original_shapes


@dataclass(frozen=True)
class GpuMarketConfig(MarketConfig):
    """A reproducible batch-compute scenario, not a fitted production trace.

    ``site_scale`` scales installed nodes and batch parallelism together.
    Default capacity accommodates two full batches per site but not three;
    reservations therefore retain meaningful migration opportunities.
    Report measured DC energy share rather than labelling scale factors as
    exact percentages of regional load.  The 0.1 GB/GPU-hour checkpoint
    assumption is light-data computing; use 2.0 for a heavy-data sensitivity.
    """

    site_scale: float = 1.0
    nodes_per_site: int = 768
    gpus_per_node: int = 8
    node_active_it_kw: float = 8.0
    node_idle_it_kw: float = 0.8
    pue: tuple = (1.15, 1.25, 1.35)
    batch_parallel_gpu_range: tuple = (2048, 2816)
    batch_duration_hours: float = 4.0
    batch_release_steps: tuple = (1, 13, 25, 33)
    base_price_yuan_per_cu: float = 18.0
    discount_levels: tuple = (0.0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03)
    beta_yuan_per_cu_hour: tuple = (0.01, 0.04, 0.12)
    migration_loss_yuan_per_cu: tuple = (0.005, 0.02, 0.05)
    compute_opex_yuan_per_cu: float = 0.05
    transfer_gb_per_cu: float = 0.1
    # 400 Gbit/s directional WAN at an assumed 70% payload efficiency.
    bandwidth_gb_per_hour: float = 126000.0
    transfer_mwh_per_gb: float = 0.00001
    transfer_cost_yuan_per_gb: float = 0.01
    carbon_price_yuan_per_t: float = 100.0
    internal_carbon_value_yuan_per_t: float = 100.0
    fixed_compute_cost_yuan_per_gpu_capacity_hour: float = 5.0

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.site_scale) or self.site_scale <= 0:
            raise ValueError("site_scale must be finite and positive")
        if (self.nodes_per_site < 1 or int(self.nodes_per_site) != self.nodes_per_site
                or self.gpus_per_node != 8):
            raise ValueError("provide a positive integer number of eight-GPU nodes")
        if not 0 <= self.node_idle_it_kw < self.node_active_it_kw <= 10.2:
            raise ValueError("node powers must satisfy 0 <= idle < active <= 10.2 kW")
        if len(self.pue) != 3 or not np.all(np.isfinite(self.pue)) or np.any(np.asarray(self.pue) < 1):
            raise ValueError("provide three finite PUE values of at least one")
        if (not np.isfinite(self.carbon_price_yuan_per_t)
                or self.carbon_price_yuan_per_t < 0):
            raise ValueError("carbon_price_yuan_per_t must be finite and nonnegative")
        if (not np.isfinite(self.internal_carbon_value_yuan_per_t)
                or self.internal_carbon_value_yuan_per_t < 0):
            raise ValueError("internal_carbon_value_yuan_per_t must be finite and nonnegative")
        if (self.transfer_gb_per_cu < 0 or self.bandwidth_gb_per_hour <= 0
                or self.fixed_compute_cost_yuan_per_gpu_capacity_hour < 0):
            raise ValueError("transfer and fixed-capacity assumptions must be nonnegative")
        if (len(self.batch_parallel_gpu_range) != 2
                or not 0 < self.batch_parallel_gpu_range[0] <= self.batch_parallel_gpu_range[1]):
            raise ValueError("batch_parallel_gpu_range must be an ordered positive pair")
        duration_steps = self.batch_duration_hours / self.dt_hours
        if not duration_steps.is_integer() or duration_steps < 1:
            raise ValueError("batch duration must be a positive multiple of the time step")
        if len(self.batch_release_steps) != self.jobs_per_user:
            raise ValueError("provide one release step per batch for each user")
        if (min(self.batch_release_steps) < 0
                or max(self.batch_release_steps) + duration_steps + 6 > self.n_steps):
            raise ValueError("every batch must retain a full three-hour delay window")
        if np.any(np.diff(self.batch_release_steps) < duration_steps):
            raise ValueError("original per-user batches must not overlap")
        nodes = int(round(self.nodes_per_site * self.site_scale))
        if nodes < 1:
            raise ValueError("site_scale must retain at least one node per site")
        capacity = nodes * self.gpus_per_node
        pue = np.asarray(self.pue, dtype=float)
        dynamic_kwh_per_gpu_hour = (self.node_active_it_kw - self.node_idle_it_kw) / self.gpus_per_node
        object.__setattr__(self, "capacity_cu_per_hour", (float(capacity),) * 3)
        object.__setattr__(self, "idle_power_mw", tuple(nodes * self.node_idle_it_kw * pue / 1000))
        object.__setattr__(self, "energy_mwh_per_cu", tuple(dynamic_kwh_per_gpu_hour * pue / 1000))

    @property
    def installed_nodes_per_site(self):
        return int(round(self.nodes_per_site * self.site_scale))

    @property
    def daily_fixed_compute_cost_yuan(self):
        """Existing capacity cost, independent of the daily placement choice."""
        return (sum(self.capacity_cu_per_hour) * self.n_steps * self.dt_hours
                * self.fixed_compute_cost_yuan_per_gpu_capacity_hour)


def make_gpu_scenario(seed=0, config=None):
    """Generate identical GPU-hour jobs/options for every price-policy comparison."""
    cfg = config or GpuMarketConfig()
    if not isinstance(cfg, GpuMarketConfig):
        raise TypeError("make_gpu_scenario requires GpuMarketConfig")
    rng = np.random.default_rng(seed)
    shape = _original_shapes()[int(rng.integers(len(_original_shapes())))]
    shape = np.clip(shape, 0.4, 2.0)
    background = np.column_stack([
        base * (0.72 + 0.28 * np.roll(shape, r * 2)) * rng.uniform(0.95, 1.05)
        for r, base in enumerate((22.0, 20.0, 18.0))])
    hour = np.arange(cfg.n_steps) * cfg.dt_hours
    solar = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    cloud = rng.uniform(0.70, 1.10)
    wind = 5.0 + 2 * np.sin(2 * np.pi * hour / 24 + rng.uniform(0, 6.28))
    renewable = np.maximum(0.5, 37 * cloud * solar + wind)
    spec = replace(default_grid(renewable, cfg.n_steps, cfg.dt_hours),
                   line_capacity_mw=np.asarray(cfg.line_capacity_mw),
                   carbon_price_yuan_per_t=cfg.carbon_price_yuan_per_t)
    duration = int(cfg.batch_duration_hours / cfg.dt_hours)
    effective_scale = cfg.installed_nodes_per_site / cfg.nodes_per_site
    minimum_nodes = int(np.ceil(cfg.batch_parallel_gpu_range[0] * effective_scale / cfg.gpus_per_node))
    maximum_nodes = int(np.floor(cfg.batch_parallel_gpu_range[1] * effective_scale / cfg.gpus_per_node))
    if minimum_nodes < 1 or maximum_nodes < minimum_nodes:
        raise ValueError("scaled batch range must contain at least one complete node")
    if maximum_nodes > cfg.installed_nodes_per_site:
        raise ValueError("a batch cannot require more GPUs than a complete site")
    jobs = []
    for user in range(cfg.n_users):
        beta = cfg.beta_yuan_per_cu_hour[user] * rng.uniform(0.8, 1.2)
        kappa = cfg.migration_loss_yuan_per_cu[user] * rng.uniform(0.8, 1.2)
        for k, release in enumerate(cfg.batch_release_steps):
            parallel_gpus = int(rng.integers(minimum_nodes, maximum_nodes + 1)) * cfg.gpus_per_node
            work = float(parallel_gpus * cfg.batch_duration_hours)
            deadline = int(release + duration + 6)
            placements = [(user, release, True), (user, release, False),
                          (user, release + 2, False), (user, release + 4, False)]
            for region in range(cfg.n_regions):
                if region != user:
                    placements.extend((region, release + delay, False) for delay in (2, 4, 6))
            options = []
            for region, start, original in placements:
                allocation = np.zeros((cfg.n_steps, cfg.n_regions))
                allocation[start:start + duration, region] = work / duration
                extra = np.zeros_like(allocation)
                transfer = np.zeros((cfg.n_steps, cfg.n_regions, cfg.n_regions))
                migrated = region != user
                if migrated:
                    volume = work * cfg.transfer_gb_per_cu
                    transfer[start - 1, user, region] = volume
                    extra[start - 1, region] = volume * cfg.transfer_mwh_per_gb
                delay_hours = (start - release) * cfg.dt_hours
                user_loss = work * (beta * delay_hours + kappa * migrated)
                operating_cost = work * cfg.compute_opex_yuan_per_cu
                operating_cost += float(transfer.sum()) * cfg.transfer_cost_yuan_per_gb
                options.append(ExecutionOption(len(options), region, start, start + duration,
                                              allocation, extra, transfer, delay_hours,
                                              migrated, user_loss, operating_cost, original))
            jobs.append(Job(f"u{user}-j{k}", user, work, user, int(release), deadline,
                            beta, kappa, tuple(options)))
    rng.shuffle(jobs)
    jobs.sort(key=lambda job: job.release_step)
    scenario = Scenario(cfg, int(seed), background, renewable, spec, None, tuple(jobs))
    grid = GridClearing(spec)
    baseline = grid.solve(scenario.baseline_load_mw, None)
    if not baseline.feasible:
        raise ValueError("GPU baseline grid is infeasible; revise the declared physical configuration")
    background_only = grid.solve(background, None)
    if not background_only.feasible:
        raise ValueError("GPU background-only counterfactual is infeasible")
    dc_energy = float(np.sum(scenario.baseline_load_mw - background) * cfg.dt_hours)
    background_energy = float(np.sum(background) * cfg.dt_hours)
    original_allocation = sum((job.options[0].allocation_cu for job in jobs), np.zeros_like(background))
    capacity = np.asarray(cfg.capacity_cu_per_hour) * cfg.dt_hours
    if np.any(original_allocation > capacity + 1e-7):
        raise ValueError("GPU original contracts exceed installed capacity")
    physical_remote_options = []
    for job in jobs:
        feasible_count = 0
        for option in job.options:
            if not option.migrated:
                continue
            proposed = original_allocation - job.options[0].allocation_cu + option.allocation_cu
            if (np.all(proposed <= capacity + 1e-7)
                    and np.all(option.transfer_gb <= cfg.bandwidth_gb_per_hour * cfg.dt_hours + 1e-7)):
                feasible_count += 1
        physical_remote_options.append(feasible_count)
    scenario.calibration = {
        "work_unit": "H100-equivalent GPU-hour on homogeneous eight-GPU nodes",
        "power_model": "PUE * (installed_nodes * idle_kW + active_GPUs / 8 * (active_kW - idle_kW))",
        "hardware_source": "https://docs.nvidia.com/dgx-superpod/design-guides/dgx-superpod-data-center-design-h100/latest/planning.html",
        "power_measurement_reference": "https://arxiv.org/abs/2412.08602",
        "pue_reference": "https://datacenters.google/efficiency/",
        "rental_price_scale_reference": "https://lambda.ai/instances",
        "price_status": "18 yuan/GPU-hour is a study assumption, not an FX-converted provider quote",
        "node_active_it_kw_assumption": cfg.node_active_it_kw,
        "node_idle_it_kw_assumption": cfg.node_idle_it_kw,
        "pue_assumption": tuple(cfg.pue),
        "installed_nodes_per_site": cfg.installed_nodes_per_site,
        "installed_gpus_per_site": cfg.installed_nodes_per_site * cfg.gpus_per_node,
        "base_price_yuan_per_gpu_hour_assumption": cfg.base_price_yuan_per_cu,
        "fixed_capacity_cost_yuan_per_gpu_capacity_hour_assumption": cfg.fixed_compute_cost_yuan_per_gpu_capacity_hour,
        "daily_fixed_compute_cost_yuan": cfg.daily_fixed_compute_cost_yuan,
        "fixed_capacity_cost_scope": "assumed cost of the complete existing installed GPU fleet, independent of daily task placement",
        "capacity_utilization_status": "four daily batch contracts per user use a minority of installed GPU-hours; this is an underutilized batch-only fleet scenario",
        "calibration_status": "hardware-referenced synthetic study assumptions, not a completed market or workload calibration",
        "checkpoint_gb_per_gpu_hour_assumption": cfg.transfer_gb_per_cu,
        "checkpoint_sensitivity_gb_per_gpu_hour": (0.1, 2.0),
        "wan_bandwidth_gb_per_hour_assumption": cfg.bandwidth_gb_per_hour,
        "wan_assumption": "400 Gbit/s per directed regional pair at 70% payload efficiency",
        "network_energy_mwh_per_gb_assumption": cfg.transfer_mwh_per_gb,
        "carbon_policy": "exogenous generator-side carbon cost; no whole-grid hard cap",
        "policy_carbon_price_yuan_per_t": cfg.carbon_price_yuan_per_t,
        "internal_carbon_value_yuan_per_t_assumption": cfg.internal_carbon_value_yuan_per_t,
        "dc_energy_mwh": dc_energy,
        "background_energy_mwh": background_energy,
        "background_only_generation_cost_yuan": background_only.total_generation_cost,
        "dc_energy_share": dc_energy / (background_energy + dc_energy),
        "gpu_capacity_utilization": scenario.total_work_cu / (sum(cfg.capacity_cu_per_hour) * 24),
        "physically_feasible_remote_options_per_job": tuple(physical_remote_options),
        "load_shape": "original EnergyNet synthetic household CSV, normalized and scaled",
        "information": "synthetic deterministic day-ahead benchmark; hardware and prices are not fitted measurements",
    }
    return scenario
