"""Seeded synthetic tasks and grid inputs, with explicit physical units.

The original EnergyNet synthetic household CSV supplies only a dimensionless
daily load shape. Grid magnitudes, weather, jobs and preferences are synthetic.
Prices are computed or offered later; they are not generated as input traces.
"""
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketConfig:
    n_steps: int = 48
    dt_hours: float = 0.5
    n_regions: int = 3
    n_users: int = 3
    jobs_per_user: int = 4
    n_blocks: int = 4
    n_options: int = 10
    base_price_yuan_per_cu: float = 0.80
    discount_levels: tuple = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    quota_levels: tuple = (0.50, 0.75, 1.00)
    capacity_cu_per_hour: tuple = (2600.0, 2600.0, 2600.0)
    idle_power_mw: tuple = (0.08, 0.08, 0.08)
    line_capacity_mw: tuple = (18.0, 7.0, 8.0)
    energy_mwh_per_cu: tuple = (0.00045, 0.00050, 0.00040)
    beta_yuan_per_cu_hour: tuple = (0.006, 0.025, 0.090)
    migration_loss_yuan_per_cu: tuple = (0.004, 0.008, 0.015)
    compute_opex_yuan_per_cu: float = 0.08
    transfer_gb_per_cu: float = 0.012
    bandwidth_gb_per_hour: float = 120.0
    transfer_mwh_per_gb: float = 0.00001
    transfer_cost_yuan_per_gb: float = 0.05
    carbon_tightness: float = 0.35

    def __post_init__(self):
        if (self.n_regions, self.n_users, self.n_steps) != (3, 3, 48):
            raise ValueError("The first scenario uses three regions, three owners and 48 half-hours.")
        if not 0 <= self.carbon_tightness <= 1:
            raise ValueError("carbon_tightness must be between economic and minimum-emission dispatch.")

    @property
    def platform_nvec(self):
        return np.array([len(self.discount_levels)] * (self.n_regions * self.n_blocks)
                        + [len(self.quota_levels)] * self.n_regions, dtype=np.int64)


@dataclass(frozen=True)
class ExecutionOption:
    index: int
    region: int
    start_step: int
    end_step: int
    allocation_cu: np.ndarray
    extra_energy_mwh: np.ndarray
    transfer_gb: np.ndarray
    delay_hours: float
    migrated: bool
    user_loss_yuan: float
    operating_cost_yuan: float
    is_baseline: bool = False


@dataclass(frozen=True)
class Job:
    job_id: str
    user_id: int
    work_cu: float
    home_region: int
    release_step: int
    deadline_step: int
    beta_yuan_per_cu_hour: float
    migration_loss_yuan_per_cu: float
    options: tuple


@dataclass
class Scenario:
    config: MarketConfig
    seed: int
    background_mw: np.ndarray
    renewable_mw: np.ndarray
    grid_spec: object
    carbon_budget_t: float | None
    jobs: tuple
    calibration: dict = field(default_factory=dict)

    @property
    def idle_load_mw(self):
        return np.broadcast_to(np.asarray(self.config.idle_power_mw), (self.config.n_steps, 3)).copy()

    @property
    def total_work_cu(self):
        return float(sum(j.work_cu for j in self.jobs))

    def option_power_mw(self, option):
        return (option.allocation_cu * np.asarray(self.config.energy_mwh_per_cu)
                + option.extra_energy_mwh) / self.config.dt_hours

    def load_for(self, option_indices):
        load = self.background_mw + self.idle_load_mw
        for job, index in zip(self.jobs, option_indices):
            load = load + self.option_power_mw(job.options[int(index)])
        return load

    @property
    def baseline_load_mw(self):
        return self.load_for([0] * len(self.jobs))


@lru_cache(maxsize=1)
def _original_shapes():
    root = Path(__file__).resolve().parents[2]
    path = root / "tests/gym/data_for_tests/synthetic_household_consumption_test.csv"
    frame = pd.read_csv(path)
    col = next((c for c in frame.columns if c.lower() == "consumption"), None)
    if col is None:
        numeric = frame.select_dtypes(include=[np.number])
        col = numeric.columns[-1]
    values = pd.to_numeric(frame[col], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("The original synthetic load needs at least one finite consumption value.")
    values = values[:len(values) // 48 * 48].reshape(-1, 48)
    return values / np.maximum(values.mean(axis=1, keepdims=True), 1e-9)


def make_scenario(seed=0, config=None):
    """Generate one complete deterministic day; all schemes receive the same day.

    The policy cap interpolates between baseline economic and minimum-emission
    dispatch. This guarantees a feasible original contract and is disclosed as
    scenario calibration, not a real ETS allowance allocation.
    """
    from .grid import GridClearing, default_grid
    config = config or MarketConfig()
    rng = np.random.default_rng(seed)
    shape = _original_shapes()[int(rng.integers(len(_original_shapes())))]
    shape = np.clip(shape, 0.4, 2.0)
    background = np.column_stack([
        base * (0.72 + 0.28 * np.roll(shape, r * 2)) * rng.uniform(0.95, 1.05)
        for r, base in enumerate((22.0, 20.0, 18.0))])
    hour = np.arange(48) * config.dt_hours
    solar = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    cloud = rng.uniform(0.70, 1.10)
    wind = 5.0 + 2 * np.sin(2 * np.pi * hour / 24 + rng.uniform(0, 6.28))
    renewable = np.maximum(0.5, 37 * cloud * solar + wind)
    spec = default_grid(renewable, n_steps=48, dt_hours=config.dt_hours)
    spec = replace(spec, line_capacity_mw=np.asarray(config.line_capacity_mw))
    jobs = []
    release_slots = np.linspace(8, 38, config.jobs_per_user).astype(int)
    for user in range(3):
        beta = config.beta_yuan_per_cu_hour[user] * rng.uniform(0.8, 1.2)
        kappa = config.migration_loss_yuan_per_cu[user] * rng.uniform(0.8, 1.2)
        for k, release0 in enumerate(release_slots):
            release = int(release0 + rng.integers(-1, 2))
            duration = 2
            work = float(rng.uniform(1200, 2100))
            deadline = min(48, release + duration + 6)
            placements = [(user, release, True), (user, release, False),
                          (user, release + 2, False), (user, release + 4, False)]
            for r in range(3):
                if r != user:
                    placements.extend((r, release + d, False) for d in (2, 4, 6))
            options = []
            for index, (region, start, original) in enumerate(placements):
                if start + duration > deadline:
                    continue
                allocation = np.zeros((48, 3))
                allocation[start:start + duration, region] = work / duration
                extra = np.zeros((48, 3))
                transfer = np.zeros((48, 3, 3))
                moved = region != user
                if moved:
                    gb = work * config.transfer_gb_per_cu
                    transfer[start - 1, user, region] = gb
                    extra[start - 1, region] = gb * config.transfer_mwh_per_gb
                delay = (start - release) * config.dt_hours
                loss = work * (beta * delay + kappa * moved)
                opex = work * config.compute_opex_yuan_per_cu
                opex += float(transfer.sum()) * config.transfer_cost_yuan_per_gb
                options.append(ExecutionOption(len(options), region, start, start + duration,
                                              allocation, extra, transfer, delay, moved,
                                              loss, opex, original))
            jobs.append(Job(f"u{user}-j{k}", user, work, user, release, deadline,
                            beta, kappa, tuple(options)))
    # Public arrival order; random tie order avoids a permanent owner advantage.
    rng.shuffle(jobs)
    jobs.sort(key=lambda j: j.release_step)
    scenario = Scenario(config, int(seed), background, renewable, spec, float("inf"), tuple(jobs))
    grid = GridClearing(spec)
    economic = grid.solve(scenario.baseline_load_mw, None)
    minimum = grid.solve(scenario.baseline_load_mw, None, objective="emissions")
    if not economic.feasible or not minimum.feasible:
        raise ValueError("Synthetic baseline grid is infeasible; change physical scenario parameters.")
    scenario.carbon_budget_t = (economic.emissions_t * (1 - config.carbon_tightness)
                               + minimum.emissions_t * config.carbon_tightness)
    scenario.calibration = {
        "load_shape": "original EnergyNet synthetic household CSV, normalized and scaled",
        "missing_load_values": "linear interpolation in time before daily normalization",
        "economic_baseline_emissions_t": economic.emissions_t,
        "minimum_baseline_emissions_t": minimum.emissions_t,
        "carbon_tightness": config.carbon_tightness,
        "information": "deterministic day-ahead benchmark; no real-time uncertainty yet",
    }
    return scenario
