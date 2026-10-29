"""Posted compute tariffs, voluntary choices and physical reservations.

An original contract is always retained. Accepted choices are binding: final
electricity prices affect the platform's costs, not the user's locked invoice.
No social-cost bonus, artificial acceptance reward or profit rollback is used.
"""
from dataclasses import dataclass
import numpy as np
from .scenario import MarketConfig, Scenario, make_scenario
from .grid import GridClearing


def zero_platform_action(config=None):
    config = config or MarketConfig()
    return np.r_[np.zeros(config.n_regions * config.n_blocks, dtype=int),
                 np.full(config.n_regions, len(config.quota_levels) - 1, dtype=int)]


def encode_platform_action(discounts, quotas=None, config=None):
    config = config or MarketConfig()
    d = np.asarray(discounts).reshape(config.n_regions, config.n_blocks)
    ids = np.abs(d[..., None] - np.asarray(config.discount_levels)).argmin(axis=-1)
    q = np.ones(3) if quotas is None else np.asarray(quotas)
    qi = np.abs(q[:, None] - np.asarray(config.quota_levels)).argmin(axis=-1)
    return np.r_[ids.ravel(), qi].astype(np.int64)


@dataclass
class MarketOutcome:
    metrics: dict
    selected_options: tuple
    requested_actions: tuple
    bills_yuan: tuple
    user_losses_yuan: tuple
    clearing: object
    reservations: tuple

    def to_dict(self):
        return dict(self.metrics)


class MarketEpisode:
    """One day-ahead pricing game, with one decision per arriving task.

    User observations contain their own monetary preference and option costs.
    The platform observes baseline forecasts/clearing, never private beta or
    its own future final clearing prices.
    """
    option_feature_count = 7
    option_feature_offset = 6

    def __init__(self, scenario: Scenario, grid=None, use_quotas=True):
        self.scenario = scenario
        self.config = scenario.config
        self.jobs = scenario.jobs
        self.grid = grid or GridClearing(scenario.grid_spec)
        self.use_quotas = use_quotas
        self.baseline_clearing = self.grid.solve(scenario.baseline_load_mw,
                                                scenario.carbon_budget_t)
        if not self.baseline_clearing.feasible:
            raise ValueError("Original contracts must have a feasible energy schedule.")
        self._baseline_alloc = sum((j.options[0].allocation_cu for j in self.jobs),
                                   np.zeros((48, 3)))
        self.baseline_bills = np.array([j.work_cu * self.config.base_price_yuan_per_cu
                                       for j in self.jobs])
        self._cache = {tuple([0] * len(self.jobs)): self.baseline_clearing}
        self.reset()
        self.baseline_metrics = self._metrics(self.baseline_clearing,
                                              self.baseline_bills, np.zeros(len(self.jobs)))

    def reset(self, platform_action=None):
        self.position = 0
        self.selected = [0] * len(self.jobs)
        self.requested = []
        self.bills = self.baseline_bills.copy()
        self.losses = np.zeros(len(self.jobs))
        self.reservations = []
        self.current_clearing = self.baseline_clearing
        self.set_platform_action(zero_platform_action(self.config) if platform_action is None
                                 else platform_action)
        return self.platform_observation()

    def set_platform_action(self, action):
        if self.position:
            raise ValueError("Published tariffs are binding for the current episode.")
        a = np.asarray(action, dtype=int).reshape(-1)
        nvec = self.config.platform_nvec
        if len(a) != len(nvec) or np.any(a < 0) or np.any(a >= nvec):
            raise ValueError("Invalid platform tariff/quota action.")
        self.platform_action = a.copy()
        nd = self.config.n_regions * self.config.n_blocks
        self.discounts = np.asarray(self.config.discount_levels)[a[:nd]].reshape(3, 4)
        self.quota_fractions = np.asarray(self.config.quota_levels)[a[nd:]]
        self.tariff_yuan_per_cu = self.config.base_price_yuan_per_cu * (1 - self.discounts)

    @property
    def done(self):
        return self.position >= len(self.jobs)

    @property
    def current_user_id(self):
        return None if self.done else self.jobs[self.position].user_id

    def platform_observation(self):
        cfg = self.config
        public_work = np.array([sum(j.work_cu for j in self.jobs if j.user_id == u)
                                for u in range(3)]) / 10000
        return np.concatenate([
            self.scenario.background_mw.ravel() / 50,
            self.scenario.renewable_mw / 50,
            self.baseline_clearing.lmp_yuan_per_mwh.ravel() / 1000,
            self._baseline_alloc.ravel() / (np.tile(cfg.capacity_cu_per_hour, 48) * cfg.dt_hours),
            public_work,
            [self.baseline_clearing.carbon_price_yuan_per_t / 1000,
             (self.scenario.carbon_budget_t or 0.0) / 2000],
        ]).astype(np.float32)

    def option_bill(self, job, option):
        if option.is_baseline:
            return job.work_cu * self.config.base_price_yuan_per_cu
        blocks = np.minimum(np.arange(48) * 4 // 48, 3)
        price_t = self.tariff_yuan_per_cu[:, blocks].T
        return float(np.sum(option.allocation_cu * price_t))

    def allocations_for(self, indices):
        allocation = np.zeros((48, 3))
        transfer = np.zeros((48, 3, 3))
        for job, index in zip(self.jobs, indices):
            o = job.options[int(index)]
            allocation += o.allocation_cu
            transfer += o.transfer_gb
        return allocation, transfer

    def _precheck(self, indices):
        allocation, transfer = self.allocations_for(indices)
        physical = np.asarray(self.config.capacity_cu_per_hour) * self.config.dt_hours
        if np.any(allocation > physical + 1e-7):
            return False, "compute_capacity"
        if np.any(transfer > self.config.bandwidth_gb_per_hour * self.config.dt_hours + 1e-7):
            return False, "bandwidth"
        if self.use_quotas:
            # An offer can restrict additional bookings, never revoke a baseline.
            limit = np.maximum(self._baseline_alloc, physical * self.quota_fractions)
            if np.any(allocation > limit + 1e-7):
                return False, "posted_quota"
        return True, "accepted"

    def _candidate(self, index):
        if self.done or index < 0 or index >= len(self.jobs[self.position].options):
            return False, None, "missing_option"
        indices = list(self.selected)
        indices[self.position] = int(index)
        ok, reason = self._precheck(indices)
        if not ok:
            return False, None, reason
        # Original and same-place new-price contracts have identical physics.
        key = tuple(0 if i == 1 else i for i in indices)
        if key not in self._cache:
            self._cache[key] = self.grid.solve(self.scenario.load_for(indices),
                                               self.scenario.carbon_budget_t)
        result = self._cache[key]
        return result.feasible, result, "accepted" if result.feasible else "energy_feasibility"

    def current_user_observation(self):
        if self.done:
            raise ValueError("No pending task.")
        j = self.jobs[self.position]
        base = j.work_cu * self.config.base_price_yuan_per_cu
        features = np.zeros((self.config.n_options, self.option_feature_count))
        for k, o in enumerate(j.options):
            proposed = list(self.selected)
            proposed[self.position] = k
            possible, _ = self._precheck(proposed)
            alloc, _ = self.allocations_for(proposed)
            features[k] = [1, possible, self.option_bill(j, o) / base,
                           o.user_loss_yuan / base, o.delay_hours / 3,
                           o.region / 2, np.max(alloc / (np.asarray(self.config.capacity_cu_per_hour)
                                                       * self.config.dt_hours))]
        remaining = sum(x.user_id == j.user_id for x in self.jobs[self.position:])
        return np.concatenate([
            [j.user_id / 2, j.work_cu / 3000, j.release_step / 48,
             j.deadline_step / 48, j.beta_yuan_per_cu_hour / .1,
             j.migration_loss_yuan_per_cu / .05], features.ravel(),
            self.discounts.ravel(), self.quota_fractions, [remaining / self.config.jobs_per_user],
        ]).astype(np.float32)

    def best_response_action(self):
        """Exact *current-task* response, not a global Nash/owner optimum."""
        j = self.jobs[self.position]
        ranked = sorted(range(len(j.options)),
                        key=lambda i: (self.option_bill(j, j.options[i]) + j.options[i].user_loss_yuan, i))
        for i in ranked:
            if self._candidate(i)[0]:
                return i
        return 0

    def step_user(self, action):
        """Return (own_normalized_reward, done, info), preserving all tasks."""
        if self.done:
            raise ValueError("Episode finished.")
        position = self.position
        job = self.jobs[position]
        requested = int(np.asarray(action).item())
        feasible, result, reason = self._candidate(requested)
        chosen = requested if feasible else 0
        option = job.options[chosen]
        if feasible:
            self.current_clearing = result
        self.selected[position] = chosen
        bill = self.option_bill(job, option)
        loss = option.user_loss_yuan
        self.bills[position], self.losses[position] = bill, loss
        self.requested.append(requested)
        own_gain = self.baseline_bills[position] - bill - loss
        info = {"job_id": job.job_id, "user_id": job.user_id,
                "requested": requested, "chosen": chosen, "reservation_reason": reason,
                "bill_yuan": bill, "loss_yuan": loss, "utility_gain_yuan": own_gain,
                "individual_rational": bool(own_gain >= -1e-7),
                "migrated": option.migrated, "delay_hours": option.delay_hours,
                "work_cu": job.work_cu}
        self.reservations.append(info)
        self.position += 1
        reward = 10 * own_gain / self.baseline_bills[position]
        return float(reward), self.done, info

    def _metrics(self, clearing, bills, losses):
        scenario, cfg = self.scenario, self.config
        load = scenario.load_for(self.selected)
        dc_power = load - scenario.background_mw
        energy_bill = float(np.sum(dc_power * cfg.dt_hours * clearing.lmp_yuan_per_mwh))
        opex = float(sum(j.options[i].operating_cost_yuan for j, i in zip(self.jobs, self.selected)))
        revenue = float(np.sum(bills))
        loss = float(np.sum(losses))
        user_gain = self.baseline_bills - np.asarray(bills) - np.asarray(losses)
        moved = sum(j.options[i].migrated for j, i in zip(self.jobs, self.selected))
        delayed = sum(j.options[i].delay_hours > 0 for j, i in zip(self.jobs, self.selected))
        deadlines = all(j.options[i].end_step <= j.deadline_step for j, i in zip(self.jobs, self.selected))
        work = sum(float(j.options[i].allocation_cu.sum()) for j, i in zip(self.jobs, self.selected))
        result = {
            "seed": scenario.seed, "jobs": len(self.jobs), "work_cu": work,
            "required_work_cu": scenario.total_work_cu,
            "completion_rate": work / scenario.total_work_cu,
            "deadline_compliance": float(deadlines),
            "generation_cost_yuan": clearing.total_generation_cost,
            "compute_migration_cost_yuan": opex,
            "resource_cost_yuan": clearing.total_generation_cost + opex,
            "user_loss_yuan": loss,
            "system_cost_yuan": clearing.total_generation_cost + opex + loss,
            "service_revenue_yuan": revenue,
            "electricity_bill_yuan": energy_bill,
            "platform_profit_yuan": revenue - energy_bill - opex,
            "user_generalized_cost_yuan": revenue + loss,
            "user_gain_yuan": float(user_gain.sum()),
            "user_ir_rate": float(np.mean(user_gain >= -1e-7)),
            "carbon_emissions_t": clearing.emissions_t,
            "carbon_budget_t": scenario.carbon_budget_t,
            "carbon_price_yuan_per_t": clearing.carbon_price_yuan_per_t,
            "mean_lmp_yuan_per_mwh": float(clearing.lmp_yuan_per_mwh.mean()),
            "migrated_jobs": int(moved), "delayed_jobs": int(delayed),
            "reservation_rejections": sum(r["requested"] != r["chosen"] for r in self.reservations),
            "max_balance_error_mw": clearing.max_balance_error,
        }
        for user in range(3):
            ids = [k for k, j in enumerate(self.jobs) if j.user_id == user]
            result[f"user_{user}_gain_yuan"] = float(user_gain[ids].sum())
        return result

    def outcome(self):
        if not self.done:
            raise ValueError("All original tasks must be assigned before settlement.")
        metrics = self._metrics(self.current_clearing, self.bills, self.losses)
        for name in ("platform_profit_yuan", "system_cost_yuan", "resource_cost_yuan"):
            metrics[f"baseline_{name}"] = self.baseline_metrics[name]
            metrics[f"delta_{name}"] = metrics[name] - self.baseline_metrics[name]
        metrics["platform_participates"] = bool(metrics["delta_platform_profit_yuan"] >= -1e-6)
        return MarketOutcome(metrics, tuple(self.selected), tuple(self.requested),
                             tuple(self.bills), tuple(self.losses), self.current_clearing,
                             tuple(self.reservations))

    @property
    def platform_reward(self):
        metrics = self.outcome().metrics
        return float(metrics["delta_platform_profit_yuan"] /
                     (self.scenario.total_work_cu * self.config.base_price_yuan_per_cu * .01))

    def play(self, user_policies=None):
        while not self.done:
            if user_policies is None:
                action = self.best_response_action()
            else:
                policy = user_policies[self.current_user_id]
                obs = self.current_user_observation()
                action = policy.predict(obs, deterministic=True)[0] if hasattr(policy, "predict") else policy(obs)
            self.step_user(action)
        return self.outcome()


def exact_user_policy(observation):
    """Observable current-task cost minimization; final energy admission is checked separately."""
    start = MarketEpisode.option_feature_offset
    features = np.asarray(observation)[start:start + 70].reshape(10, 7)
    costs = features[:, 2] + features[:, 3]
    costs[(features[:, 0] < .5) | (features[:, 1] < .5)] = np.inf
    return int(np.argmin(costs))


def make_episode(seed=0, config=None):
    return MarketEpisode(make_scenario(seed, config))
