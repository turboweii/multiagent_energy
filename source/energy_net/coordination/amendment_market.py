"""Pre-execution amendments to already accepted legacy compute contracts.

Legacy bookings remain executable throughout. Users first propose a conditional
amendment; physical feasibility and the public net-benefits test are checked
before it becomes binding. Accepted amendments share the surplus remaining
after their additional user rebate. No realised contract is undone afterward.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, replace
from functools import lru_cache

import numpy as np

from .cached_grid import MemoizedGridClearing
from .carbon import consumption_emissions_t, nodal_carbon_intensity
from .flex_market import FlexMarketConfig
from .gpu_scenario import GpuMarketConfig
from .grid import GridClearing
from .learning import ActorEnv
from .market import MarketEpisode, MarketOutcome, zero_platform_action


@dataclass(frozen=True)
class AmendmentConfig(FlexMarketConfig):
    """The same GPU physics, with direct, net-first or gross-first settlement."""

    settlement_order: str = "net"
    carbon_eligibility: bool = False  # The physical cap is common to all regimes.
    internal_carbon_value_yuan_per_t: float = 0.0
    admission_tolerance_yuan: float = 1e-7

    def __post_init__(self):
        super().__post_init__()
        if self.settlement_order not in ("net", "gross", "direct"):
            raise ValueError("settlement_order must be 'net', 'gross' or 'direct'")
        if (not np.isfinite(self.admission_tolerance_yuan)
                or self.admission_tolerance_yuan < 0):
            raise ValueError("admission_tolerance_yuan must be finite and nonnegative")
        if self.internal_carbon_value_yuan_per_t != 0:
            raise ValueError("Amendment firms use settled cash profit, with internal carbon value zero")


def option_physics_key(option):
    """The complete power/compute/network plan, independent of index aliases."""
    return tuple(np.asarray(getattr(option, name), dtype=np.float64).tobytes(order="C")
                 for name in ("allocation_cu", "extra_energy_mwh", "transfer_gb"))


def _booking_value(booking, name, default=None):
    return booking.get(name, default) if isinstance(booking, dict) else getattr(booking, name, default)


class AmendmentEpisode(MarketEpisode):
    """One extra-rebate menu, followed by sequential pre-binding proposals.

    Index zero in each task is its accepted legacy execution. Remaining unique
    physical executions are reindexed, and unavailable padding keeps ten user
    actions. The old option-1 alias is never reused as a clearing-cache key.
    """

    def __init__(self, scenario, booking, grid=None, config=None):
        if config is None:
            if isinstance(scenario.config, AmendmentConfig):
                config = scenario.config
            else:
                values = {field.name: getattr(scenario.config, field.name)
                          for field in fields(GpuMarketConfig)}
                values["internal_carbon_value_yuan_per_t"] = 0.
                config = AmendmentConfig(**values)
        if not isinstance(config, AmendmentConfig):
            raise TypeError("AmendmentEpisode requires AmendmentConfig")
        original = replace(scenario, config=config)
        booking_scenario = _booking_value(booking, "scenario", scenario)
        if (scenario.seed != booking_scenario.seed
                or scenario.carbon_budget_t != booking_scenario.carbon_budget_t):
            raise ValueError("Bookings and amendments must use the same day and physical carbon cap")
        indices = tuple(int(index) for index in _booking_value(booking, "selected_options"))
        if len(indices) != len(original.jobs):
            raise ValueError("Provide one accepted legacy execution per task")
        self.booking = booking
        # The factory caches this unmodified source object. Booking copies may
        # carry independent scalar/calibration snapshots of the same inputs.
        self.original_scenario = scenario
        self.config = config
        self.outside_selected_options = indices
        self.outside_platform_action = tuple(_booking_value(booking, "platform_action", ()))
        self.outside_policy_label = _booking_value(booking, "platform_policy_label", "legacy_selected")
        self.original_metrics = deepcopy(_booking_value(
            booking, "original_metrics", _booking_value(booking, "baseline_metrics", {})))
        self.outside_reported_metrics = deepcopy(_booking_value(booking, "metrics", {}))
        self.baseline_bills = np.asarray(_booking_value(booking, "bills_yuan"), dtype=float).copy()
        self.outside_losses = np.asarray(_booking_value(booking, "user_losses_yuan"), dtype=float).copy()
        if (self.baseline_bills.shape != (len(indices),) or self.outside_losses.shape != (len(indices),)
                or not np.all(np.isfinite(self.baseline_bills)) or np.any(self.baseline_bills < 0)
                or not np.all(np.isfinite(self.outside_losses))):
            raise ValueError("Legacy invoices and private inconvenience must be finite task vectors")
        jobs, availability, source_indices = [], [], []
        for job, outside_index, outside_loss in zip(original.jobs, indices, self.outside_losses):
            if not 0 <= outside_index < len(job.options):
                raise ValueError("The booked legacy execution is missing")
            outside = job.options[outside_index]
            if not np.isclose(outside.user_loss_yuan, outside_loss, rtol=0, atol=1e-6):
                raise ValueError("Booked user inconvenience must describe the accepted physical execution")
            ordered = [outside_index]
            seen = {option_physics_key(outside)}
            for index, option in enumerate(job.options):
                key = option_physics_key(option)
                if key not in seen:
                    ordered.append(index); seen.add(key)
            if len(ordered) > config.n_options:
                raise ValueError("The amendment policy supports at most ten distinct physical executions")
            available = [True] * len(ordered)
            ordered += [outside_index] * (config.n_options - len(ordered))
            available += [False] * (config.n_options - len(available))
            options = tuple(replace(job.options[source], index=index, is_baseline=index == 0)
                            for index, source in enumerate(ordered))
            jobs.append(replace(job, options=options))
            availability.append(tuple(available)); source_indices.append(tuple(ordered))
        self.jobs = tuple(jobs)
        self.option_available = tuple(availability)
        self.option_source_indices = tuple(source_indices)
        self.scenario = replace(original, jobs=self.jobs)
        self.grid = grid or GridClearing(original.grid_spec)
        self.use_quotas = True
        self.baseline_clearing = self.grid.solve(self.scenario.baseline_load_mw,
                                                self.scenario.carbon_budget_t)
        if not self.baseline_clearing.feasible:
            raise ValueError("Accepted legacy bookings must remain physically feasible")
        booked_clearing = _booking_value(booking, "clearing")
        if booked_clearing is not None and (not booked_clearing.feasible or not np.allclose(
                booked_clearing.generation_mw, self.baseline_clearing.generation_mw, rtol=0, atol=1e-6)):
            raise ValueError("Booking clearing and common-cap amendment reference disagree")
        self._baseline_alloc = sum((job.options[0].allocation_cu for job in self.jobs),
                                   np.zeros((config.n_steps, config.n_regions)))
        self._outside_operations = float(sum(job.options[0].operating_cost_yuan for job in self.jobs))
        self._outside_electricity_bill = self._energy_bill(
            self.baseline_clearing, [0] * len(self.jobs))
        self._outside_profit = (float(self.baseline_bills.sum()) - self._outside_electricity_bill
                                - self._outside_operations - config.daily_fixed_compute_cost_yuan)
        self._cache = {tuple([0] * len(self.jobs)): self.baseline_clearing}
        self.reset()
        self.baseline_metrics = self._metrics(self.baseline_clearing, self.baseline_bills,
                                              self.outside_losses)

    def reset(self, platform_action=None):
        self.position = 0
        self.selected = [0] * len(self.jobs)
        self.requested, self.reservations, self.amendment_ledger = [], [], []
        self.bills = self.baseline_bills.copy()
        self.losses = self.outside_losses.copy()
        self.current_clearing = self.baseline_clearing
        self.set_platform_action(np.r_[zero_platform_action(self.config), 0]
                                 if platform_action is None else platform_action)
        return self.platform_observation()

    def set_platform_action(self, action):
        values = np.asarray(action, dtype=int).reshape(-1)
        if len(values) == 15:
            opened = bool(np.any(values[:12] != 0))
        elif len(values) == 16:
            if values[-1] not in (0, 1):
                raise ValueError("The amendment-open flag must be zero or one")
            opened = bool(values[-1]); values = values[:15]
        else:
            raise ValueError("Provide fifteen prices/quotas and an optional amendment-open flag")
        MarketEpisode.set_platform_action(self, values)
        self.amendment_open = opened
        self.platform_action = np.r_[self.platform_action, int(opened)].astype(np.int64)

    def option_is_changed(self, job, option):
        return option_physics_key(option) != option_physics_key(job.options[0])

    def option_bill(self, job, option):
        index = next(index for index, candidate in enumerate(self.jobs) if candidate.job_id == job.job_id)
        outside_bill = self.baseline_bills[index]
        if not self.option_is_changed(job, option):
            return float(outside_bill)
        blocks = np.minimum(np.arange(self.config.n_steps) * self.config.n_blocks
                            // self.config.n_steps, self.config.n_blocks - 1)
        # A new schedule retains the booked rental invoice and receives only
        # an additional target-region/time rebate, never a repriced base bill.
        rebate_rate = (outside_bill / job.work_cu) * self.discounts[:, blocks].T
        return float(outside_bill - np.sum(option.allocation_cu * rebate_rate))

    def _energy_bill(self, clearing, indices):
        dc_power = self.scenario.load_for(indices) - self.scenario.background_mw
        return float(np.sum(dc_power * clearing.lmp_yuan_per_mwh) * self.config.dt_hours)

    def _operations(self, indices):
        return float(sum(job.options[index].operating_cost_yuan
                         for job, index in zip(self.jobs, indices)))

    def preview_amendment(self, index):
        """Public pre-binding buyer calculation; private inconvenience is absent."""
        index = int(np.asarray(index).item())
        empty = dict(admissible=False, feasible=False, clearing=None,
                     resource_saving_yuan=0., additional_rebate_yuan=0.,
                     pre_rebate_private_saving_yuan=0., net_surplus_yuan=0.,
                     platform_gain_yuan=0., buyer_gain_yuan=0., settlement_yuan=0.)
        if self.done or not 0 <= index < self.config.n_options:
            return dict(empty, reason="missing_option")
        if index == 0:
            return dict(empty, admissible=True, feasible=True, clearing=self.current_clearing,
                        reason="outside_contract")
        if not self.amendment_open:
            return dict(empty, reason="amendment_round_closed")
        if not self.option_available[self.position][index]:
            return dict(empty, reason="padding_option")
        job = self.jobs[self.position]
        option = job.options[index]
        if not self.option_is_changed(job, option):
            return dict(empty, reason="unchanged_execution")
        proposed = list(self.selected); proposed[self.position] = index
        possible, reason = self._precheck(proposed)
        if not possible:
            return dict(empty, reason=reason)
        key = tuple(proposed)
        if key not in self._cache:
            self._cache[key] = self.grid.solve(self.scenario.load_for(proposed),
                                               self.scenario.carbon_budget_t)
        result = self._cache[key]
        if not result.feasible:
            return dict(empty, reason="energy_feasibility")
        old_ops, new_ops = self._operations(self.selected), self._operations(proposed)
        resource = float(self.current_clearing.total_generation_cost + old_ops
                         - result.total_generation_cost - new_ops)
        rebate = float(self.baseline_bills[self.position] - self.option_bill(job, option))
        private = float(self._energy_bill(self.current_clearing, self.selected) + old_ops
                        - self._energy_bill(result, proposed) - new_ops)
        net = resource - rebate
        if self.config.settlement_order == "net":
            profit = self.config.resource_share * net
            admissible = bool(net >= -self.config.admission_tolerance_yuan)
        elif self.config.settlement_order == "gross":
            profit = self.config.resource_share * resource - rebate
            admissible = bool(profit >= -self.config.admission_tolerance_yuan
                              and net >= -self.config.admission_tolerance_yuan)
        else:
            profit = private - rebate
            admissible = bool(profit >= -self.config.admission_tolerance_yuan)
        return dict(admissible=admissible, feasible=True, clearing=result,
                    resource_saving_yuan=resource, additional_rebate_yuan=rebate,
                    pre_rebate_private_saving_yuan=private, net_surplus_yuan=net,
                    platform_gain_yuan=float(profit), buyer_gain_yuan=float(net - profit),
                    settlement_yuan=float(profit - (private - rebate)),
                    reason="accepted" if admissible else "negative_distributable_surplus")

    def _candidate(self, index):
        proposal = self.preview_amendment(index)
        return proposal["admissible"], proposal["clearing"], proposal["reason"]

    def current_user_observation(self):
        if self.done:
            raise ValueError("No pending amendment")
        job = self.jobs[self.position]
        quantity = job.work_cu
        feature = np.zeros((self.config.n_options, self.option_feature_count))
        capacity = np.asarray(self.config.capacity_cu_per_hour) * self.config.dt_hours
        for index, option in enumerate(job.options):
            proposed = list(self.selected); proposed[self.position] = index
            # Commodity availability is public: the same pre-binding physical
            # and budget test used by the buyer. No private loss enters it.
            possible = self.preview_amendment(index)["admissible"]
            available = self.option_available[self.position][index] and (self.amendment_open or index == 0)
            allocation, _ = self.allocations_for(proposed)
            feature[index] = [float(available), float(possible),
                              (self.option_bill(job, option) - self.baseline_bills[self.position]) / quantity,
                              (option.user_loss_yuan - self.outside_losses[self.position]) / quantity,
                              (option.delay_hours - job.options[0].delay_hours) / 3,
                              option.region / 2, np.max(allocation / capacity)]
        remaining = sum(other.user_id == job.user_id for other in self.jobs[self.position:])
        return np.r_[[job.user_id / 2, quantity / 20000, job.release_step / 48,
                      job.deadline_step / 48, job.beta_yuan_per_cu_hour / .1,
                      job.migration_loss_yuan_per_cu / .05], feature.ravel(),
                     self.discounts.ravel(), self.quota_fractions,
                     remaining / self.config.jobs_per_user].astype(np.float32)

    def best_response_action(self):
        if not self.amendment_open:
            return 0
        job = self.jobs[self.position]
        ranked = sorted(range(self.config.n_options), key=lambda index: (
            self.option_bill(job, job.options[index]) + job.options[index].user_loss_yuan, index))
        outside_cost = self.baseline_bills[self.position] + self.outside_losses[self.position]
        for index in ranked:
            cost = self.option_bill(job, job.options[index]) + job.options[index].user_loss_yuan
            if cost <= outside_cost + 1e-7 and self.preview_amendment(index)["admissible"]:
                return index
        return 0

    def step_user(self, action):
        if self.done:
            raise ValueError("Episode finished")
        position, job = self.position, self.jobs[self.position]
        requested = int(np.asarray(action).item())
        proposal = self.preview_amendment(requested)
        accepted = bool(requested != 0 and proposal["admissible"])
        reason = proposal["reason"]
        if accepted:
            offered = job.options[requested]
            own_gain = (self.baseline_bills[position] + self.outside_losses[position]
                        - self.option_bill(job, offered) - offered.user_loss_yuan)
            # This is the user's private reservation decision, separate from
            # the buyer's public resource-surplus test in preview_amendment.
            if own_gain < -1e-7:
                accepted = False; reason = "user_reservation_utility"
        chosen = requested if accepted else 0
        option = job.options[chosen]
        if accepted:
            self.selected[position] = chosen
            self.current_clearing = proposal["clearing"]
            self.amendment_ledger.append(dict(job_id=job.job_id, position=position,
                **{key: value for key, value in proposal.items() if key != "clearing"}))
        self.bills[position] = self.option_bill(job, option)
        self.losses[position] = option.user_loss_yuan
        own_gain = (self.baseline_bills[position] + self.outside_losses[position]
                    - self.bills[position] - self.losses[position])
        info = dict(job_id=job.job_id, user_id=job.user_id, requested=requested, chosen=chosen,
                    chosen_source_option=self.option_source_indices[position][chosen],
                    outside_source_option=self.outside_selected_options[position],
                    reservation_reason=reason, amendment_accepted=accepted, amended=accepted,
                    bill_yuan=float(self.bills[position]), loss_yuan=float(self.losses[position]),
                    outside_bill_yuan=float(self.baseline_bills[position]),
                    outside_loss_yuan=float(self.outside_losses[position]),
                    additional_rebate_yuan=float(self.baseline_bills[position] - self.bills[position]),
                    utility_gain_yuan=float(own_gain), individual_rational=bool(own_gain >= -1e-7),
                    migrated=option.migrated, delay_hours=option.delay_hours,
                    additional_delay_hours=option.delay_hours - job.options[0].delay_hours,
                    work_cu=job.work_cu)
        self.requested.append(requested); self.reservations.append(info); self.position += 1
        return float(own_gain / job.work_cu), self.done, info

    def _metrics(self, clearing, bills, losses):
        metrics = MarketEpisode._metrics(self, clearing, bills, losses)
        cfg, scenario = self.config, self.scenario
        gains = self.baseline_bills + self.outside_losses - np.asarray(bills) - np.asarray(losses)
        transfer = float(sum(row["settlement_yuan"] for row in self.amendment_ledger))
        promised_profit = float(sum(row["platform_gain_yuan"] for row in self.amendment_ledger))
        buyer = float(sum(row["buyer_gain_yuan"] for row in self.amendment_ledger))
        resource = (self.baseline_clearing.total_generation_cost + self._outside_operations
                    - metrics["resource_cost_yuan"])
        rebate = float(self.baseline_bills.sum() - np.asarray(bills).sum())
        private = self._outside_electricity_bill + self._outside_operations - metrics["electricity_bill_yuan"] - metrics["compute_migration_cost_yuan"]
        cash_before_fixed = metrics["platform_profit_yuan"]
        cash_before = cash_before_fixed - cfg.daily_fixed_compute_cost_yuan
        metrics["platform_profit_yuan"] = cash_before + transfer
        metrics["platform_profit_before_transfer_yuan"] = cash_before
        metrics["platform_profit_before_fixed_and_transfer_yuan"] = cash_before_fixed
        metrics["platform_profit_before_fixed_cost_yuan"] = cash_before_fixed + transfer
        metrics["fixed_compute_cost_yuan"] = float(cfg.daily_fixed_compute_cost_yuan)
        metrics["dispatch_system_cost_yuan"] = metrics["system_cost_yuan"]
        metrics["full_system_cost_yuan"] = metrics["system_cost_yuan"] + cfg.daily_fixed_compute_cost_yuan
        metrics["user_gain_yuan"] = float(gains.sum())
        metrics["user_ir_rate"] = float(np.mean(gains >= -1e-7))
        for user in range(cfg.n_users):
            metrics[f"user_{user}_gain_yuan"] = float(sum(gains[i] for i, job in enumerate(self.jobs) if job.user_id == user))
        load = scenario.load_for(self.selected)
        dc_power = load - scenario.background_mw
        intensity = nodal_carbon_intensity(scenario.grid_spec, clearing, load)
        metrics.update(
            work_gpu_hours=metrics["work_cu"],
            dc_consumption_emissions_t=consumption_emissions_t(dc_power, intensity, cfg.dt_hours),
            dc_energy_mwh=float(dc_power.sum() * cfg.dt_hours),
            grid_energy_mwh=float(load.sum() * cfg.dt_hours),
            policy_carbon_price_yuan_per_t=clearing.policy_carbon_price_yuan_per_t,
            cap_shadow_price_yuan_per_t=clearing.cap_shadow_price_yuan_per_t,
            generator_carbon_payment_yuan=clearing.carbon_payment_yuan,
            grid_dispatch_objective_yuan=clearing.dispatch_objective_yuan,
            internal_carbon_value_yuan_per_t=0., internal_carbon_valuation_yuan=0.,
            platform_objective_yuan=float(metrics["platform_profit_yuan"]),
            settlement_order=cfg.settlement_order, resource_share=cfg.resource_share,
            amendment_open=bool(self.amendment_open), outside_policy_label=self.outside_policy_label,
            outside_platform_profit_yuan=float(self._outside_profit),
            outside_resource_cost_yuan=float(self.baseline_clearing.total_generation_cost + self._outside_operations),
            outside_user_loss_yuan=float(self.outside_losses.sum()),
            outside_service_revenue_yuan=float(self.baseline_bills.sum()),
            outside_electricity_bill_yuan=float(self._outside_electricity_bill),
            outside_carbon_emissions_t=float(self.baseline_clearing.emissions_t),
            flexibility_resource_saving_yuan=float(resource),
            platform_pre_rebate_saving_yuan=float(private), user_rebate_yuan=rebate,
            additional_user_rebate_yuan=rebate,
            net_distributable_surplus_yuan=float(resource - rebate),
            shared_resource_entitlement_yuan=promised_profit,
            flexibility_settlement_yuan=transfer,
            buyer_payment_to_platform_yuan=float(sum(max(row["settlement_yuan"], 0.) for row in self.amendment_ledger)),
            platform_payment_to_buyer_yuan=float(sum(max(-row["settlement_yuan"], 0.) for row in self.amendment_ledger)),
            buyer_pre_settlement_benefit_yuan=float(resource - private),
            buyer_post_settlement_benefit_yuan=buyer,
            cash_change_before_transfer_yuan=float(private - rebate),
            cash_change_after_transfer_yuan=float(metrics["platform_profit_yuan"] - self._outside_profit),
            platform_gain_vs_outside_yuan=float(metrics["platform_profit_yuan"] - self._outside_profit),
            platform_floor_shortfall_yuan=float(max(self._outside_profit - metrics["platform_profit_yuan"], 0.)),
            platform_floor_satisfied=bool(metrics["platform_profit_yuan"] >= self._outside_profit - 1e-6),
            accepted_amendments=len(self.amendment_ledger),
            proposed_amendments=sum(index != 0 for index in self.requested),
            amendment_admission_rejections=sum(row["requested"] != 0 and not row["amendment_accepted"] for row in self.reservations),
            contract_system_cost_saving_yuan=float(resource - (np.sum(losses) - self.outside_losses.sum())),
            sum_step_platform_gains_yuan=promised_profit,
            platform_profit_identity_error_yuan=float(metrics["platform_profit_yuan"] - self._outside_profit - promised_profit),
            settlement_grid_emissions_change_t=float(clearing.emissions_t - self.baseline_clearing.emissions_t),
        )
        metrics["dc_energy_share"] = metrics["dc_energy_mwh"] / metrics["grid_energy_mwh"]
        metrics["surplus_allocation_sum_yuan"] = metrics["platform_gain_vs_outside_yuan"] + metrics["user_gain_yuan"] + buyer
        metrics["surplus_identity_error_yuan"] = metrics["surplus_allocation_sum_yuan"] - metrics["contract_system_cost_saving_yuan"]
        background = float(scenario.calibration["background_only_generation_cost_yuan"])
        metrics["background_only_generation_cost_yuan"] = background
        metrics["incremental_system_cost_yuan"] = metrics["system_cost_yuan"] - background
        return metrics

    def outcome(self):
        if not self.done:
            raise ValueError("All legacy bookings must be resolved before settlement")
        metrics = self._metrics(self.current_clearing, self.bills, self.losses)
        names = ("platform_profit_yuan", "system_cost_yuan", "resource_cost_yuan",
                 "dispatch_system_cost_yuan", "full_system_cost_yuan", "platform_objective_yuan",
                 "carbon_emissions_t", "dc_consumption_emissions_t", "dc_energy_mwh",
                 "incremental_system_cost_yuan")
        for name in names:
            metrics[f"baseline_{name}"] = self.baseline_metrics[name]
            metrics[f"outside_{name}"] = self.baseline_metrics[name]
            metrics[f"delta_{name}"] = metrics[name] - self.baseline_metrics[name]
            if name in self.original_metrics:
                metrics[f"original_{name}"] = self.original_metrics[name]
                metrics[f"delta_{name}_vs_original"] = metrics[name] - self.original_metrics[name]
        metrics["platform_participates"] = bool(metrics["delta_platform_profit_yuan"] >= -1e-6)
        metrics["platform_objective_participates"] = metrics["platform_participates"]
        metrics["grid_emissions_reduction_t"] = -metrics["delta_carbon_emissions_t"]
        metrics["dc_footprint_reduction_t"] = -metrics["delta_dc_consumption_emissions_t"]
        metrics["social_saving_yuan"] = -metrics["delta_system_cost_yuan"]
        metrics["incremental_system_cost_saving_vs_outside_yuan"] = -metrics["delta_system_cost_yuan"]
        metrics["selected_original_option_indices"] = [self.option_source_indices[i][index]
                                                       for i, index in enumerate(self.selected)]
        metrics["outside_original_option_indices"] = list(self.outside_selected_options)
        return MarketOutcome(metrics, tuple(self.selected), tuple(self.requested),
                             tuple(self.bills), tuple(self.losses), self.current_clearing,
                             tuple(deepcopy(self.reservations)))

    @property
    def platform_reward(self):
        return float(self.outcome().metrics["platform_gain_vs_outside_yuan"] / 1000.)


class AmendmentPlatformEnv(ActorEnv):
    def __init__(self, *args, reward_scale_yuan=1000., **kwargs):
        super().__init__(*args, **kwargs)
        if self.actor != "platform" or not np.isfinite(reward_scale_yuan) or reward_scale_yuan <= 0:
            raise ValueError("AmendmentPlatformEnv needs the platform and a positive reward scale")
        self.reward_scale_yuan = float(reward_scale_yuan)

    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)
        reward = info["market_metrics"]["platform_gain_vs_outside_yuan"] / self.reward_scale_yuan
        return observation, float(reward), terminated, truncated, info


def amendment_episode_factory(scenario_factory, booking_factory, config=None,
                              maxsize=64, grid_cache_size=512):
    """Reuse immutable booked days and exact OPF results, never live episodes."""
    config = config or AmendmentConfig()

    @lru_cache(maxsize=maxsize)
    def scenario(seed):
        return replace(scenario_factory(int(seed)), config=config)

    @lru_cache(maxsize=maxsize)
    def booking(seed):
        return booking_factory(int(seed))

    @lru_cache(maxsize=maxsize)
    def grid(seed):
        return MemoizedGridClearing(scenario(int(seed)).grid_spec, maxsize=grid_cache_size)

    def factory(seed):
        return AmendmentEpisode(scenario(int(seed)), booking(int(seed)), grid(int(seed)))
    factory.config = config
    factory.scenario_cache_info = scenario.cache_info
    factory.booking_cache_info = booking.cache_info
    return factory
