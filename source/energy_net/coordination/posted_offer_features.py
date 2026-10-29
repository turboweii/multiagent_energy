"""Static single-contract actions and public sequential pricing observations.

The platform picks the actual region, delay and rebate. Public previews only
exclude contracts that cannot be honoured; they never replace its choice by
a better route. User inconvenience, preferences and policies are absent here.
The binary user's reservation-utility mask is private. Public feedback has
only its acceptance bit, so an IR refusal and a strategic refusal are not
distinguished in the platform observation.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from .carbon import nodal_carbon_intensity


class PostedOfferCatalogue:
    """Exit plus 3 destinations x 4 original-release delays x 5 rebates."""

    regions = (0, 1, 2)
    delays_hours = (0, 1, 2, 3)
    rates = (0., .0025, .005, .01, .02)
    n_actions = 61

    def __init__(self, config=None):
        self.config = config
        self.terms = (None, *tuple(product(self.regions, self.delays_hours, self.rates)))
        self.labels = ("keep_outside", *(f"region_{region}_delay_{delay}h_extra_discount_{rate:.4f}"
                                        for region, delay, rate in self.terms[1:]))
        # Match the conventional entries[menu][1] discount field.
        self.entries = (None, *((f"region_{region}_delay_{delay}h", rate, 1.)
                                for region, delay, rate in self.terms[1:]))

    @staticmethod
    def _index(index):
        value = np.asarray(index)
        if value.size != 1:
            raise ValueError("One scalar posted-offer action is required")
        scalar = value.item()
        if not np.isfinite(scalar) or int(scalar) != scalar or not 0 <= scalar < 61:
            raise ValueError("Posted-offer action must be an integer in [0,61)")
        return int(scalar)

    def describe(self, index):
        return self.terms[self._index(index)]

    def encode(self, region, delay_hours, rate):
        try:
            region_index = self.regions.index(region)
            delay_index = self.delays_hours.index(delay_hours)
            rate_index = self.rates.index(rate)
        except ValueError as exc:
            raise ValueError("The route/delay/rebate is not a disclosed posted action") from exc
        return 1 + (region_index * len(self.delays_hours) + delay_index) * len(self.rates) + rate_index

    def decode(self, episode, index):
        # Decode has no state-dependent routing or price substitution.
        return np.asarray(self._index(index), dtype=np.int64)

    def sample(self, episode, rng):
        """Random legal offers for user learning; retain a 5% no-offer rate."""
        valid = []
        for index in range(1, self.n_actions):
            preview = episode.preview_offer(index)
            if preview.get("admissible", False):
                valid.append(index)
        if not valid or rng.random() < .05:
            return np.asarray(0, dtype=np.int64)
        return np.asarray(rng.choice(valid), dtype=np.int64)


HISTORY_FIELDS = (
    "observed", "offered", "accepted", "region_0", "region_1", "region_2",
    "delay_3h", "additional_delay_3h", "additional_migrated", "rebate_rate_2pct",
    "rebate_yuan_per_gpuh", "resource_saving_yuan_per_gpuh",
    "task_position_fraction", "start_fraction",
)


def feature_names():
    names = []
    for prefix in ("current_energy_yuan_per_gpuh", "current_capacity_fraction",
                   "renewable_available_mw", "renewable_curtailed_mw",
                   "current_carbon_t_per_mwh"):
        names.extend(f"{prefix}_r{region}_b{block}" for region in range(3) for block in range(4))
    names.extend(f"site_capacity_thousand_gpu_{region}" for region in range(3))
    names.extend(f"line_mean_utilization_{line}" for line in range(3))
    names.extend(f"line_peak_utilization_{line}" for line in range(3))
    for prefix in ("current_owner", "current_home", "outside_region"):
        names.extend(f"{prefix}_{owner}" for owner in range(3))
    names.extend(("task_work_thousand_gpuh", "task_release_fraction", "task_deadline_fraction",
                  "outside_start_fraction", "outside_delay_3h", "task_duration_4h",
                  "outside_invoice_yuan_per_gpuh", "task_position_fraction",
                  "remaining_job_fraction", "owner_remaining_job_fraction"))
    names.extend(("carbon_budget_thousand_t", "current_grid_emissions_thousand_t",
                  "remaining_carbon_thousand_t", "policy_carbon_price_thousand_yuan_per_t",
                  "cap_shadow_thousand_yuan_per_t", "current_resource_cost_yuan_per_gpuh",
                  "current_dc_electricity_yuan_per_gpuh", "current_platform_cash_profit_yuan_per_gpuh",
                  "resource_share", "day_work_thousand_gpuh"))
    names.extend(f"remaining_owner{owner}_work_thousand_gpuh_r{region}_b{block}"
                 for owner in range(3) for region in range(3) for block in range(4))
    names.extend(f"remaining_owner_job_fraction_{owner}" for owner in range(3))
    names.extend(f"history_owner{owner}_slot{slot}_{field}"
                 for owner in range(3) for slot in range(4) for field in HISTORY_FIELDS)
    names.extend(("candidate_exit", "candidate_region_0", "candidate_region_1", "candidate_region_2",
                  "candidate_delay_3h", "candidate_rebate_rate_2pct", "candidate_route_exists",
                  "candidate_changed_execution", "candidate_physically_feasible", "candidate_start_fraction",
                  "candidate_additional_delay_3h", "candidate_additional_migrated",
                  "candidate_resource_saving_yuan_per_gpuh", "candidate_platform_bill_saving_yuan_per_gpuh",
                  "candidate_rebate_yuan_per_gpuh", "candidate_net_surplus_yuan_per_gpuh",
                  "candidate_platform_gain_yuan_per_gpuh", "candidate_buyer_gain_yuan_per_gpuh",
                  "candidate_settlement_yuan_per_gpuh", "candidate_emission_reduction_kg_per_gpuh",
                  "candidate_dc_energy_delta_kwh_per_gpuh", "candidate_operations_delta_yuan_per_gpuh",
                  "candidate_transfer_gb_per_gpuh", "candidate_peak_compute_fraction",
                  "candidate_peak_line_fraction", "offer_admissible"))
    return tuple(names)


posted_offer_feature_names = feature_names


def _onehot(index):
    result = np.zeros(3)
    if index is not None and 0 <= int(index) < 3:
        result[int(index)] = 1.
    return result


def _number(preview, *names):
    for name in names:
        if name in preview and preview[name] is not None:
            return float(preview[name])
    return 0.


def _selected(episode):
    return tuple(int(index) for index in episode.selected)


def _public_context(episode):
    config, scenario, clearing = episode.config, episode.scenario, episode.current_clearing
    position = int(episode.position)
    if position >= len(episode.jobs):
        raise ValueError("There is no current task after all posted offers are resolved")
    job = episode.jobs[position]
    outside = job.options[0]
    indices = _selected(episode)
    allocation = sum((other.options[index].allocation_cu for other, index in zip(episode.jobs, indices)),
                     np.zeros((config.n_steps, 3)))
    load = scenario.load_for(indices)
    blocks = np.minimum(np.arange(config.n_steps) * config.n_blocks // config.n_steps, config.n_blocks - 1)
    by_block = lambda values: np.column_stack([values[blocks == block].mean(axis=0)
                                               for block in range(config.n_blocks)])
    available, curtailed = np.zeros_like(load), np.zeros_like(load)
    spec = scenario.grid_spec
    for generator, region in enumerate(spec.generator_bus):
        if spec.generator_emissions_t_per_mwh[generator] == 0:
            capacity = spec.generator_capacity_mw[:, generator]
            available[:, region] += capacity
            curtailed[:, region] += np.maximum(capacity - clearing.generation_mw[:, generator], 0.)
    intensity = nodal_carbon_intensity(spec, clearing, load)
    utilization = np.divide(np.abs(clearing.flows_mw), spec.line_capacity_mw,
                            out=np.zeros_like(clearing.flows_mw), where=spec.line_capacity_mw > 0)
    total_work = float(sum(other.work_cu for other in episode.jobs))
    denominator = max(total_work, 1e-12)
    operations = float(sum(other.options[index].operating_cost_yuan
                           for other, index in zip(episode.jobs, indices)))
    dc_bill = float(np.sum((load - scenario.background_mw) * clearing.lmp_yuan_per_mwh) * config.dt_hours)
    # Public invoices, grid bills, execution costs and signed settlements.
    # Inconvenience and generalized system cost are intentionally absent.
    transfer = float(sum(row.get("settlement_yuan", 0.)
                         for row in getattr(episode, "amendment_ledger", ())))
    current_profit = (float(np.asarray(episode.bills).sum()) - dc_bill - operations
                      - config.daily_fixed_compute_cost_yuan + transfer)
    remaining = episode.jobs[position:]
    remaining_owner = sum(other.user_id == job.user_id for other in remaining)
    context = np.r_[
        by_block(clearing.lmp_yuan_per_mwh * np.asarray(config.energy_mwh_per_cu)).ravel(),
        by_block(allocation / (np.asarray(config.capacity_cu_per_hour) * config.dt_hours)).ravel(),
        by_block(available).ravel(), by_block(curtailed).ravel(), by_block(intensity).ravel(),
        np.asarray(config.capacity_cu_per_hour) / 1000., utilization.mean(axis=0), utilization.max(axis=0),
        _onehot(job.user_id), _onehot(job.home_region), _onehot(outside.region),
        job.work_cu / 1000., job.release_step / config.n_steps, job.deadline_step / config.n_steps,
        outside.start_step / config.n_steps, outside.delay_hours / 3.,
        (outside.end_step - outside.start_step) * config.dt_hours / 4.,
        float(episode.baseline_bills[position]) / job.work_cu,
        position / len(episode.jobs), len(remaining) / len(episode.jobs),
        remaining_owner / config.jobs_per_user,
        (scenario.carbon_budget_t or 0.) / 1000., clearing.emissions_t / 1000.,
        ((scenario.carbon_budget_t or 0.) - clearing.emissions_t) / 1000.,
        spec.carbon_price_yuan_per_t / 1000., clearing.cap_shadow_price_yuan_per_t / 1000.,
        (clearing.total_generation_cost + operations) / denominator, dc_bill / denominator,
        current_profit / denominator, config.resource_share, total_work / 1000.]
    remaining_work = np.zeros((3, 3, 4))
    remaining_count = np.zeros(3)
    for other in remaining:
        # All remaining jobs already have a known outside booking. Its actual
        # allocation, rather than a simulated future user reply, is observable.
        booked = other.options[0]
        remaining_count[other.user_id] += 1
        for block in range(4):
            remaining_work[other.user_id, :, block] += booked.allocation_cu[blocks == block].sum(axis=0) / 1000.
    return np.r_[context, remaining_work.ravel(), remaining_count / config.jobs_per_user], allocation


def _public_history(episode, enabled):
    values = np.zeros((3, 4, len(HISTORY_FIELDS)))
    if not enabled:
        return values.ravel()
    records = getattr(episode, "public_history", getattr(episode, "history", ()))
    owners = [[] for _ in range(3)]
    for record in records:
        owner = int(record["user_id"])
        if not 0 <= owner < 3:
            raise ValueError("Public quote history needs a valid owner")
        owners[owner].append(record)
    for owner, history in enumerate(owners):
        if len(history) > 4:
            raise ValueError("The posted observation supports four previous tasks per owner")
        for slot, record in enumerate(history):
            offered = bool(record.get("offered", False))
            values[owner, slot] = np.r_[
                1., float(offered), float(bool(record.get("accepted", False))),
                _onehot(record.get("region") if offered else None),
                float(record.get("delay_hours") or 0.) / 3.,
                float(record.get("additional_delay_hours") or 0.) / 3.,
                float(record.get("additional_migrated") or 0.),
                float(record.get("rebate_rate") or 0.) / .02,
                float(record.get("rebate_yuan_per_gpuh") or 0.),
                float(record.get("resource_saving_yuan_per_gpuh") or 0.),
                float(record.get("task_position") or 0.) / max(len(episode.jobs), 1),
                float(record.get("start_step") or 0.) / episode.config.n_steps]
    return values.ravel()


def _route_option(episode, preview, terms):
    if terms is None:
        return None
    job = episode.jobs[episode.position]
    option = preview.get("option")
    if option is not None:
        return option
    index = preview.get("option_index", preview.get("index"))
    if index is not None and 0 <= int(index) < len(job.options):
        return job.options[int(index)]
    region, delay, _ = terms
    target_start = job.release_step + int(round(delay / episode.config.dt_hours))
    available = getattr(episode, "option_available", None)
    for candidate_index, candidate in enumerate(job.options):
        if available is not None and not available[episode.position][candidate_index]:
            continue
        if candidate.region == region and candidate.start_step == target_start:
            return candidate
    return None


def candidate_posted_offer_observation(episode, catalogue=None, *, enable_history=True):
    catalogue = catalogue or PostedOfferCatalogue(episode.config)
    context, allocation = _public_context(episode)
    history = _public_history(episode, enable_history)
    config, job = episode.config, episode.jobs[episode.position]
    outside = job.options[0]
    quantity = job.work_cu
    capacity = np.asarray(config.capacity_cu_per_hour) * config.dt_hours
    rows = []
    for action in range(catalogue.n_actions):
        terms = catalogue.describe(action)
        preview = episode.preview_offer(action)
        candidate = np.zeros(25)
        candidate[0] = float(action == 0)
        if action == 0:
            candidate[8] = 1.
            candidate[9] = outside.start_step / config.n_steps
            candidate[23] = float(np.max(allocation / capacity))
            line_limits = episode.scenario.grid_spec.line_capacity_mw
            candidate[24] = float(np.max(np.divide(np.abs(episode.current_clearing.flows_mw),
                line_limits, out=np.zeros_like(episode.current_clearing.flows_mw), where=line_limits > 0)))
        if terms is not None:
            region, delay, rate = terms
            candidate[1:4] = _onehot(region)
            candidate[4:6] = [delay / 3., rate / .02]
            option = _route_option(episode, preview, terms)
            candidate[6] = float(option is not None)
            if option is not None:
                changed = any(not np.array_equal(getattr(option, name), getattr(outside, name))
                              for name in ("allocation_cu", "extra_energy_mwh", "transfer_gb"))
                feasible = bool(preview.get("feasible", False))
                resource = _number(preview, "resource_saving_yuan", "G")
                private = _number(preview, "pre_rebate_private_saving_yuan", "private_pre_rebate_saving_yuan", "a")
                rebate = _number(preview, "additional_rebate_yuan", "R")
                net = _number(preview, "net_surplus_yuan", "M")
                gain = _number(preview, "platform_gain_yuan", "platform_increment_yuan")
                buyer = _number(preview, "buyer_gain_yuan")
                settlement = _number(preview, "settlement_yuan")
                clearing = preview.get("clearing")
                reduction = ((episode.current_clearing.emissions_t - clearing.emissions_t)
                             if feasible and clearing is not None else 0.)
                energy_delta = float((episode.scenario.option_power_mw(option)
                                      - episode.scenario.option_power_mw(outside)).sum() * config.dt_hours)
                occupancy = allocation - outside.allocation_cu + option.allocation_cu
                line_peak = (float(np.max(np.divide(np.abs(clearing.flows_mw), episode.scenario.grid_spec.line_capacity_mw,
                    out=np.zeros_like(clearing.flows_mw), where=episode.scenario.grid_spec.line_capacity_mw > 0)))
                    if feasible and clearing is not None else 0.)
                candidate[7:] = [float(changed), float(feasible), option.start_step / config.n_steps,
                    (option.delay_hours - outside.delay_hours) / 3., float(option.migrated) - float(outside.migrated),
                    resource / quantity, private / quantity, rebate / quantity, net / quantity,
                    gain / quantity, buyer / quantity, settlement / quantity, reduction * 1000. / quantity,
                    energy_delta * 1000. / quantity, (option.operating_cost_yuan - outside.operating_cost_yuan) / quantity,
                    float(option.transfer_gb.sum()) / quantity, float(np.max(occupancy / capacity)), line_peak]
        mask = float(action == 0 or preview.get("admissible", False))
        rows.append(np.r_[context, history, candidate, mask])
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (61, len(feature_names())) or not np.isfinite(result).all():
        raise ValueError("Expected a finite [61,feature] public posted-offer observation")
    return result


class PostedOfferObserver:
    """No episode/draw cache: every state uses its current plan and replies."""

    def __init__(self, catalogue=None, *, enable_history=True):
        self.catalogue = catalogue or PostedOfferCatalogue()
        self.enable_history = bool(enable_history)
        self.feature_names = feature_names()

    def raw(self, episode):
        return candidate_posted_offer_observation(episode, self.catalogue,
                                                 enable_history=self.enable_history)

    def __call__(self, episode, *, training=False):
        return self.raw(episode)

    def fit_normalization(self, episode_factory, seeds):
        rows = np.concatenate([self.raw(episode_factory(int(seed)))[:, :-1] for seed in seeds], axis=0)
        mean = rows.mean(axis=0, dtype=np.float64)
        scale = np.maximum(rows.std(axis=0, dtype=np.float64), .01)
        # A fit from initial training states has no replies or late positions.
        # These already-scaled entries keep their units as the episode evolves.
        fixed = {"task_release_fraction", "task_deadline_fraction", "outside_start_fraction",
                 "outside_delay_3h", "task_duration_4h", "task_position_fraction",
                 "remaining_job_fraction", "owner_remaining_job_fraction"}
        for index, name in enumerate(self.feature_names[:-1]):
            if (name.startswith("history_") or name.startswith("remaining_owner_job_fraction_")
                    or name in fixed):
                mean[index], scale[index] = 0., 1.
        return mean.astype(np.float32), scale.astype(np.float32)
