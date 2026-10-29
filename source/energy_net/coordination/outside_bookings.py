"""Accepted legacy bookings used as the amendment stage's genuine fallback.

The legacy policy and its user policies are selected on training/validation
days and frozen before this stage. One deterministic rollout forms accepted,
not-yet-executed contracts. The function does not search test-day alternatives.
Private inconvenience stays in the environment/user accounts, not in the
public booking record made available to the new platform.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from .capped_market import capped_episode_factory


def _copy_clearing(clearing):
    return replace(clearing,
        generation_mw=clearing.generation_mw.copy(),
        flows_mw=clearing.flows_mw.copy(),
        lmp_yuan_per_mwh=clearing.lmp_yuan_per_mwh.copy())


def _scenario_snapshot(scenario):
    # Scalar fields and calibration are independent; the large physical input
    # arrays can be shared because accepted outside contracts never mutate them.
    arrays = [scenario.background_mw, scenario.renewable_mw]
    arrays.extend(value for value in vars(scenario.grid_spec).values()
                  if isinstance(value, np.ndarray))
    for job in scenario.jobs:
        for option in job.options:
            arrays.extend((option.allocation_cu, option.extra_energy_mwh, option.transfer_gb))
    for values in arrays:
        values.setflags(write=False)
    return replace(scenario, calibration=deepcopy(scenario.calibration))


@dataclass(frozen=True)
class OutsideBooking:
    """Frozen booking fields; mutable account/dispatch values are copied.

    ``scenario`` retains the original jobs and original common carbon cap.
    ``selected_options`` index those original jobs. ``baseline_metrics`` is
    the original no-shift account; ``metrics`` is the actual accepted legacy
    plan and includes absolute user inconvenience for private comparisons.
    """
    scenario: object
    selected_options: tuple
    bills_yuan: tuple
    user_losses_yuan: tuple
    clearing: object
    metrics: dict
    baseline_metrics: dict
    platform_action: tuple
    platform_policy_label: str = "legacy_selected"
    reservations: tuple = ()

    def __post_init__(self):
        n = len(self.scenario.jobs)
        selections = tuple(int(i) for i in self.selected_options)
        bills = tuple(float(value) for value in self.bills_yuan)
        losses = tuple(float(value) for value in self.user_losses_yuan)
        action = tuple(int(value) for value in self.platform_action)
        if len(selections) != n or len(bills) != n or len(losses) != n:
            raise ValueError("Every original job needs one accepted outside contract")
        if any(not 0 <= index < len(job.options)
               for job, index in zip(self.scenario.jobs, selections)):
            raise ValueError("Outside choices must index the original execution options")
        if not np.isfinite(bills).all() or np.any(np.asarray(bills) < 0):
            raise ValueError("Outside invoices must be finite and nonnegative")
        if not np.isfinite(losses).all():
            raise ValueError("Outside inconvenience values must be finite")
        if not self.clearing.feasible:
            raise ValueError("Accepted outside bookings require a feasible complete dispatch")
        cap = self.scenario.carbon_budget_t
        if cap is None or self.clearing.emissions_t > cap + 1e-6:
            raise ValueError("Outside bookings must use the common physical carbon cap")
        if len(action) != len(self.scenario.config.platform_nvec):
            raise ValueError("Record the complete original raw platform quote")
        object.__setattr__(self, "selected_options", selections)
        object.__setattr__(self, "scenario", _scenario_snapshot(self.scenario))
        object.__setattr__(self, "bills_yuan", bills)
        object.__setattr__(self, "user_losses_yuan", losses)
        object.__setattr__(self, "platform_action", action)
        object.__setattr__(self, "metrics", deepcopy(dict(self.metrics)))
        object.__setattr__(self, "baseline_metrics", deepcopy(dict(self.baseline_metrics)))
        object.__setattr__(self, "clearing", _copy_clearing(self.clearing))
        object.__setattr__(self, "reservations", tuple(deepcopy(self.reservations)))

    @property
    def original_metrics(self):
        """Original no-shift reference, not the accepted legacy plan."""
        return deepcopy(self.baseline_metrics)

    def copy(self):
        """Independent dispatch/account copies, same original scenario inputs."""
        return replace(self)

    def public_metrics(self):
        """Observable booking/resource accounts without private user values."""
        names = (
            "seed", "jobs", "work_cu", "required_work_cu", "work_gpu_hours",
            "service_revenue_yuan", "electricity_bill_yuan", "compute_migration_cost_yuan",
            "generation_cost_yuan", "resource_cost_yuan", "platform_profit_yuan",
            "platform_profit_before_fixed_cost_yuan", "fixed_compute_cost_yuan",
            "carbon_emissions_t", "carbon_budget_t", "policy_carbon_price_yuan_per_t",
            "cap_shadow_price_yuan_per_t", "dc_energy_mwh", "grid_energy_mwh",
        )
        return {name: deepcopy(self.metrics[name]) for name in names if name in self.metrics}

    @classmethod
    def from_episode(cls, episode, policy_label="legacy_selected"):
        if not episode.done:
            raise ValueError("Outside contracts must be completely accepted before amendments")
        if episode.config.sharing_enabled or episode.config.carbon_eligibility:
            raise ValueError("The outside booking must use legacy cash settlement and the common cap")
        outcome = episode.outcome()
        return cls(episode.scenario, tuple(outcome.selected_options),
            tuple(outcome.bills_yuan), tuple(outcome.user_losses_yuan), outcome.clearing,
            outcome.metrics, episode.baseline_metrics, tuple(episode.platform_action),
            str(policy_label), tuple(outcome.reservations))


def make_outside_booking(seed, legacy_bundle, capped_factory=None,
                         policy_label="legacy_selected"):
    """Execute the already selected legacy policies once, without quote search."""
    factory = capped_factory or capped_episode_factory()
    episode = factory(int(seed))
    previous = legacy_bundle.training
    legacy_bundle.training = False
    try:
        episode.reset()
        episode.set_platform_action(legacy_bundle.platform_action(episode))
        while not episode.done:
            episode.step_user(legacy_bundle.user_action(episode))
        return OutsideBooking.from_episode(episode, policy_label)
    finally:
        legacy_bundle.training = previous


def _model_fingerprint(model):
    if model is None:
        return None
    policy = getattr(model, "policy", None)
    params = (() if policy is None else
              tuple((name, parameter._version) for name, parameter in policy.named_parameters()))
    return (id(model), getattr(model, "num_timesteps", None),
            getattr(model, "_n_updates", None), params)


def outside_booking_factory(legacy_bundle, capped_factory=None, maxsize=64,
                            policy_label="legacy_selected"):
    """Cache accepted bookings of fixed old models, never new joint models.

    Every call returns a copy of mutable dispatch/accounts. The old policies
    must remain frozen: training or replacing one invalidates this factory
    explicitly instead of silently changing the platform's participation floor.
    """
    if int(maxsize) != maxsize or maxsize < 1:
        raise ValueError("The outside cache must hold at least one day")
    factory = capped_factory or capped_episode_factory()

    def fingerprint():
        return (_model_fingerprint(legacy_bundle.platform),
                tuple((owner, _model_fingerprint(model))
                      for owner, model in sorted(legacy_bundle.users.items())),
                id(legacy_bundle.platform_codec), id(legacy_bundle.platform_observer))
    frozen_fingerprint = fingerprint()

    @lru_cache(maxsize=int(maxsize))
    def booking(seed):
        return make_outside_booking(int(seed), legacy_bundle, factory, policy_label)

    def get(seed):
        if fingerprint() != frozen_fingerprint:
            raise RuntimeError("Legacy models changed after the outside-booking factory was frozen")
        return booking(int(seed)).copy()

    get.cache_info = booking.cache_info
    get.cache_clear = booking.cache_clear
    get.capped_factory = factory
    return get
