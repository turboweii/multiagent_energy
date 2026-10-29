"""Accepted outside contracts remain independent of later amendment learning."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from energy_net.coordination.capped_market import capped_episode_factory
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue
from energy_net.coordination.learning import PolicyBundle
from energy_net.coordination.outside_bookings import (
    OutsideBooking, make_outside_booking, outside_booking_factory,
)


class FixedQuote:
    def __init__(self, index):
        self.index = index
        self.calls = 0

    def predict(self, observation, deterministic=True):
        assert deterministic
        self.calls += 1
        return np.asarray(self.index), None


class OwnCostUser:
    """A deterministic local-choice fixture, not a trained research result."""
    def __init__(self):
        self.calls = 0
        self.policy = torch.nn.Linear(1, 1)

    def predict(self, observation, deterministic=True):
        assert deterministic
        self.calls += 1
        options = observation[6:76].reshape(10, 7)
        costs = options[:, 2] + options[:, 3]
        possible = (options[:, 0] > .5) & (options[:, 1] > .5)
        action = int(np.argmin(np.where(possible, costs, np.inf)))
        return np.asarray(action), None


def legacy_bundle(menu=65):
    factory = capped_episode_factory()
    bundle = PolicyBundle(platform=FixedQuote(menu),
        users={u: OwnCostUser() for u in range(3)},
        platform_codec=FixedGpuTariffCatalogue(factory.config), training=True,
        sample_opponents=True, opponent_sampling_probability=1., price_exploration_probability=1.)
    return factory, bundle


def test_booking_is_one_frozen_policy_rollout_with_original_reference_retained():
    factory, bundle = legacy_bundle()
    booking = make_outside_booking(4, bundle, factory, "training_selected_menu_65")
    assert bundle.training
    assert bundle.platform.calls == 1
    assert sum(u.calls for u in bundle.users.values()) == 12
    assert len(booking.selected_options) == len(booking.scenario.jobs) == 12
    assert booking.metrics["completion_rate"] == 1 and booking.metrics["deadline_compliance"] == 1
    assert booking.metrics["user_ir_rate"] == 1
    assert booking.metrics["flexibility_settlement_yuan"] == 0
    assert booking.metrics["carbon_emissions_t"] <= booking.scenario.carbon_budget_t + 1e-6
    assert booking.platform_policy_label == "training_selected_menu_65"
    assert sum(booking.bills_yuan) == pytest.approx(booking.metrics["service_revenue_yuan"])
    assert sum(booking.user_losses_yuan) == pytest.approx(booking.metrics["user_loss_yuan"])
    for job, selected, loss in zip(booking.scenario.jobs, booking.selected_options, booking.user_losses_yuan):
        assert loss == pytest.approx(job.options[selected].user_loss_yuan)
    assert booking.baseline_metrics["user_loss_yuan"] == 0
    assert booking.original_metrics == booking.baseline_metrics
    with pytest.raises(FrozenInstanceError):
        booking.platform_policy_label = "a different policy"


def test_old_original_plan_booking_preserves_work_and_has_zero_original_changes():
    factory, bundle = legacy_bundle(0)
    booking = make_outside_booking(1, bundle, factory)
    assert booking.selected_options == (0,) * 12
    assert booking.metrics["delta_platform_profit_yuan"] == pytest.approx(0)
    assert booking.metrics["delta_system_cost_yuan"] == pytest.approx(0)
    assert booking.scenario.total_work_cu == booking.metrics["required_work_cu"]


def test_public_booking_accounts_exclude_private_inconvenience_and_user_utilities():
    factory, bundle = legacy_bundle()
    booking = make_outside_booking(2, bundle, factory)
    public = booking.public_metrics()
    assert public["platform_profit_yuan"] == booking.metrics["platform_profit_yuan"]
    assert public["resource_cost_yuan"] == booking.metrics["resource_cost_yuan"]
    assert not any("user_" in name or "loss" in name or "system_cost" in name for name in public)
    assert "carbon_budget_t" in public and "service_revenue_yuan" in public


def test_cache_returns_independent_accounts_dispatch_and_scenario_scalars():
    factory, bundle = legacy_bundle()
    cached = outside_booking_factory(bundle, factory, maxsize=2)
    first = cached(3)
    expected = first.metrics["platform_profit_yuan"]
    cap = first.scenario.carbon_budget_t
    first.metrics["platform_profit_yuan"] = -123456789
    first.baseline_metrics["system_cost_yuan"] = -999
    first.clearing.generation_mw[0, 0] = -1234
    first.scenario.carbon_budget_t = 0
    first.scenario.calibration["common_carbon_cap_t"] = 0
    second = cached(3)
    assert second.metrics["platform_profit_yuan"] == expected
    assert second.baseline_metrics["system_cost_yuan"] > 0
    assert second.clearing.generation_mw[0, 0] >= 0
    assert second.scenario.carbon_budget_t == cap
    assert second.scenario.calibration["common_carbon_cap_t"] == cap
    assert first is not second and first.scenario is not second.scenario
    assert bundle.platform.calls == 1 and cached.cache_info().hits == 1
    with pytest.raises(ValueError):
        second.scenario.background_mw[0, 0] = 0
    with pytest.raises(ValueError):
        second.scenario.jobs[0].options[0].allocation_cu[0, 0] = 123


def test_updating_or_replacing_legacy_models_invalidates_the_booking_cache():
    factory, bundle = legacy_bundle()
    cached = outside_booking_factory(bundle, factory)
    cached(0)
    with torch.no_grad():
        bundle.users[0].policy.weight.add_(1)
    with pytest.raises(RuntimeError, match="Legacy models changed"):
        cached(0)
    other_factory, other_bundle = legacy_bundle()
    replacement_cache = outside_booking_factory(other_bundle, other_factory)
    other_bundle.platform = FixedQuote(1)
    with pytest.raises(RuntimeError, match="Legacy models changed"):
        replacement_cache(0)


def test_incomplete_or_nonlegacy_episodes_are_not_outside_bookings():
    factory, _ = legacy_bundle()
    with pytest.raises(ValueError, match="completely accepted"):
        OutsideBooking.from_episode(factory(0))


def test_training_mode_is_restored_even_when_a_selected_policy_raises():
    class BrokenQuote(FixedQuote):
        def predict(self, *args, **kwargs):
            raise RuntimeError("selected quote unavailable")
    factory, bundle = legacy_bundle()
    bundle.platform = BrokenQuote(0)
    with pytest.raises(RuntimeError, match="selected quote unavailable"):
        make_outside_booking(0, bundle, factory)
    assert bundle.training
