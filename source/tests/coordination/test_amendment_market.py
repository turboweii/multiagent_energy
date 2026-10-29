"""Two-stage contract accounting and pre-binding admission, with real OPF days."""
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from energy_net.coordination.amendment_market import (
    AmendmentConfig, AmendmentEpisode, AmendmentPlatformEnv,
    amendment_episode_factory, option_physics_key,
)
from energy_net.coordination.capped_market import capped_episode_factory
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue
from energy_net.coordination.learning import PolicyBundle
from energy_net.coordination.market import encode_platform_action, exact_user_policy
from energy_net.coordination.outside_bookings import OutsideBooking


@lru_cache(maxsize=8)
def booked_day(seed=0, old_menu=7):
    # Exact local choice is a unit-test fixture, not a learned evaluation arm.
    legacy = capped_episode_factory()(seed)
    menu = FixedGpuTariffCatalogue(legacy.config)
    legacy.set_platform_action(menu.decode(legacy, old_menu))
    legacy.play()
    return OutsideBooking.from_episode(legacy)


def episode(order="net", seed=0, old_menu=7, share=.8):
    booking = booked_day(seed, old_menu)
    return AmendmentEpisode(booking.scenario, booking,
        config=AmendmentConfig(settlement_order=order, resource_share=share))


def open_quote(ep, discount=.005):
    raw = encode_platform_action(np.full((3, 4), discount), config=ep.config)
    ep.set_platform_action(np.r_[raw, 1])


def test_exit_preserves_already_booked_bills_private_losses_and_profit():
    ep = episode()
    booking = ep.booking
    assert sum(booking.user_losses_yuan) > 0
    assert any(index > 1 for index in booking.selected_options)
    assert ep.baseline_metrics["platform_profit_yuan"] == pytest.approx(
        booking.metrics["platform_profit_yuan"])
    result = ep.play()
    assert result.selected_options == (0,) * len(ep.jobs)
    assert result.bills_yuan == booking.bills_yuan
    assert result.user_losses_yuan == booking.user_losses_yuan
    for name in ("platform_gain_vs_outside_yuan", "user_gain_yuan",
                 "social_saving_yuan", "flexibility_settlement_yuan",
                 "additional_user_rebate_yuan"):
        assert result.metrics[name] == pytest.approx(0, abs=1e-6)
    assert result.metrics["platform_floor_satisfied"]
    assert result.metrics["outside_user_loss_yuan"] > 0
    assert result.metrics["delta_system_cost_yuan_vs_original"] == pytest.approx(
        booking.metrics["system_cost_yuan"] - booking.original_metrics["system_cost_yuan"])
    ep.reset()
    np.testing.assert_array_equal(ep.losses, booking.user_losses_yuan)


def test_outside_is_physical_index_zero_with_unique_options_and_padding_mask():
    ep = episode()
    original_options = tuple(tuple(job.options) for job in ep.original_scenario.jobs)
    for position, job in enumerate(ep.jobs):
        outside = original_options[position][ep.outside_selected_options[position]]
        assert option_physics_key(job.options[0]) == option_physics_key(outside)
        keys = [option_physics_key(option) for option, available in
                zip(job.options, ep.option_available[position]) if available]
        assert len(keys) == len(set(keys))
        assert len(job.options) == 10
        assert job.options[0].is_baseline
        assert all(not option.is_baseline for option in job.options[1:])
        assert sum(ep.option_available[position]) == 9
    assert ep.original_scenario.jobs[1].options == original_options[1]
    assert ep.outside_selected_options[1] == 6
    assert ep.option_source_indices[1][1] == 0  # Returning home is a real amendment.


def test_closed_round_open_zero_rebate_and_unchanged_execution_have_distinct_meaning():
    ep = episode()
    assert ep.preview_amendment(1)["reason"] == "amendment_round_closed"
    _, _, rejected = ep.step_user(1)
    assert rejected["chosen"] == 0 and not rejected["amendment_accepted"]
    ep.reset()
    raw = encode_platform_action(np.zeros((3, 4)), config=ep.config)
    ep.set_platform_action(np.r_[raw, 1])
    assert ep.amendment_open and ep.platform_action.shape == (16,)
    assert ep.preview_amendment(1)["reason"] != "amendment_round_closed"
    assert ep.option_bill(ep.jobs[0], ep.jobs[0].options[0]) == ep.baseline_bills[0]
    assert ep.preview_amendment(9)["reason"] == "padding_option"
    ep.set_platform_action(raw)
    assert not ep.amendment_open
    open_quote(ep)
    assert ep.amendment_open


@pytest.mark.parametrize("order", ["net", "gross", "direct"])
@pytest.mark.parametrize("seed", [0, 1, 3])
def test_real_sequential_clearing_preserves_cap_contracts_and_money_identities(order, seed):
    ep = episode(order, seed=seed)
    open_quote(ep)
    result = ep.play()
    m = result.metrics
    assert m["completion_rate"] == pytest.approx(1)
    assert m["deadline_compliance"] == m["user_ir_rate"] == 1
    assert m["carbon_budget_t"] == ep.booking.scenario.carbon_budget_t
    assert m["carbon_emissions_t"] <= ep.booking.scenario.carbon_budget_t + 1e-6
    assert m["platform_floor_satisfied"]
    assert m["platform_gain_vs_outside_yuan"] >= -1e-6
    assert m["platform_profit_identity_error_yuan"] == pytest.approx(0, abs=1e-6)
    assert m["surplus_identity_error_yuan"] == pytest.approx(0, abs=1e-6)
    assert m["social_saving_yuan"] == pytest.approx(
        m["contract_system_cost_saving_yuan"], abs=1e-6)
    assert m["flexibility_resource_saving_yuan"] == pytest.approx(
        sum(row["resource_saving_yuan"] for row in ep.amendment_ledger), abs=1e-6)
    assert m["platform_pre_rebate_saving_yuan"] == pytest.approx(
        sum(row["pre_rebate_private_saving_yuan"] for row in ep.amendment_ledger), abs=1e-6)
    assert m["user_rebate_yuan"] == pytest.approx(
        sum(row["additional_rebate_yuan"] for row in ep.amendment_ledger), abs=1e-6)
    assert m["outside_platform_profit_yuan"] == pytest.approx(
        ep.booking.metrics["platform_profit_yuan"])
    if order != "direct":
        assert m["buyer_post_settlement_benefit_yuan"] >= -1e-6
        assert m["social_saving_yuan"] >= -1e-6
    else:
        assert m["flexibility_settlement_yuan"] == pytest.approx(0, abs=1e-6)
    assert ep.platform_reward == pytest.approx(m["platform_gain_vs_outside_yuan"] / 1000)
    assert ep.outcome().metrics == result.metrics  # Reading settlement never pays twice.


def algebra_case(monkeypatch, order, resource=100., rebate=90., private=120., share=.8):
    """Control scalar cost accounts to isolate formulas from the real-OPF tests."""
    ep = episode(order, old_menu=0, share=share)
    open_quote(ep)
    index = 1
    options = list(ep.jobs[0].options)
    options[index] = replace(options[index], user_loss_yuan=ep.outside_losses[0])
    ep.jobs = (replace(ep.jobs[0], options=tuple(options)), *ep.jobs[1:])
    ep.scenario = replace(ep.scenario, jobs=ep.jobs)
    target = [0] * len(ep.jobs); target[0] = index
    old_ops, new_ops = ep._operations(ep.selected), ep._operations(target)
    ep._cache[tuple(target)] = replace(ep.current_clearing,
        total_generation_cost=ep.current_clearing.total_generation_cost + old_ops - new_ops - resource)
    monkeypatch.setattr(ep, "_precheck", lambda indices: (True, "accepted"))
    monkeypatch.setattr(ep, "_energy_bill", lambda clearing, indices:
        10000. if not any(indices) else 10000. + old_ops - new_ops - private)
    monkeypatch.setattr(ep, "option_bill", lambda job, option:
        float(ep.baseline_bills[0] - rebate) if option.index == index else float(ep.baseline_bills[0]))
    return ep, index


def test_net_first_can_finance_rebate_that_gross_first_cannot(monkeypatch):
    net, index = algebra_case(monkeypatch, "net")
    p = net.preview_amendment(index)
    assert p["admissible"]
    assert p["net_surplus_yuan"] == pytest.approx(10)
    assert p["platform_gain_yuan"] == pytest.approx(8)
    assert p["buyer_gain_yuan"] == pytest.approx(2)
    assert p["settlement_yuan"] == pytest.approx(-22)
    reward, _, info = net.step_user(index)
    assert info["amendment_accepted"] and reward > 0
    gross, index = algebra_case(monkeypatch, "gross")
    before = gross.current_clearing
    p = gross.preview_amendment(index)
    assert not p["admissible"] and p["platform_gain_yuan"] == pytest.approx(-10)
    reward, _, info = gross.step_user(index)
    assert reward == 0 and info["chosen"] == 0
    assert gross.current_clearing is before
    assert gross.selected == [0] * len(gross.jobs)
    assert gross.amendment_ledger == []
    np.testing.assert_array_equal(gross.bills, gross.baseline_bills)


def test_direct_institution_uses_private_profit_and_records_negative_buyer_residual(monkeypatch):
    ep, index = algebra_case(monkeypatch, "direct", resource=-100, rebate=50, private=150)
    p = ep.preview_amendment(index)
    assert p["admissible"] and p["net_surplus_yuan"] == pytest.approx(-150)
    assert p["platform_gain_yuan"] == pytest.approx(100)
    assert p["buyer_gain_yuan"] == pytest.approx(-250)
    assert p["settlement_yuan"] == pytest.approx(0)


@pytest.mark.parametrize("order", ["net", "gross"])
def test_zero_platform_share_does_not_allow_unfunded_user_rebates(monkeypatch, order):
    ep, index = algebra_case(monkeypatch, order, resource=20, rebate=30, share=0)
    assert not ep.preview_amendment(index)["admissible"]


def test_public_buyer_preview_does_not_read_private_inconvenience(monkeypatch):
    ep, index = algebra_case(monkeypatch, "net")
    first = ep.preview_amendment(index)
    options = list(ep.jobs[0].options)
    options[index] = replace(options[index], user_loss_yuan=1e9)
    ep.jobs = (replace(ep.jobs[0], options=tuple(options), beta_yuan_per_cu_hour=999), *ep.jobs[1:])
    ep.scenario = replace(ep.scenario, jobs=ep.jobs)
    second = ep.preview_amendment(index)
    assert first == second
    features = ep.current_user_observation()[6:76].reshape(10, 7)
    assert features[index, 1] == 1  # Public availability excludes private IR.
    reward, _, info = ep.step_user(index)
    assert reward == 0 and info["reservation_reason"] == "user_reservation_utility"
    assert ep.amendment_ledger == []


def test_user_observation_uses_outside_differences_and_exact_public_admission(monkeypatch):
    ep, index = algebra_case(monkeypatch, "net", resource=20, rebate=30)
    obs = ep.current_user_observation()
    assert obs.shape == (92,) and obs.dtype == np.float32
    features = obs[6:76].reshape(10, 7)
    assert features[0, 2] == features[0, 3] == features[0, 4] == 0
    assert features[index, 0] == 1 and features[index, 1] == 0
    assert not ep.preview_amendment(index)["admissible"]
    booked = episode()
    open_quote(booked, 0.)
    booked.position = 1
    f = booked.current_user_observation()[6:76].reshape(10, 7)
    return_home = booked.option_source_indices[1].index(0)
    assert f[return_home, 3] < 0  # Undoing booked inconvenience can benefit the user.


def test_factory_reuses_source_and_opf_but_not_live_state_or_booked_accounts():
    calls = []
    booking = booked_day()
    def booking_factory(seed):
        calls.append(seed)
        return booking.copy()
    factory = amendment_episode_factory(lambda seed: booking.scenario,
        booking_factory, AmendmentConfig(), maxsize=2)
    first, second = factory(0), factory(0)
    assert calls == [0]
    assert first is not second and first.grid is second.grid
    assert first.original_scenario is second.original_scenario
    first.bills[0] = 0
    first.losses[0] = 1234
    first.selected[0] = 1
    assert second.bills[0] == booking.bills_yuan[0]
    assert second.losses[0] == booking.user_losses_yuan[0]
    assert second.selected[0] == 0
    assert factory.config.settlement_order == "net"


def test_platform_environment_rewards_actual_increment_above_outside_profit():
    from energy_net.coordination.amendment_features import AmendmentTariffCatalogue
    booking = booked_day()
    factory = amendment_episode_factory(lambda seed: booking.scenario,
        lambda seed: booking.copy(), AmendmentConfig())
    codec = AmendmentTariffCatalogue(factory.config)
    class TestLocalCostUser:
        def predict(self, observation, deterministic=True):
            return np.asarray(exact_user_policy(observation)), None
    bundle = PolicyBundle(users={user: TestLocalCostUser() for user in range(3)},
        platform_codec=codec)
    env = AmendmentPlatformEnv(factory, bundle, "platform", scenario_seeds=[0])
    env.reset(seed=0)
    _, reward, terminated, _, info = env.step(3)
    assert terminated
    assert reward == pytest.approx(info["market_metrics"]["platform_gain_vs_outside_yuan"] / 1000)
    assert env.action_space.n == 78


def test_invalid_booking_cap_or_nonzero_internal_carbon_objective_is_rejected():
    booking = booked_day()
    with pytest.raises(ValueError, match="same day and physical carbon cap"):
        AmendmentEpisode(replace(booking.scenario, carbon_budget_t=booking.scenario.carbon_budget_t - 1), booking)
    with pytest.raises(ValueError, match="internal carbon value zero"):
        AmendmentConfig(internal_carbon_value_yuan_per_t=100)
