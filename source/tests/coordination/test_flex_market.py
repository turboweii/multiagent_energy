"""Settlement identities plus real complete-work contract executions."""
from dataclasses import replace

import numpy as np
import pytest

from energy_net.coordination.flex_market import (
    FlexMarketConfig, FlexMarketEpisode, FlexPlatformEnv, flex_episode_factory,
)
from energy_net.coordination.gpu_market import GpuMarketEpisode
from energy_net.coordination.gpu_scenario import make_gpu_scenario
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue, GpuTariffObserver
from energy_net.coordination.learning import PolicyBundle
from energy_net.coordination.market import encode_platform_action


def complete_episode(seed=0, config=None, discount=.005):
    episode = FlexMarketEpisode(make_gpu_scenario(seed, config or FlexMarketConfig()))
    discounts = np.array([[0, discount, discount, discount],
                          [discount, 0, discount, discount],
                          [discount, discount, 0, discount]])
    episode.set_platform_action(encode_platform_action(discounts, config=episode.config))
    return episode


def controlled_accounts(*, resource_saving, private_saving, rebate=200., loss=100.,
                        emissions_change=0., share=.8, sharing=True, gate=True):
    """Algebra fixture; altered result accounts are not physical test outcomes."""
    config = FlexMarketConfig(resource_share=share, sharing_enabled=sharing, carbon_eligibility=gate)
    episode = FlexMarketEpisode(make_gpu_scenario(0, config))
    baseline = episode.baseline_clearing
    reference_bill = episode.baseline_metrics["electricity_bill_yuan"]
    clearing = replace(
        baseline,
        total_generation_cost=baseline.total_generation_cost - resource_saving,
        lmp_yuan_per_mwh=baseline.lmp_yuan_per_mwh * ((reference_bill - private_saving) / reference_bill),
        emissions_t=baseline.emissions_t + emissions_change,
    )
    bills = episode.baseline_bills.copy()
    bills[0] -= rebate
    losses = np.zeros(len(episode.jobs))
    losses[0] = loss
    return episode, episode._metrics(clearing, bills, losses)


@pytest.mark.parametrize("share", [0., .8, 1.])
@pytest.mark.parametrize("resource_saving", [-1000., 0., 1000.])
@pytest.mark.parametrize("private_saving", [-3000., 2000.])
@pytest.mark.parametrize("emissions_change", [-1., 0., 1.])
def test_shared_settlement_identity_and_buyer_budget(share, resource_saving, private_saving, emissions_change):
    _, metrics = controlled_accounts(resource_saving=resource_saving, private_saving=private_saving,
                                     emissions_change=emissions_change, share=share)
    phi = min(resource_saving, 0.)
    if resource_saving > 0 and emissions_change <= 0:
        phi = share * resource_saving
    assert metrics["flexibility_resource_saving_yuan"] == pytest.approx(resource_saving)
    assert metrics["platform_pre_rebate_saving_yuan"] == pytest.approx(private_saving)
    assert metrics["shared_resource_entitlement_yuan"] == pytest.approx(phi)
    assert metrics["flexibility_settlement_yuan"] == pytest.approx(phi - private_saving)
    assert metrics["cash_change_after_transfer_yuan"] == pytest.approx(phi - 200)
    assert metrics["buyer_post_settlement_benefit_yuan"] == pytest.approx(resource_saving - phi)
    assert metrics["buyer_post_settlement_benefit_yuan"] >= -1e-7
    assert metrics["buyer_payment_to_platform_yuan"] >= 0
    assert metrics["platform_payment_to_buyer_yuan"] >= 0
    assert metrics["buyer_payment_to_platform_yuan"] - metrics["platform_payment_to_buyer_yuan"] == pytest.approx(
        metrics["flexibility_settlement_yuan"])
    assert metrics["surplus_allocation_sum_yuan"] == pytest.approx(resource_saving - 100)
    assert abs(metrics["surplus_identity_error_yuan"]) < 1e-7


def test_program_does_not_reward_unchanged_rebooking():
    episode = complete_episode()
    episode.set_platform_action(encode_platform_action(np.full((3, 4), .02), config=episode.config))
    job = episode.jobs[0]
    original_bill = job.work_cu * episode.config.base_price_yuan_per_cu
    assert episode.option_bill(job, job.options[0]) == pytest.approx(original_bill)
    assert episode.option_bill(job, job.options[1]) == pytest.approx(original_bill)
    assert episode.option_bill(job, job.options[2]) == pytest.approx(.98 * original_bill)
    # Executing all unchanged rebookings gets no user discount or procurement payment.
    while not episode.done:
        episode.step_user(1)
    result = episode.outcome()
    assert result.metrics["user_rebate_yuan"] == pytest.approx(0)
    assert result.metrics["flexibility_resource_saving_yuan"] == pytest.approx(0)
    assert result.metrics["flexibility_settlement_yuan"] == pytest.approx(0)
    assert result.metrics["delta_platform_profit_yuan"] == pytest.approx(0)


def test_unconditional_contract_ablation_retains_original_posted_price_behavior():
    config = FlexMarketConfig(conditional_rebates=False, sharing_enabled=False)
    flex = complete_episode(3, config)
    gpu = GpuMarketEpisode(make_gpu_scenario(3, config))
    gpu.set_platform_action(flex.platform_action)
    for left, right in zip(flex.jobs, gpu.jobs):
        for i in range(10):
            assert flex.option_bill(left, left.options[i]) == pytest.approx(gpu.option_bill(right, right.options[i]))
    expected, actual = gpu.play(), flex.play()
    assert expected.selected_options == actual.selected_options
    for name in ("system_cost_yuan", "platform_profit_yuan", "user_gain_yuan", "electricity_bill_yuan"):
        assert actual.metrics[name] == pytest.approx(expected.metrics[name])


def test_disabled_sharing_keeps_negative_buyer_residual_visible():
    _, metrics = controlled_accounts(resource_saving=1000, private_saving=2000, sharing=False)
    assert metrics["flexibility_settlement_yuan"] == 0
    assert metrics["buyer_post_settlement_benefit_yuan"] == pytest.approx(-1000)
    assert metrics["cash_change_after_transfer_yuan"] == pytest.approx(1800)
    assert abs(metrics["surplus_identity_error_yuan"]) < 1e-7


def test_carbon_condition_is_a_prepublished_rule_not_an_extra_cash_carbon_charge():
    _, gated = controlled_accounts(resource_saving=1000, private_saving=100, emissions_change=1)
    _, ungated = controlled_accounts(resource_saving=1000, private_saving=100, emissions_change=1, gate=False)
    assert not gated["carbon_eligible"]
    assert gated["cash_change_after_transfer_yuan"] == pytest.approx(-200)
    assert ungated["carbon_eligible"]
    assert ungated["cash_change_after_transfer_yuan"] == pytest.approx(600)
    assert gated["system_cost_yuan"] == pytest.approx(ungated["system_cost_yuan"])
    assert gated["electricity_bill_yuan"] == pytest.approx(ungated["electricity_bill_yuan"])


def test_private_inconvenience_does_not_determine_procurement_payment():
    _, first = controlled_accounts(resource_saving=1000, private_saving=100, loss=20)
    _, second = controlled_accounts(resource_saving=1000, private_saving=100, loss=190)
    assert first["flexibility_settlement_yuan"] == pytest.approx(second["flexibility_settlement_yuan"])
    assert first["contract_system_cost_saving_yuan"] - second["contract_system_cost_saving_yuan"] == pytest.approx(170)


def test_settlement_does_not_repair_an_unprofitable_platform_contract():
    episode, metrics = controlled_accounts(resource_saving=-1000, private_saving=4000, rebate=200)
    assert metrics["platform_profit_yuan"] - episode.baseline_metrics["platform_profit_yuan"] == pytest.approx(-1200)
    assert metrics["cash_change_after_transfer_yuan"] == pytest.approx(-1200)
    assert metrics["buyer_post_settlement_benefit_yuan"] == pytest.approx(0)


@pytest.mark.parametrize("seed", range(16))
def test_real_executions_preserve_work_user_ir_resources_and_accounting(seed):
    episode = complete_episode(seed)
    original_work = episode.scenario.total_work_cu
    outcome = episode.play()
    metrics = outcome.metrics
    assert metrics["work_gpu_hours"] == pytest.approx(original_work)
    assert metrics["completion_rate"] == 1
    assert metrics["deadline_compliance"] == 1
    assert metrics["user_ir_rate"] == 1
    assert metrics["buyer_post_settlement_benefit_yuan"] >= -1e-7
    assert metrics["platform_profit_yuan"] == pytest.approx(
        metrics["platform_profit_before_transfer_yuan"] + metrics["flexibility_settlement_yuan"])
    assert metrics["platform_profit_yuan"] == pytest.approx(
        metrics["platform_profit_before_fixed_cost_yuan"] - metrics["fixed_compute_cost_yuan"])
    assert metrics["platform_objective_yuan"] == pytest.approx(metrics["platform_profit_yuan"])
    assert metrics["contract_system_cost_saving_yuan"] == pytest.approx(-metrics["delta_system_cost_yuan"])
    assert (metrics["delta_platform_profit_yuan"] + metrics["user_gain_yuan"]
            + metrics["buyer_post_settlement_benefit_yuan"]) == pytest.approx(
        -metrics["delta_system_cost_yuan"], abs=1e-6)
    assert metrics["full_system_cost_yuan"] == pytest.approx(
        metrics["system_cost_yuan"] + metrics["fixed_compute_cost_yuan"])
    # Outcome recomputation cannot transfer money again or mutate accepted contracts.
    repeated = episode.outcome()
    assert repeated.selected_options == outcome.selected_options
    assert repeated.bills_yuan == outcome.bills_yuan
    assert repeated.metrics == metrics


def test_sharing_changes_financial_distribution_without_creating_resource_savings():
    config = FlexMarketConfig()
    shared = complete_episode(7, config)
    absent = complete_episode(7, replace(config, sharing_enabled=False))
    actual, original = shared.play(), absent.play()
    assert actual.selected_options == original.selected_options
    assert actual.bills_yuan == original.bills_yuan
    for key in ("system_cost_yuan", "full_system_cost_yuan", "resource_cost_yuan", "carbon_emissions_t",
                "user_loss_yuan", "user_gain_yuan", "electricity_bill_yuan"):
        assert actual.metrics[key] == pytest.approx(original.metrics[key])
    assert actual.metrics["platform_profit_yuan"] - original.metrics["platform_profit_yuan"] == pytest.approx(
        actual.metrics["flexibility_settlement_yuan"])


def test_original_contract_and_factory_state_are_independent_and_unsubsidized():
    factory = flex_episode_factory()
    first, second = factory(0), factory(0)
    assert first.scenario is second.scenario and first is not second
    first.step_user(0)
    assert second.position == 0
    metrics = second.play().metrics
    for name in ("flexibility_resource_saving_yuan", "user_rebate_yuan", "flexibility_settlement_yuan",
                 "buyer_post_settlement_benefit_yuan", "delta_platform_profit_yuan"):
        assert metrics[name] == pytest.approx(0)


def test_platform_reward_is_actual_post_settlement_cash_profit():
    config = FlexMarketConfig(internal_carbon_value_yuan_per_t=300)
    catalogue = FixedGpuTariffCatalogue(config)
    bundle = PolicyBundle(platform_codec=catalogue,
                          platform_observer=GpuTariffObserver(catalogue, carbon_aware=True))
    env = FlexPlatformEnv(flex_episode_factory(config), bundle, "platform", [0], reward_scale_yuan=1000)
    env.reset(seed=0)
    _, reward, terminal, truncated, info = env.step(0)
    assert terminal and not truncated
    assert reward == pytest.approx(info["market_metrics"]["delta_platform_profit_yuan"] / 1000)


@pytest.mark.parametrize("settings", [
    {"resource_share": -1}, {"resource_share": 1.1}, {"resource_share": np.nan},
    {"resource_share": np.inf}, {"carbon_tolerance_t": -1}, {"carbon_tolerance_t": np.nan},
    {"conditional_rebates": 1}, {"sharing_enabled": "yes"}, {"carbon_eligibility": 1},
])
def test_invalid_contract_parameters_are_rejected(settings):
    with pytest.raises(ValueError):
        FlexMarketConfig(**settings)
