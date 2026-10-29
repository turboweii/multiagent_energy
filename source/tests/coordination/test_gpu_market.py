"""Physical and accounting checks; exact task response is only a test fixture."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from energy_net.coordination.carbon import consumption_emissions_t, nodal_carbon_intensity
from energy_net.coordination.gpu_market import GpuMarketEpisode, GpuPlatformEnv, gpu_episode_factory
from energy_net.coordination.gpu_scenario import GpuMarketConfig, make_gpu_scenario
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue, GpuTariffObserver
from energy_net.coordination.learning import PolicyBundle
from energy_net.coordination.market import encode_platform_action
from energy_net.coordination.policies import ContractExtractor


def discounted_episode(seed=0, config=None):
    episode = GpuMarketEpisode(make_gpu_scenario(seed, config))
    discounts = np.array([[0, .005, .01, .02], [.01, 0, .02, .005],
                          [.02, .01, 0, .005]])
    episode.set_platform_action(encode_platform_action(discounts, config=episode.config))
    return episode


def test_carbon_cash_is_charged_once_through_cleared_electricity_bill():
    episode = discounted_episode()
    outcome = episode.play()
    metrics, clearing = outcome.metrics, outcome.clearing
    cfg = episode.config
    dc_power = episode.scenario.load_for(outcome.selected_options) - episode.scenario.background_mw
    actual_bill = float(np.sum(dc_power * clearing.lmp_yuan_per_mwh) * cfg.dt_hours)
    assert metrics["electricity_bill_yuan"] == pytest.approx(actual_bill)
    assert metrics["generator_carbon_payment_yuan"] == pytest.approx(
        cfg.carbon_price_yuan_per_t * metrics["carbon_emissions_t"])
    assert metrics["generator_carbon_payment_yuan"] > 0
    assert metrics["platform_profit_before_fixed_cost_yuan"] == pytest.approx(
        metrics["service_revenue_yuan"] - actual_bill - metrics["compute_migration_cost_yuan"])
    assert metrics["platform_profit_yuan"] == pytest.approx(
        metrics["platform_profit_before_fixed_cost_yuan"] - cfg.daily_fixed_compute_cost_yuan)
    assert metrics["grid_dispatch_objective_yuan"] == pytest.approx(
        metrics["generation_cost_yuan"] + metrics["generator_carbon_payment_yuan"])


def test_internal_carbon_value_changes_firm_objective_but_not_cash_or_dispatch():
    ordinary = discounted_episode(7, GpuMarketConfig(internal_carbon_value_yuan_per_t=0)).play()
    green = discounted_episode(7, GpuMarketConfig(internal_carbon_value_yuan_per_t=400)).play()
    assert ordinary.selected_options == green.selected_options
    for name in ("platform_profit_yuan", "service_revenue_yuan", "electricity_bill_yuan",
                 "compute_migration_cost_yuan", "generation_cost_yuan", "carbon_emissions_t"):
        assert ordinary.metrics[name] == pytest.approx(green.metrics[name])
    assert ordinary.metrics["platform_objective_yuan"] == pytest.approx(ordinary.metrics["platform_profit_yuan"])
    assert green.metrics["platform_objective_yuan"] == pytest.approx(
        green.metrics["platform_profit_yuan"] - 400 * green.metrics["dc_consumption_emissions_t"])


def test_fixed_capacity_cost_is_separate_and_cancels_in_paired_profit_change():
    free = discounted_episode(12, GpuMarketConfig(fixed_compute_cost_yuan_per_gpu_capacity_hour=0)).play()
    fixed = discounted_episode(12, GpuMarketConfig(fixed_compute_cost_yuan_per_gpu_capacity_hour=5)).play()
    assert free.selected_options == fixed.selected_options
    m, zero = fixed.metrics, free.metrics
    assert m["fixed_compute_cost_yuan"] > 0
    assert m["full_system_cost_yuan"] == pytest.approx(m["dispatch_system_cost_yuan"] + m["fixed_compute_cost_yuan"])
    assert m["system_cost_yuan"] == pytest.approx(m["dispatch_system_cost_yuan"])
    assert m["platform_profit_before_fixed_cost_yuan"] == pytest.approx(zero["platform_profit_yuan"])
    assert m["platform_profit_yuan"] == pytest.approx(zero["platform_profit_yuan"] - m["fixed_compute_cost_yuan"])
    for name in ("delta_platform_profit_yuan", "delta_platform_objective_yuan",
                 "delta_dispatch_system_cost_yuan", "delta_full_system_cost_yuan"):
        assert m[name] == pytest.approx(zero[name], abs=1e-7)
    assert m["incremental_system_cost_yuan"] == pytest.approx(
        m["dispatch_system_cost_yuan"] - m["background_only_generation_cost_yuan"])


def test_gpu_monetary_observations_preserve_private_ir_and_choice_ranking():
    episode = discounted_episode(23)
    extractor = ContractExtractor(participation_constraint=True)
    while not episode.done:
        job = episode.jobs[episode.position]
        observation = episode.current_user_observation()
        options = observation[6:76].reshape(10, 7)
        original_bill = job.work_cu * episode.config.base_price_yuan_per_cu
        expected_bill_difference = np.array([
            (episode.option_bill(job, option) - original_bill) / job.work_cu
            for option in job.options])
        expected_loss = np.array([option.user_loss_yuan / job.work_cu for option in job.options])
        np.testing.assert_allclose(options[:, 2], expected_bill_difference, atol=1e-7)
        np.testing.assert_allclose(options[:, 3], expected_loss, atol=1e-7)
        assert options[0, 2] == 0 and options[0, 3] == 0
        exact_costs = expected_bill_difference + expected_loss
        np.testing.assert_allclose(options[:, 2] + options[:, 3], exact_costs, atol=1e-7)
        with torch.no_grad():
            logits = extractor.forward_actor(torch.from_numpy(observation[None])).numpy()[0]
        allowed = (options[:, 0] > .5) & (options[:, 1] > .5) & (options[:, 2] + options[:, 3] <= 0)
        assert np.array_equal(logits > np.finfo(np.float32).min, allowed)
        _, _, info = episode.step_user(episode.best_response_action())
        assert info["individual_rational"]


@pytest.mark.parametrize("seed", range(32))
def test_gpu_days_complete_same_work_deadlines_and_zero_quote_has_no_loss(seed):
    episode = GpuMarketEpisode(make_gpu_scenario(seed))
    assert episode.scenario.carbon_budget_t is None
    assert np.all(np.isfinite(episode.platform_observation()))
    zero = episode.play()
    assert zero.metrics["completion_rate"] == pytest.approx(1)
    assert zero.metrics["deadline_compliance"] == 1
    assert zero.metrics["user_ir_rate"] == 1
    assert zero.metrics["migrated_jobs"] == 0
    assert zero.metrics["delayed_jobs"] == 0
    assert zero.metrics["user_loss_yuan"] == pytest.approx(0)
    assert zero.metrics["delta_platform_profit_yuan"] == pytest.approx(0)
    assert zero.metrics["delta_system_cost_yuan"] == pytest.approx(0)
    assert zero.metrics["delta_platform_objective_yuan"] == pytest.approx(0)
    assert zero.metrics["carbon_budget_t"] is None
    assert zero.metrics["cap_shadow_price_yuan_per_t"] == 0
    proposal = discounted_episode(seed).play()
    assert proposal.metrics["work_gpu_hours"] == pytest.approx(zero.metrics["work_gpu_hours"])
    assert proposal.metrics["completion_rate"] == pytest.approx(1)
    assert proposal.metrics["deadline_compliance"] == 1
    assert proposal.metrics["user_ir_rate"] == 1
    assert proposal.metrics["max_balance_error_mw"] < 1e-7


def test_final_consumption_footprint_is_not_confused_with_actual_grid_emissions():
    episode = discounted_episode(12)
    outcome = episode.play()
    load = episode.scenario.load_for(outcome.selected_options)
    intensity = nodal_carbon_intensity(episode.scenario.grid_spec, outcome.clearing, load)
    footprint = consumption_emissions_t(load - episode.scenario.background_mw, intensity,
                                       episode.config.dt_hours)
    assert outcome.metrics["dc_consumption_emissions_t"] == pytest.approx(footprint)
    assert footprint < outcome.metrics["carbon_emissions_t"]
    assert outcome.metrics["grid_emissions_reduction_t"] == pytest.approx(
        -outcome.metrics["delta_carbon_emissions_t"])
    assert outcome.metrics["dc_footprint_reduction_t"] == pytest.approx(
        -outcome.metrics["delta_dc_consumption_emissions_t"])


def test_gpu_factory_returns_independent_episode_state_and_none_cap_cache():
    factory = gpu_episode_factory(maxsize=2)
    first, second = factory(7), factory(7)
    assert first is not second and first.scenario is second.scenario
    first.step_user(0)
    assert first.position == 1 and second.position == 0
    assert second.scenario.carbon_budget_t is None
    assert second.play().metrics["completion_rate"] == 1


def test_platform_environment_rewards_internal_objective_in_fixed_yuan_units():
    cfg = GpuMarketConfig()
    catalogue = FixedGpuTariffCatalogue(cfg)
    bundle = PolicyBundle(platform_codec=catalogue,
                          platform_observer=GpuTariffObserver(catalogue, carbon_aware=True))
    env = GpuPlatformEnv(gpu_episode_factory(cfg), bundle, "platform", [0], reward_scale_yuan=1000)
    observation, _ = env.reset(seed=0)
    assert observation.shape[0] == 77
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert reward == pytest.approx(info["market_metrics"]["delta_platform_objective_yuan"] / 1000)
    env.close()
