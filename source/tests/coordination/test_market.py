"""Task, contract, private utility and settlement relationships of the market."""
from dataclasses import replace

import numpy as np
import pytest

from energy_net.coordination.grid import GridClearing, default_grid
from energy_net.coordination.market import MarketEpisode, encode_platform_action
from energy_net.coordination.scenario import (
    ExecutionOption, Job, MarketConfig, Scenario, _original_shapes, make_scenario,
)


def simple_scenario(*, restricted_destination=False, beta=0.025):
    """One fixed task with separate same-place, delayed and migrated contracts."""
    cfg = MarketConfig(idle_power_mw=(0.08, 0.08, 0.08))
    work, release, duration, deadline = 1800.0, 8, 2, 16
    options = []
    for index, (region, start, original) in enumerate(
        [(0, 8, True), (0, 8, False), (0, 10, False), (2, 10, False)]
    ):
        allocation = np.zeros((48, 3))
        allocation[start:start + duration, region] = work / duration
        extra_energy = np.zeros((48, 3))
        transfer = np.zeros((48, 3, 3))
        moved = region != 0
        if moved:
            gb = work * cfg.transfer_gb_per_cu
            transfer[start - 1, 0, region] = gb
            extra_energy[start - 1, region] = gb * cfg.transfer_mwh_per_gb
        delay = (start - release) * cfg.dt_hours
        loss = work * (beta * delay + 0.008 * moved)
        opex = work * cfg.compute_opex_yuan_per_cu
        opex += transfer.sum() * cfg.transfer_cost_yuan_per_gb
        options.append(ExecutionOption(
            index, region, start, start + duration, allocation, extra_energy,
            transfer, delay, moved, loss, float(opex), original,
        ))
    job = Job("fixed-task", 0, work, 0, release, deadline, beta, 0.008, tuple(options))
    spec = default_grid(np.zeros(48))
    capacities = np.zeros((48, 4))
    capacities[:, 0] = 55.0
    spec = replace(spec, generator_capacity_mw=capacities)
    if restricted_destination:
        spec = replace(spec, line_capacity_mw=np.array([10.0, 0.1, 0.1]))
    background = np.zeros((48, 3))
    background[:, 0] = 5.0
    return Scenario(cfg, 123, background, np.zeros(48), spec, 10000.0, (job,))


def uniform_action(config, discount=0.15, quota=1.0):
    return encode_platform_action(np.full((3, 4), discount),
                                  quotas=np.full(3, quota), config=config)


def play_option(scenario, index, discount=0.15):
    episode = MarketEpisode(scenario)
    episode.reset(uniform_action(scenario.config, discount))
    reward, done, info = episode.step_user(index)
    assert done
    return episode, reward, info, episode.outcome()


def test_generated_options_preserve_task_work_and_hard_time_window():
    scenario = make_scenario(7)
    for job in scenario.jobs:
        for option in job.options:
            assert option.allocation_cu.sum() == pytest.approx(job.work_cu)
            assert job.release_step <= option.start_step < option.end_step <= job.deadline_step
            busy_steps = np.flatnonzero(option.allocation_cu.sum(axis=1))
            assert np.all(busy_steps >= job.release_step)
            assert np.all(busy_steps < job.deadline_step)
    episode = MarketEpisode(scenario)
    episode.reset(uniform_action(scenario.config))
    outcome = episode.play()
    assert outcome.metrics["work_cu"] == pytest.approx(scenario.total_work_cu)
    assert outcome.metrics["completion_rate"] == pytest.approx(1.0)
    assert outcome.metrics["deadline_compliance"] == 1.0


def test_original_contract_is_retained_and_option_one_only_changes_invoice():
    scenario = simple_scenario()
    job = scenario.jobs[0]
    original, new_quote = job.options[:2]
    np.testing.assert_array_equal(original.allocation_cu, new_quote.allocation_cu)
    np.testing.assert_array_equal(original.extra_energy_mwh, new_quote.extra_energy_mwh)
    episode = MarketEpisode(scenario)
    episode.reset(uniform_action(scenario.config, discount=0.30, quota=0.5))
    base_bill = job.work_cu * scenario.config.base_price_yuan_per_cu
    assert episode.option_bill(job, original) == pytest.approx(base_bill)
    assert episode.option_bill(job, new_quote) == pytest.approx(base_bill * 0.70)
    assert episode._candidate(0)[0]
    assert episode._candidate(1)[0]


def test_uniform_discount_reprices_without_needless_delay_or_migration():
    scenario = make_scenario(0)
    original = MarketEpisode(scenario).play()
    episode = MarketEpisode(scenario)
    episode.reset(uniform_action(scenario.config, discount=0.15))
    discounted = episode.play()
    assert discounted.selected_options == (1,) * len(scenario.jobs)
    assert discounted.metrics["migrated_jobs"] == 0
    assert discounted.metrics["delayed_jobs"] == 0
    assert discounted.metrics["user_ir_rate"] == 1.0
    assert discounted.metrics["generation_cost_yuan"] == pytest.approx(original.metrics["generation_cost_yuan"])
    assert discounted.metrics["service_revenue_yuan"] == pytest.approx(
        original.metrics["service_revenue_yuan"] * 0.85)


def test_quota_preserves_base_reservation_but_can_reject_new_capacity():
    scenario = simple_scenario()
    episode = MarketEpisode(scenario)
    episode.reset(uniform_action(scenario.config, quota=0.5))
    # Original slot needs 900 CU, above the offered 650-CU quota, but remains
    # reserved. The same 900-CU task in a new slot exceeds that slot's quota.
    assert episode._candidate(0)[0]
    assert episode._candidate(1)[0]
    feasible, _, reason = episode._candidate(2)
    assert not feasible and reason == "posted_quota"
    _, done, info = episode.step_user(2)
    assert done and info["chosen"] == 0
    assert episode.outcome().metrics["completion_rate"] == pytest.approx(1.0)


def test_energy_infeasible_application_falls_back_to_original_full_task():
    scenario = simple_scenario(restricted_destination=True)
    episode = MarketEpisode(scenario)
    episode.reset(uniform_action(scenario.config))
    assert episode._precheck([3])[0]  # Compute, WAN and posted quota permit it.
    feasible, _, reason = episode._candidate(3)
    assert not feasible and reason == "energy_feasibility"
    reward, _, info = episode.step_user(3)
    outcome = episode.outcome()
    assert reward == pytest.approx(0.0)
    assert info["requested"] == 3 and info["chosen"] == 0
    assert outcome.clearing.feasible
    assert outcome.metrics["work_cu"] == pytest.approx(scenario.total_work_cu)
    assert outcome.metrics["deadline_compliance"] == 1.0
    assert outcome.metrics["reservation_rejections"] == 1


def test_private_delay_loss_affects_user_reward_not_platform_profit_reward():
    low, low_reward, _, low_outcome = play_option(simple_scenario(beta=0.01), 2)
    high, high_reward, high_info, high_outcome = play_option(simple_scenario(beta=0.30), 2)
    assert high_reward < low_reward
    assert not high_info["individual_rational"]
    assert high_reward == pytest.approx(10 * high_info["utility_gain_yuan"] / high.baseline_bills[0])
    assert high.platform_reward == pytest.approx(low.platform_reward)
    assert high_outcome.metrics["system_cost_yuan"] > low_outcome.metrics["system_cost_yuan"]


def test_learning_user_receives_own_utility_instead_of_global_profit():
    from energy_net.coordination.learning import ActorEnv, PolicyBundle

    scenario = simple_scenario(beta=0.30)

    class PostedDiscount:
        def predict(self, observation, deterministic=True):
            return uniform_action(scenario.config), None

    bundle = PolicyBundle(platform=PostedDiscount())
    env = ActorEnv(lambda seed: MarketEpisode(scenario), bundle, actor=0,
                   scenario_seeds=(123,))
    env.reset(seed=0)
    _, reward, done, _, info = env.step(2)
    assert done
    expected = 10 * info["utility_gain_yuan"] / env.episode.baseline_bills[0]
    assert reward == pytest.approx(expected)
    assert reward != pytest.approx(env.episode.platform_reward)


def test_price_transfers_do_not_change_same_physical_resource_cost():
    scenario = simple_scenario()
    _, _, _, first = play_option(scenario, 3, discount=0.10)
    _, _, _, second = play_option(scenario, 3, discount=0.20)
    assert second.metrics["service_revenue_yuan"] < first.metrics["service_revenue_yuan"]
    assert second.metrics["resource_cost_yuan"] == pytest.approx(first.metrics["resource_cost_yuan"])
    assert second.metrics["system_cost_yuan"] == pytest.approx(first.metrics["system_cost_yuan"])
    expected_resource = first.clearing.total_generation_cost + scenario.jobs[0].options[3].operating_cost_yuan
    assert first.metrics["resource_cost_yuan"] == pytest.approx(expected_resource)
    assert first.metrics["system_cost_yuan"] == pytest.approx(expected_resource + first.user_losses_yuan[0])
    change_in_revenue = second.metrics["service_revenue_yuan"] - first.metrics["service_revenue_yuan"]
    assert second.metrics["platform_profit_yuan"] - first.metrics["platform_profit_yuan"] == pytest.approx(change_in_revenue)


def test_system_cost_does_not_add_carbon_shadow_payments_to_generation_resources():
    scenario = make_scenario(0)
    outcome = MarketEpisode(scenario).play()
    assert outcome.clearing.carbon_price_yuan_per_t > 0
    physical_generation_cost = float(np.sum(
        outcome.clearing.generation_mw * scenario.grid_spec.generator_cost_yuan_per_mwh
    ) * scenario.config.dt_hours)
    operation_cost = sum(job.options[index].operating_cost_yuan
                         for job, index in zip(scenario.jobs, outcome.selected_options))
    user_loss = sum(outcome.user_losses_yuan)
    assert outcome.metrics["system_cost_yuan"] == pytest.approx(
        physical_generation_cost + operation_cost + user_loss)
    # The binding cap's shadow value is already reflected in settlement prices;
    # it is not an additional physical resource expenditure.
    assert outcome.metrics["system_cost_yuan"] != pytest.approx(
        physical_generation_cost + operation_cost + user_loss
        + outcome.clearing.carbon_price_yuan_per_t * outcome.clearing.emissions_t)


def test_platform_electricity_bill_contains_idle_compute_transfer_not_background():
    scenario = simple_scenario()
    _, _, _, first = play_option(scenario, 3)
    option = scenario.jobs[0].options[3]
    idle_mwh = np.sum(scenario.idle_load_mw) * scenario.config.dt_hours
    compute_mwh = np.sum(option.allocation_cu * np.asarray(scenario.config.energy_mwh_per_cu))
    transfer_mwh = np.sum(option.extra_energy_mwh)
    assert transfer_mwh > 0
    np.testing.assert_allclose(first.clearing.lmp_yuan_per_mwh, 180.0)
    expected_bill = (idle_mwh + compute_mwh + transfer_mwh) * 180.0
    assert first.metrics["electricity_bill_yuan"] == pytest.approx(expected_bill)
    background = scenario.background_mw.copy()
    background[:, 0] += 2.0
    changed = replace(scenario, background_mw=background)
    _, _, _, other = play_option(changed, 3)
    assert other.metrics["electricity_bill_yuan"] == pytest.approx(expected_bill)
    expected_system_increase = 2.0 * 48 * scenario.config.dt_hours * 180.0
    assert other.clearing.total_generation_cost - first.clearing.total_generation_cost == pytest.approx(expected_system_increase)


def test_platform_observation_does_not_expose_private_delay_preferences():
    original = simple_scenario(beta=0.01)
    private_change = simple_scenario(beta=0.30)
    first, second = MarketEpisode(original), MarketEpisode(private_change)
    np.testing.assert_array_equal(first.platform_observation(), second.platform_observation())
    assert not np.array_equal(first.current_user_observation(), second.current_user_observation())


def test_original_load_shapes_are_finite_and_training_seed_pool_is_feasible():
    shapes = _original_shapes()
    assert shapes.ndim == 2 and shapes.shape[1] == 48
    assert np.isfinite(shapes).all()
    np.testing.assert_allclose(shapes.mean(axis=1), 1.0, atol=1e-10)
    for seed in range(64):
        scenario = make_scenario(seed)
        assert np.isfinite(scenario.background_mw).all(), seed
        result = GridClearing(scenario.grid_spec).solve(scenario.baseline_load_mw,
                                                       scenario.carbon_budget_t)
        assert result.feasible, (seed, result.status)
        assert result.emissions_t <= scenario.carbon_budget_t + 1e-6
