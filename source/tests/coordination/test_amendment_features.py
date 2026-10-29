"""Public amendment state uses actual accepted contracts and one-job OPFs."""
from copy import copy
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from energy_net.coordination.amendment_features import (
    AmendmentTariffCatalogue, AmendmentTariffObserver,
    candidate_amendment_observation, feature_names,
    public_amendment_forecasts, quote_bill_for_option, same_execution,
)
from energy_net.coordination.amendment_market import AmendmentConfig, amendment_episode_factory
from energy_net.coordination.capped_market import capped_episode_factory
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue
from energy_net.coordination.outside_bookings import OutsideBooking


@pytest.fixture(scope="module")
def cases():
    # Controlled accepted bookings: a published time-1 offer moves the three
    # jobs before the amendment experiment. This is a test fixture, not an
    # evaluation-time search over competing legacy policies.
    config = AmendmentConfig(jobs_per_user=1, batch_release_steps=(13,),
                             conditional_rebates=True)
    legacy = capped_episode_factory(config)

    @lru_cache(maxsize=16)
    def booked(seed):
        episode = legacy(seed)
        episode.set_platform_action(FixedGpuTariffCatalogue().decode(episode, 8))
        while not episode.done:
            episode.step_user(episode.best_response_action())
        return OutsideBooking.from_episode(episode, "fixed_fixture_quote_8")

    @lru_cache(maxsize=4)
    def factory(order="net"):
        cfg = replace(config, settlement_order=order)
        return amendment_episode_factory(lambda seed: legacy(seed).scenario,
                                         lambda seed: booked(seed).copy(), cfg)
    return factory, booked


def test_catalogue_preserves_the_free_amendment_as_a_separate_action(cases):
    episode = cases[0]()(4)
    catalogue = AmendmentTariffCatalogue(episode.config)
    raw = [catalogue.decode(episode, index) for index in range(78)]
    assert len(catalogue.entries) == len(catalogue.labels) == 78
    assert all(action.shape == (16,) for action in raw)
    assert len({tuple(action) for action in raw}) == 78
    np.testing.assert_array_equal(raw[0][:-1], raw[1][:-1])
    assert raw[0][-1] == 0 and raw[1][-1] == 1
    base = FixedGpuTariffCatalogue(episode.config)
    for index in range(2, 78):
        np.testing.assert_array_equal(raw[index][:-1], base.decode(episode, index - 1))
        assert raw[index][-1] == 1


@pytest.mark.parametrize("bad", [-1, 78, .5, np.nan, np.inf, [1, 2]])
def test_catalogue_rejects_invalid_indices(cases, bad):
    with pytest.raises(ValueError):
        AmendmentTariffCatalogue().decode(cases[0]()(4), bad)


def test_sampler_returns_valid_quotes_for_user_pretraining(cases):
    episode, rng = cases[0]()(4), np.random.default_rng(24)
    catalogue = AmendmentTariffCatalogue(episode.config)
    samples = [catalogue.sample(episode, rng) for _ in range(100)]
    assert all(action.shape == (16,) for action in samples)
    assert all(np.all((action[:15] >= 0) & (action[:15] < episode.config.platform_nvec))
               and action[-1] in (0, 1) for action in samples)
    assert any(action[-1] == 0 for action in samples)
    assert any(action[-1] == 1 and not np.all(action[:12] == action[0]) for action in samples)


def test_additional_rebates_use_the_accepted_invoice_and_physical_change(cases):
    episode = cases[0]()(4)
    assert any(index > 1 for index in episode.outside_selected_options)
    discounts = np.full((episode.config.n_steps, 3), .02)
    for index, job in enumerate(episode.jobs):
        outside_bill = episode.baseline_bills[index]
        assert outside_bill < episode.config.base_price_yuan_per_cu * job.work_cu
        assert quote_bill_for_option(job, job.options[0], discounts, outside_bill) == outside_bill
        alias = replace(job.options[0], index=9, is_baseline=False,
                        user_loss_yuan=12345., migrated=not job.options[0].migrated)
        assert same_execution(alias, job.options[0])
        assert quote_bill_for_option(job, alias, discounts, outside_bill) == outside_bill
        original_index = episode.option_source_indices[index].index(0)
        original = job.options[original_index]
        assert not original.migrated and not same_execution(original, job.options[0])
        assert quote_bill_for_option(job, original, discounts, outside_bill) == pytest.approx(.98 * outside_bill)
        assert quote_bill_for_option(job, original, discounts, outside_bill,
                                     amendment_open=False) == outside_bill
    # A real episode and the public helper implement the same invoice.
    episode.set_platform_action(AmendmentTariffCatalogue().decode(episode, 9))
    blocks = np.minimum(np.arange(episode.config.n_steps) * 4 // episode.config.n_steps, 3)
    quoted_discounts = episode.discounts[:, blocks].T
    for index, job in enumerate(episode.jobs):
        for option in job.options:
            assert episode.option_bill(job, option) == pytest.approx(
                quote_bill_for_option(job, option, quoted_discounts, episode.baseline_bills[index]))


class CountingGrid:
    def __init__(self, grid):
        self.grid, self.caps = grid, []

    def solve(self, load, cap):
        self.caps.append(cap)
        return self.grid.solve(load, cap)


def test_forecasts_match_the_complete_same_cap_one_job_counterfactual(cases):
    episode = cases[0]()(4)
    episode.grid = CountingGrid(episode.grid)
    public, allocation = public_amendment_forecasts(episode)
    np.testing.assert_allclose(allocation, episode._baseline_alloc)
    assert episode.grid.caps and set(episode.grid.caps) == {episode.scenario.carbon_budget_t}
    assert all(len(alternatives) == sum(available) - 1
               for (_, alternatives), available in zip(public, episode.option_available))
    for job_index, (job, alternatives) in enumerate(public):
        for forecast in alternatives:
            if not forecast.feasible:
                continue
            selected = [0] * len(episode.jobs)
            selected[job_index] = forecast.option.index
            load = episode.scenario.load_for(selected)
            clearing = episode.grid.solve(load, episode.scenario.carbon_budget_t)
            operating_change = forecast.option.operating_cost_yuan - job.options[0].operating_cost_yuan
            bill = np.sum((load - episode.scenario.background_mw) * clearing.lmp_yuan_per_mwh) * episode.config.dt_hours
            assert forecast.resource_saving_yuan == pytest.approx(
                episode.baseline_clearing.total_generation_cost - clearing.total_generation_cost - operating_change)
            assert forecast.private_pre_rebate_saving_yuan == pytest.approx(
                episode._outside_electricity_bill - bill - operating_change)
            assert forecast.grid_emission_reduction_t == pytest.approx(
                episode.baseline_clearing.emissions_t - clearing.emissions_t)


def test_public_state_ignores_private_preferences_losses_and_user_policies(cases):
    episode = cases[0]()(4)
    expected = candidate_amendment_observation(episode)
    other = copy(episode)
    other.jobs = tuple(replace(job, beta_yuan_per_cu_hour=999., migration_loss_yuan_per_cu=333.,
        options=tuple(replace(option, user_loss_yuan=option.user_loss_yuan + 777777.)
                      for option in job.options)) for job in episode.jobs)
    other.scenario = replace(episode.scenario, jobs=other.jobs)
    other.outside_losses = np.full(len(other.jobs), -1e12)
    other.booking = replace(episode.booking, metrics=dict(episode.booking.metrics,
                            user_loss_yuan=1e30, system_cost_yuan=-1e30, user_gain_yuan=1e20))
    def private_response(*args, **kwargs):
        raise AssertionError("Public features must not run user responses")
    other.best_response_action = private_response
    other.outcome = private_response
    np.testing.assert_array_equal(candidate_amendment_observation(other), expected)


def test_state_keeps_frozen_outside_cash_and_exposes_both_zero_quote_actions(cases):
    episode = cases[0]()(4)
    selected, bills, position = list(episode.selected), episode.bills.copy(), episode.position
    action = episode.platform_action.copy()
    observation = AmendmentTariffObserver().raw(episode)
    names = feature_names()
    assert observation.shape == (78, len(names)) == (78, 165)
    assert len(set(names)) == len(names) and names[-1] == "unique_raw_action"
    assert np.isfinite(observation).all()
    assert observation[0, -1] == observation[1, -1] == 1
    assert observation[0, names.index("amendment_open")] == 0
    assert observation[1, names.index("amendment_open")] == 1
    assert observation[0, names.index("owner_best_resource_saving_yuan_per_gpuh_0")] == 0
    assert observation[1, names.index("owner_best_resource_saving_yuan_per_gpuh_0")] > 0
    assert observation[0, names.index("accepted_outside_profit_yuan_per_gpuh")] == pytest.approx(
        episode._outside_profit / episode.scenario.total_work_cu, abs=1e-7)
    np.testing.assert_array_equal(episode.bills, bills)
    np.testing.assert_array_equal(episode.platform_action, action)
    assert episode.selected == selected and episode.position == position
    assert not episode.reservations and not episode.amendment_ledger


@pytest.mark.parametrize("order", ["net", "gross", "direct"])
def test_forecast_margin_and_qualification_match_prebinding_market_rule(cases, order):
    episode, catalogue = cases[0](order)(4), AmendmentTariffCatalogue()
    observer, names = AmendmentTariffObserver(catalogue), feature_names()
    observation = observer.raw(episode)
    assert observation.shape == (78, 165)
    assert observation[1, names.index("net_before_share")] == int(order == "net")
    assert observation[1, names.index("direct_no_settlement")] == int(order == "direct")
    for menu in (1, 5, 37, 77):
        episode.position = 0
        episode.set_platform_action(catalogue.decode(episode, menu))
        for job_index, job in enumerate(episode.jobs):
            episode.position = job_index
            previews = [episode.preview_amendment(index) for index, available
                        in enumerate(episode.option_available[job_index]) if available and index > 0]
            feasible = [preview for preview in previews if preview["feasible"]]
            assert feasible
            margin = np.mean([preview["platform_gain_yuan"] for preview in feasible]) / job.work_cu
            admitted = np.mean([preview["admissible"] for preview in feasible])
            owner = job.user_id
            assert observation[menu, names.index(f"owner_mean_proposed_platform_gain_yuan_per_gpuh_{owner}")] == pytest.approx(margin, abs=1e-7)
            assert observation[menu, names.index(f"owner_admissible_fraction_{owner}")] == pytest.approx(admitted)


def test_cache_reuses_frozen_case_across_new_episode_views_and_not_changed_invoices(cases):
    factory = cases[0]()
    first, second = factory(4), factory(4)
    assert first.scenario is not second.scenario
    assert first.original_scenario is second.original_scenario
    observer = AmendmentTariffObserver()
    first.grid = CountingGrid(first.grid)
    expected = observer.raw(first)
    assert first.grid.caps
    second.grid = CountingGrid(second.grid)
    actual = observer.raw(second)
    assert not second.grid.caps
    np.testing.assert_array_equal(actual, expected)
    actual[0, 0] = -1e10
    np.testing.assert_array_equal(observer.raw(second), expected)
    second.baseline_bills += 11.
    changed = observer.raw(second)
    assert second.grid.caps
    column = feature_names().index("owner_outside_invoice_yuan_per_gpuh_0")
    assert changed[0, column] > expected[0, column]


def test_curriculum_retains_exit_and_free_amendment_and_normalizes_only_given_seeds(cases):
    factory, observer = cases[0](), AmendmentTariffObserver()
    episode = factory(4)
    observer.training_max_discount = 0
    raw, training = observer.raw(episode), observer(episode, training=True)
    np.testing.assert_array_equal(raw[:, :-1], training[:, :-1])
    np.testing.assert_array_equal(training[:, -1], [1, 1] + [0] * 76)
    np.testing.assert_array_equal(observer(episode, training=False), raw)
    called = []
    def selected_days(seed):
        called.append(seed)
        return factory(seed)
    mean, scale = observer.fit_normalization(selected_days, [2, 3])
    assert called == [2, 3]
    rows = np.concatenate([observer.raw(factory(seed))[:, :-1] for seed in (2, 3)])
    np.testing.assert_allclose(mean, rows.mean(axis=0, dtype=np.float64), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(scale, np.maximum(rows.std(axis=0, dtype=np.float64), .01), rtol=1e-6)
    assert mean.shape == scale.shape == (164,)
    assert np.all(scale >= .01 - 1e-9)
