"""Forecasts remain public, contract-consistent and separate from user choices."""
from dataclasses import replace

import numpy as np
import pytest

from energy_net.coordination.flex_market import FlexMarketConfig, FlexMarketEpisode, flex_episode_factory
from energy_net.coordination.flex_tariffs import (
    FlexTariffCatalogue, FlexTariffObserver, candidate_flex_tariff_observation,
    feature_names, public_option_forecasts, quote_bill_for_option,
)
from energy_net.coordination.gpu_scenario import make_gpu_scenario


@pytest.fixture(scope="module")
def episode():
    return flex_episode_factory()(0)


def test_all_contract_and_sharing_arms_have_identical_raw_77_menus_and_feature_shape(episode):
    expected = [FlexTariffCatalogue(episode.config).decode(episode, i) for i in range(77)]
    observed_shapes = []
    for conditional, sharing, carbon in ((False, False, False), (True, False, False),
                                         (True, True, False), (True, True, True)):
        cfg = replace(episode.config, conditional_rebates=conditional,
                      sharing_enabled=sharing, carbon_eligibility=carbon)
        current = flex_episode_factory(cfg)(0)
        catalogue = FlexTariffCatalogue(cfg)
        for index in range(77):
            np.testing.assert_array_equal(catalogue.decode(current, index), expected[index])
        values = FlexTariffObserver(catalogue)(current)
        observed_shapes.append(values.shape)
        assert np.isfinite(values).all()
        assert np.all(values[:, -1] == 1)
    assert observed_shapes == [(77, len(feature_names()))] * 4


def test_known_quote_bill_matches_real_episode_and_unchanged_contract_has_no_conditional_rebate(episode):
    job = episode.jobs[0]
    raw = np.r_[np.full(12, 5), np.full(3, 2)]
    episode.reset(raw)
    blocks = np.arange(48) // 12
    tariff = episode.tariff_yuan_per_cu[:, blocks].T
    for option in job.options:
        proposed = quote_bill_for_option(job, option, tariff, episode.config.base_price_yuan_per_cu, True)
        assert proposed == pytest.approx(episode.option_bill(job, option))
    assert quote_bill_for_option(job, job.options[1], tariff, 18., True) == job.work_cu * 18.
    assert quote_bill_for_option(job, job.options[1], tariff, 18., False) < job.work_cu * 18.
    assert quote_bill_for_option(job, job.options[2], tariff, 18., True) < job.work_cu * 18.
    episode.reset()


def test_conditional_observer_removes_original_windfall_statistics(episode):
    conditional = FlexTariffObserver()(episode)
    unconditional_ep = flex_episode_factory(replace(episode.config, conditional_rebates=False))(0)
    unconditional = FlexTariffObserver()(unconditional_ep)
    columns = [i for i, name in enumerate(feature_names()) if name.startswith("owner_original_windfall")]
    np.testing.assert_array_equal(conditional[:, columns], 0.)
    assert np.any(unconditional[:, columns] > 0)


def test_single_job_public_forecasts_match_actual_counterfactual_clearing(episode):
    public, _ = public_option_forecasts(episode)
    job, forecasts = public[0]
    feasible = next(f for f in forecasts if f.feasible and f.option.migrated)
    indices = [0] * len(episode.jobs)
    indices[0] = feasible.option.index
    result = episode.grid.solve(episode.scenario.load_for(indices), episode.scenario.carbon_budget_t)
    assert feasible.resource_saving_yuan == pytest.approx(
        episode.baseline_clearing.total_generation_cost - result.total_generation_cost
        - feasible.option.operating_cost_yuan + job.options[0].operating_cost_yuan)
    assert feasible.grid_emission_reduction_t == pytest.approx(
        episode.baseline_clearing.emissions_t - result.emissions_t)


def test_public_features_ignore_beta_kappa_losses_and_never_call_a_user_policy(episode):
    altered_jobs = tuple(replace(job, beta_yuan_per_cu_hour=999., migration_loss_yuan_per_cu=888.,
        options=tuple(replace(option, user_loss_yuan=123456789.) for option in job.options))
        for job in episode.jobs)
    changed = FlexMarketEpisode(replace(episode.scenario, jobs=altered_jobs))
    original = candidate_flex_tariff_observation(episode)
    np.testing.assert_array_equal(original, candidate_flex_tariff_observation(changed))


def test_observer_never_publishes_or_changes_bookings_and_caches_public_solves(episode):
    before_action, before_position = episode.platform_action.copy(), episode.position
    before_selected = list(episode.selected)
    observer = FlexTariffObserver()
    count = 0
    actual_solve = episode.grid.solve
    def counted(*args, **kwargs):
        nonlocal count
        count += 1
        return actual_solve(*args, **kwargs)
    episode.grid.solve = counted
    try:
        first = observer(episode)
        first_count = count
        assert first_count > 0
        second = observer(episode)
        assert count == first_count
        first[0, 0] = 999999.
        assert observer(episode)[0, 0] != 999999.
        np.testing.assert_array_equal(second, observer(episode))
    finally:
        episode.grid.solve = actual_solve
    np.testing.assert_array_equal(episode.platform_action, before_action)
    assert episode.position == before_position and episode.selected == before_selected


def test_resource_sharing_prediction_respects_resource_and_carbon_contract_rules(episode):
    plain_cfg = replace(episode.config, carbon_eligibility=False)
    green_cfg = replace(episode.config, carbon_eligibility=True)
    plain = candidate_flex_tariff_observation(flex_episode_factory(plain_cfg)(0))
    green = candidate_flex_tariff_observation(flex_episode_factory(green_cfg)(0))
    columns = [feature_names().index(f"owner_best_forecast_settled_margin_yuan_per_gpuh_{u}") for u in range(3)]
    # A green eligibility restriction cannot improve any individual forecast's
    # entitlement; owner aggregation of max margins preserves this ordering.
    assert np.all(green[:, columns] <= plain[:, columns] + 1e-6)
    resource_columns = [feature_names().index(f"owner_best_forecast_resource_saving_yuan_per_gpuh_{u}") for u in range(3)]
    np.testing.assert_array_equal(green[:, resource_columns], plain[:, resource_columns])


@pytest.mark.parametrize("sharing,green", [(False, False), (True, False), (True, True)])
def test_one_job_margin_forecast_equals_actual_contract_settlement(sharing, green):
    """With one responding job, the public counterfactual is the real market."""
    config = FlexMarketConfig(sharing_enabled=sharing, carbon_eligibility=green)
    scenario = make_gpu_scenario(3, config)
    scenario = replace(scenario, jobs=(scenario.jobs[0],))
    current = FlexMarketEpisode(scenario)
    catalogue = FlexTariffCatalogue(config)
    values = FlexTariffObserver(catalogue)(current)
    owner = scenario.jobs[0].user_id
    margin_column = feature_names().index(f"owner_best_forecast_settled_margin_yuan_per_gpuh_{owner}")
    quantity = scenario.jobs[0].work_cu
    for index in (5, 18, 29, 42, 65, 76):
        raw = catalogue.decode(current, index)
        actual_offered_margins = []
        for option in scenario.jobs[0].options[1:]:
            current.reset(raw)
            bill = current.option_bill(scenario.jobs[0], option)
            if bill >= current.baseline_bills[0] - 1e-9 or not current._candidate(option.index)[0]:
                continue
            current.step_user(option.index)
            actual_offered_margins.append(current.outcome().metrics["delta_platform_profit_yuan"] / quantity)
        expected = max(actual_offered_margins) if actual_offered_margins else 0.
        assert values[index, margin_column] == pytest.approx(expected, abs=2e-7)


def test_training_curriculum_only_masks_discounts_and_normalization_uses_training_seeds():
    observer = FlexTariffObserver()
    factory = flex_episode_factory()
    current = factory(0)
    observer.training_max_discount = .005
    training, evaluation = observer(current, training=True), observer(current)
    np.testing.assert_array_equal(training[:, :-1], evaluation[:, :-1])
    assert training[:, -1].sum() == 39 and evaluation[:, -1].sum() == 77
    called = []
    def selected(seed):
        called.append(seed)
        return factory(seed)
    mean, scale = observer.fit_normalization(selected, [0, 1])
    assert called == [0, 1]
    assert mean.shape == scale.shape == (len(feature_names()) - 1,)
    assert np.isfinite(mean).all() and np.all(scale >= .01)


def test_duplicated_raw_menu_is_the_only_reason_to_mask_a_menu(episode):
    class DuplicateCatalogue(FlexTariffCatalogue):
        def decode(self, current, index):
            return super().decode(current, 0 if index == 1 else index)
    values = candidate_flex_tariff_observation(episode, DuplicateCatalogue(episode.config))
    assert values[0, -1] == 1 and values[1, -1] == 0
    np.testing.assert_array_equal(values[0, :-1], values[1, :-1])
