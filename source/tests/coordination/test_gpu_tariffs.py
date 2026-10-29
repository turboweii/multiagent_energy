"""Identical tariff actions, public-state isolation and carbon-only ablation."""
from dataclasses import replace

import numpy as np
import pytest

from energy_net.coordination.gpu_tariffs import (
    FixedGpuTariffCatalogue, GpuTariffObserver, candidate_gpu_tariff_observation, feature_names,
)
from energy_net.coordination.market import MarketEpisode
from energy_net.coordination.scenario import MarketConfig, make_scenario


@pytest.fixture(scope="module")
def episode():
    config = replace(MarketConfig(), discount_levels=(0., .0025, .005, .01, .02, .03))
    result = MarketEpisode(make_scenario(0, config))
    result.internal_carbon_value_yuan_per_t = 100.
    return result


def test_fixed_masks_have_exact_magnitudes_full_quotas_and_77_unique_actions(episode):
    catalogue = FixedGpuTariffCatalogue(episode.config)
    actions = [catalogue.decode(episode, index) for index in range(catalogue.n_actions)]
    assert len(catalogue.labels) == len(catalogue.entries) == 77
    assert len({tuple(action) for action in actions}) == 77
    for index, action in enumerate(actions):
        np.testing.assert_array_equal(np.asarray(episode.config.quota_levels)[action[12:]], 1.)
        discounts = np.asarray(episode.config.discount_levels)[action[:12]]
        if index:
            pattern, amplitude, _ = catalogue.entries[index]
            assert np.all(discounts[(discounts > 0)] == amplitude)
            count = 3 if pattern.startswith("time") else 4 if pattern.startswith("region") else 1
            assert np.count_nonzero(discounts) == count
        else:
            assert np.all(discounts == 0)


def test_fixed_menus_do_not_follow_electricity_or_carbon_signals(episode):
    changed = MarketEpisode(make_scenario(1, episode.config))
    catalogue = FixedGpuTariffCatalogue(episode.config)
    for index in range(catalogue.n_actions):
        np.testing.assert_array_equal(catalogue.decode(episode, index), catalogue.decode(changed, index))
    with pytest.raises(ValueError, match="exact"):
        FixedGpuTariffCatalogue(MarketConfig())


def test_carbon_ablation_keeps_all_other_features_actions_and_masks_identical(episode):
    catalogue = FixedGpuTariffCatalogue(episode.config)
    carbon = GpuTariffObserver(catalogue, carbon_aware=True)
    electric = GpuTariffObserver(catalogue, carbon_aware=False)
    actual, masked = carbon(episode), electric(episode)
    assert actual.shape == masked.shape == (77, len(feature_names()))
    carbon_indices = list(carbon.carbon_feature_indices)
    other_indices = [index for index in range(actual.shape[1]) if index not in carbon_indices]
    np.testing.assert_array_equal(actual[:, other_indices], masked[:, other_indices])
    np.testing.assert_array_equal(masked[:, carbon_indices], 0.)
    assert np.any(actual[:, carbon_indices] != 0)
    assert np.all(actual[:, -1] == 1)


def test_public_features_ignore_private_preferences_and_user_losses(episode):
    jobs = tuple(replace(job, beta_yuan_per_cu_hour=999., migration_loss_yuan_per_cu=111.,
                          options=tuple(replace(option, user_loss_yuan=99999.) for option in job.options))
                 for job in episode.scenario.jobs)
    changed = MarketEpisode(replace(episode.scenario, jobs=jobs))
    changed.internal_carbon_value_yuan_per_t = episode.internal_carbon_value_yuan_per_t
    np.testing.assert_array_equal(candidate_gpu_tariff_observation(episode),
                                  candidate_gpu_tariff_observation(changed))


def test_curriculum_only_changes_masks_and_cached_arrays_cannot_be_mutated(episode):
    observer = GpuTariffObserver(FixedGpuTariffCatalogue(episode.config))
    reference = observer(episode)
    reference[0, 0] = 999999.
    assert observer(episode)[0, 0] != 999999.
    observer.training_max_discount = .005
    training, evaluation = observer(episode, training=True), observer(episode)
    np.testing.assert_array_equal(training[:, :-1], evaluation[:, :-1])
    assert np.sum(training[:, -1]) == 1 + 19 * 2
    assert np.sum(evaluation[:, -1]) == 77


def test_duplicate_mask_removes_only_exact_raw_actions(episode):
    class DuplicateCatalogue(FixedGpuTariffCatalogue):
        def decode(self, current, index):
            return super().decode(current, 0 if index == 1 else index)

    x = candidate_gpu_tariff_observation(episode, DuplicateCatalogue(episode.config))
    assert x[0, -1] == 1 and x[1, -1] == 0
    np.testing.assert_array_equal(x[0, :-1], x[1, :-1])


def test_observation_does_not_publish_or_simulate_any_candidate(episode):
    before_action = episode.platform_action.copy()
    before_selected, before_position = list(episode.selected), episode.position
    candidate_gpu_tariff_observation(episode)
    np.testing.assert_array_equal(episode.platform_action, before_action)
    assert episode.selected == before_selected and episode.position == before_position


def test_normalization_uses_only_requested_training_seeds(episode):
    observer = GpuTariffObserver(FixedGpuTariffCatalogue(episode.config), carbon_aware=False)
    called = []

    def factory(seed):
        called.append(seed)
        return MarketEpisode(make_scenario(seed, episode.config))

    mean, scale = observer.fit_normalization(factory, [0, 1])
    assert called == [0, 1]
    assert mean.shape == scale.shape == (len(feature_names()) - 1,)
    assert np.isfinite(mean).all() and np.all(scale >= .01)


def test_carbon_mask_stays_zero_after_train_only_normalization(episode):
    catalogue = FixedGpuTariffCatalogue(episode.config)
    electric = GpuTariffObserver(catalogue, carbon_aware=False)
    carbon = GpuTariffObserver(catalogue, carbon_aware=True)

    def factory(seed):
        current = MarketEpisode(make_scenario(seed, episode.config))
        current.internal_carbon_value_yuan_per_t = 100.
        return current

    electric_mean, electric_scale = electric.fit_normalization(factory, [0, 1])
    carbon_mean, carbon_scale = carbon.fit_normalization(factory, [0, 1])
    carbon_indices = list(electric.carbon_feature_indices)
    common_indices = [index for index in range(len(feature_names()) - 1)
                      if index not in carbon_indices]
    np.testing.assert_array_equal(electric_mean[carbon_indices], 0.)
    np.testing.assert_array_equal(electric_mean[common_indices], carbon_mean[common_indices])
    np.testing.assert_array_equal(electric_scale[common_indices], carbon_scale[common_indices])
    heldout = factory(7)
    normalized = (electric(heldout)[:, :-1] - electric_mean) / electric_scale
    np.testing.assert_array_equal(normalized[:, carbon_indices], 0.)
