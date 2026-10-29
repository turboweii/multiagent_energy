"""Static quote expansion preserves all legacy actions and public interfaces."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from energy_net.coordination.amendment_features import AmendmentTariffCatalogue, AmendmentTariffObserver
from energy_net.coordination.amendment_market import AmendmentConfig, amendment_episode_factory
from energy_net.coordination.capped_market import capped_episode_factory
from energy_net.coordination.dynamic_tariffs import ExpandedAmendmentCatalogue
from energy_net.coordination.outside_bookings import OutsideBooking


def matrix(catalogue, episode, index):
    raw = catalogue.decode(episode, index)
    return np.asarray(episode.config.discount_levels)[raw[:12]].reshape(3, 4)


def matching_indices(catalogue, episode, expected):
    return [index for index in range(catalogue.n_actions)
            if np.allclose(matrix(catalogue, episode, index), expected, rtol=0, atol=1e-12)
            and catalogue.decode(episode, index)[-1] == 1]


@pytest.mark.parametrize("configured", [False, True])
def test_exactly_204_unique_static_quotes_preserve_the_original_78_indices(configured):
    config = AmendmentConfig()
    episode = SimpleNamespace(config=config)
    catalogue = ExpandedAmendmentCatalogue(config if configured else None)
    legacy = AmendmentTariffCatalogue(config)
    assert catalogue.n_actions == len(catalogue.entries) == len(catalogue.labels) == 204
    quotes = [catalogue.decode(episode, index) for index in range(catalogue.n_actions)]
    assert all(raw.shape == (16,) for raw in quotes)
    assert len({tuple(raw) for raw in quotes}) == catalogue.n_actions
    assert catalogue.labels[:78] == legacy.labels and catalogue.entries[:78] == legacy.entries
    for index in range(78):
        np.testing.assert_array_equal(quotes[index], legacy.decode(episode, index))
    assert quotes[0][-1] == 0 and all(raw[-1] == 1 for raw in quotes[1:])
    for raw, entry in zip(quotes, catalogue.entries):
        assert np.all((raw[:15] >= 0) & (raw[:15] < config.platform_nvec))
        assert np.all(np.asarray(config.quota_levels)[raw[12:15]] == 1)
        if entry is not None:
            assert entry[1] == pytest.approx(np.asarray(config.discount_levels)[raw[:12]].max())


def test_day_and_night_can_target_different_regions_at_different_discounts():
    episode = SimpleNamespace(config=AmendmentConfig())
    catalogue = ExpandedAmendmentCatalogue()
    expected = np.zeros((3, 4))
    expected[0, [1, 2]] = .0025
    expected[2, [0, 3]] = .01
    found = matching_indices(catalogue, episode, expected)
    assert len(found) == 1 and found[0] >= 78
    assert catalogue.entries[found[0]][1] == .01
    assert matching_indices(AmendmentTariffCatalogue(), episode, expected) == []


@pytest.mark.parametrize("rates", [(0., .0025, .01),(.0025,.005,.01),(.005,.005,.005)])
def test_all_day_independent_regional_rates_and_uniform_all_region_quote(rates):
    episode = SimpleNamespace(config=AmendmentConfig())
    catalogue = ExpandedAmendmentCatalogue()
    expected = np.repeat(np.asarray(rates)[:, None], 4, axis=1)
    found = matching_indices(catalogue, episode, expected)
    assert len(found) == 1 and found[0] >= 78


class OnlyConfiguration:
    def __init__(self, config):
        self.config = config

    def __getattr__(self, name):
        raise AssertionError(f"A static quote decoder must not read scenario state: {name}")


def test_decoder_uses_no_grid_weather_preferences_or_user_outcome_and_returns_copies():
    config = AmendmentConfig()
    catalogue = ExpandedAmendmentCatalogue()
    first = OnlyConfiguration(config)
    second = SimpleNamespace(config=config, scenario="different weather", market_metrics={"profit":1e30})
    for index in range(catalogue.n_actions):
        expected = catalogue.decode(first, index)
        np.testing.assert_array_equal(expected, catalogue.decode(second, index))
        expected[:] = 0
        assert catalogue.decode(first, index)[-1] == int(index != 0)


def test_decode_respects_semantic_levels_when_the_config_order_changes():
    config = AmendmentConfig()
    reversed_config = replace(config, discount_levels=tuple(reversed(config.discount_levels)))
    catalogue = ExpandedAmendmentCatalogue()
    first, second = SimpleNamespace(config=config), SimpleNamespace(config=reversed_config)
    for index in range(catalogue.n_actions):
        np.testing.assert_array_equal(matrix(catalogue, first, index), matrix(catalogue, second, index))
        if index < 78:
            np.testing.assert_array_equal(catalogue.decode(second,index),
                                          AmendmentTariffCatalogue().decode(second,index))


@pytest.mark.parametrize("bad", [-1, 204, .5, np.nan, np.inf, [1, 2]])
def test_invalid_menu_indices_are_rejected(bad):
    with pytest.raises(ValueError):
        ExpandedAmendmentCatalogue().decode(SimpleNamespace(config=AmendmentConfig()), bad)


def test_user_pretraining_sampler_preserves_valid_diverse_static_prices():
    episode = SimpleNamespace(config=AmendmentConfig())
    catalogue = ExpandedAmendmentCatalogue()
    rng = np.random.default_rng(24)
    samples = [catalogue.sample(episode, rng) for _ in range(1000)]
    assert all(raw.shape == (16,) and raw[-1] in (0,1) for raw in samples)
    assert all(np.all((raw[:15] >= 0) & (raw[:15] < episode.config.platform_nvec)) for raw in samples)
    assert any(raw[-1] == 0 for raw in samples)
    expanded = {tuple(catalogue.decode(episode,index)) for index in range(78,204)}
    assert any(tuple(raw) in expanded for raw in samples)
    assert any(np.all(raw[:12] == raw[0]) and raw[0] != 0 for raw in samples)


def test_existing_public_observer_accepts_all_204_actions_without_curriculum_mask():
    config = AmendmentConfig(jobs_per_user=1, batch_release_steps=(13,))
    legacy = capped_episode_factory(config)(4)
    while not legacy.done:
        legacy.step_user(0)
    booking = OutsideBooking.from_episode(legacy, "original_test_bookings")
    factory = amendment_episode_factory(lambda seed:booking.scenario, lambda seed:booking.copy(), config)
    episode = factory(4)
    catalogue = ExpandedAmendmentCatalogue(config)
    observer = AmendmentTariffObserver(catalogue)
    observer.training_max_discount = 0.
    before = tuple(episode.selected), episode.bills.copy(), episode.platform_action.copy()
    values = observer(episode,training=False)
    assert values.shape == (204,165) and np.isfinite(values).all()
    assert np.all(values[:,-1] == 1)
    assert np.any(observer(episode,training=True)[:,-1] == 0)
    np.testing.assert_array_equal(episode.bills,before[1])
    np.testing.assert_array_equal(episode.platform_action,before[2])
    assert tuple(episode.selected) == before[0] and episode.position == 0
