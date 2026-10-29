"""Public catalogue actions retain stable meaning and legal market inputs."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from energy_net.coordination.market import MarketEpisode, zero_platform_action
from energy_net.coordination.scenario import MarketConfig, make_scenario
from energy_net.coordination.tariffs import TariffCatalogue


@pytest.fixture(scope="module")
def episode():
    return MarketEpisode(make_scenario(0))


def test_catalogue_has_stable_labels_and_legal_actions(episode):
    catalogue = TariffCatalogue()
    assert catalogue.n_actions == 55
    assert len(set(catalogue.labels)) == 55
    assert catalogue.labels[0] == "original"
    assert catalogue.labels[1] == "time_only_discount_0.05_quota_0.50"
    assert catalogue.labels[-1] == "time_region_discount_0.30_quota_1.00"
    assert catalogue.labels == TariffCatalogue().labels
    for index in range(catalogue.n_actions):
        action = catalogue.decode(episode, np.array([index]))
        assert action.shape == (15,)
        assert np.issubdtype(action.dtype, np.integer)
        assert np.all(action >= 0)
        assert np.all(action < episode.config.platform_nvec)
    assert np.array_equal(catalogue.decode(episode, 0), zero_platform_action(episode.config))


def test_quote_direction_and_pattern_geometry():
    cfg = MarketConfig()
    # Turn energy cost into a deliberately separable region-and-time surface.
    costs = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], dtype=float)
    lmp = np.repeat(costs.T, 12, axis=0) / np.asarray(cfg.energy_mwh_per_cu)
    probe = SimpleNamespace(config=cfg, baseline_clearing=SimpleNamespace(lmp_yuan_per_mwh=lmp))
    catalogue = TariffCatalogue()
    scores = catalogue.pattern_scores(probe)
    assert np.allclose(scores["time_only"], scores["time_only"][0])
    assert np.allclose(scores["region_only"], scores["region_only"][:, :1])
    assert scores["time_region"][0, 0] == pytest.approx(1)
    assert scores["time_region"][-1, -1] == pytest.approx(0)
    for index in range(1, catalogue.n_actions):
        action = catalogue.decode(probe, index)
        discounts = np.asarray(cfg.discount_levels)[action[:12]].reshape(3, 4)
        pattern = catalogue.entries[index][0]
        if pattern == "time_only":
            assert np.all(discounts[:, :-1] >= discounts[:, 1:])
            assert np.all(discounts == discounts[0])
        elif pattern == "region_only":
            assert np.all(discounts[:-1] >= discounts[1:])
            assert np.all(discounts == discounts[:, :1])
        else:
            assert discounts[0, 0] >= discounts[-1, -1]


def test_private_preferences_do_not_change_catalogue(episode):
    changed_jobs = tuple(replace(job,
                                 beta_yuan_per_cu_hour=job.beta_yuan_per_cu_hour * 10,
                                 migration_loss_yuan_per_cu=job.migration_loss_yuan_per_cu * 20)
                         for job in episode.scenario.jobs)
    other = MarketEpisode(replace(episode.scenario, jobs=changed_jobs))
    catalogue = TariffCatalogue()
    for index in range(catalogue.n_actions):
        assert np.array_equal(catalogue.decode(episode, index), catalogue.decode(other, index))


def test_diverse_sampler_is_seeded_and_valid(episode):
    catalogue = TariffCatalogue()
    first, second = np.random.default_rng(4), np.random.default_rng(4)
    actions = [catalogue.sample(episode, first) for _ in range(100)]
    assert all(np.array_equal(action, catalogue.sample(episode, second)) for action in actions)
    assert len(set(tuple(action) for action in actions)) > 20
    for action in actions:
        assert np.all(action >= 0)
        assert np.all(action < episode.config.platform_nvec)


@pytest.mark.parametrize("index", [-1, 55, 1.5, [1, 2]])
def test_invalid_catalogue_index_is_rejected(episode, index):
    with pytest.raises(ValueError):
        TariffCatalogue().decode(episode, index)
