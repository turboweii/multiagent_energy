from dataclasses import replace

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from energy_net.coordination.learning import cached_episode_factory
from energy_net.coordination.market import MarketEpisode
from energy_net.coordination.platform_features import candidate_tariff_observation, feature_names
from energy_net.coordination.platform_policy import CandidateTariffActorCriticPolicy
from energy_net.coordination.tariffs import TariffCatalogue


@pytest.fixture(scope="module")
def episode():
    return cached_episode_factory()(0)


def test_candidate_public_features_shape_and_do_not_change_contracts(episode):
    action = episode.platform_action.copy()
    x = candidate_tariff_observation(episode)
    assert x.shape == (55, len(feature_names())) == (55, 88)
    assert x.dtype == np.float32 and np.isfinite(x).all()
    assert x[0, -1] == 1
    np.testing.assert_array_equal(action, episode.platform_action)
    assert episode.position == 0


def test_candidate_features_do_not_read_private_preference_or_loss(episode):
    changed_jobs = tuple(replace(job, beta_yuan_per_cu_hour=123.0,
                                 migration_loss_yuan_per_cu=456.0,
                                 options=tuple(replace(o, user_loss_yuan=98765.) for o in job.options))
                         for job in episode.scenario.jobs)
    changed = MarketEpisode(replace(episode.scenario, jobs=changed_jobs))
    np.testing.assert_array_equal(candidate_tariff_observation(episode),
                                  candidate_tariff_observation(changed))


def test_forecast_discount_exposure_and_zero_menu(episode):
    x = candidate_tariff_observation(episode)
    names = feature_names()
    assert x[0, names.index("original_invoice_discount_exposure")] == 0
    assert np.max(x[:, names.index("original_invoice_discount_exposure")]) > 0
    for prefix in ("owner_original_invoice_discount", "owner_possible_price_cut",
                   "owner_best_reference_margin", "owner_average_reference_margin"):
        assert np.all(x[0, [names.index(f"{prefix}_{owner}") for owner in range(3)]] == 0)


def test_mask_removes_only_identical_raw_actions(episode):
    class DuplicateCatalogue(TariffCatalogue):
        def decode(self, ep, index):
            return super().decode(ep, 0 if index == 1 else index)

    x = candidate_tariff_observation(episode, DuplicateCatalogue())
    assert x[0, -1] == 1 and x[1, -1] == 0
    np.testing.assert_array_equal(x[0, :-1], x[1, :-1])


class QuoteBandit(gym.Env):
    def __init__(self, observation):
        self.x = observation.copy()
        self.observation_space = spaces.Box(-np.inf, np.inf, self.x.shape, np.float32)
        self.action_space = spaces.Discrete(len(self.x))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.x.copy(), {}

    def step(self, action):
        return self.x.copy(), float(1 - action / (len(self.x) - 1)), True, False, {}


def _model(observation):
    mean = observation[:, :-1].mean(axis=0)
    scale = np.maximum(observation[:, :-1].std(axis=0), .05)
    return PPO(CandidateTariffActorCriticPolicy, QuoteBandit(observation), seed=0,
               n_steps=4, batch_size=4, n_epochs=1,
               policy_kwargs={"feature_mean": mean, "feature_scale": scale})


def test_neutral_initial_logits_and_exact_action_mask(episode):
    torch.set_num_threads(1)
    x = candidate_tariff_observation(episode)
    x[1, -1] = 0
    model = _model(x)
    obs, _ = model.policy.obs_to_tensor(x)
    with torch.no_grad():
        probabilities = model.policy.get_distribution(obs).distribution.probs[0]
    assert probabilities[1] == 0
    np.testing.assert_allclose(probabilities[np.array(x[:, -1] > .5)], 1 / (x[:, -1] > .5).sum())


def test_shared_scorer_is_quote_permutation_equivariant_after_learning(episode):
    torch.set_num_threads(1)
    x = candidate_tariff_observation(episode)
    model = _model(x)
    model.learn(8)
    permutation = np.random.default_rng(0).permutation(len(x))
    with torch.no_grad():
        first = model.policy.mlp_extractor.forward_actor(torch.as_tensor(x.reshape(1, -1)))
        swapped = model.policy.mlp_extractor.forward_actor(torch.as_tensor(x[permutation].reshape(1, -1)))
    torch.testing.assert_close(swapped, first[:, permutation])


def test_real_ppo_updates_actor_and_saves_normalization(episode, tmp_path):
    torch.set_num_threads(1)
    x = candidate_tariff_observation(episode)
    model = _model(x)
    before = model.policy.mlp_extractor.scorer[-1].weight.detach().clone()
    model.learn(16)
    assert not torch.equal(before, model.policy.mlp_extractor.scorer[-1].weight)
    model.save(tmp_path / "platform")
    loaded = PPO.load(tmp_path / "platform.zip")
    assert int(model.predict(x, deterministic=True)[0]) == int(loaded.predict(x, deterministic=True)[0])
    torch.testing.assert_close(model.policy.mlp_extractor.feature_mean,
                               loaded.policy.mlp_extractor.feature_mean)
    torch.testing.assert_close(model.policy.mlp_extractor.feature_scale,
                               loaded.policy.mlp_extractor.feature_scale)
    assert not any(isinstance(layer, torch.nn.Tanh)
                   for layer in loaded.policy.mlp_extractor.modules())
