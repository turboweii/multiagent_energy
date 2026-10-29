"""Learnable menu priors stay inside PPO and preserve contextual warm starts."""
from __future__ import annotations

import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from energy_net.coordination.flex_policy import (
    FlexTariffActorCriticPolicy, warm_start_with_action_bias,
)
from energy_net.coordination.platform_policy import CandidateTariffActorCriticPolicy


class CandidateEnv(gym.Env):
    """Small synthetic environment for policy mechanics, not market results."""
    def __init__(self):
        self.action_space = spaces.Discrete(77)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(77, 186), dtype=np.float32)
        self.observation = np.random.default_rng(13).normal(size=(77, 186)).astype(np.float32)
        self.observation[:, -1] = 1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation.copy(), {}

    def step(self, action):
        return self.observation.copy(), float(self.observation[int(action), 0]), True, False, {}


def build(policy):
    return PPO(policy, CandidateEnv(), n_steps=4, batch_size=4, n_epochs=1,
               gamma=0, seed=0, device="cpu", verbose=0,
               policy_kwargs=dict(actor_hidden=(8, 8), value_hidden=(8, 8)))


def probabilities(model, observation):
    tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        return model.policy.get_distribution(tensor).distribution.probs.detach().clone()


def test_new_menu_bias_starts_at_zero_and_can_change_identical_context_scores():
    model = build(FlexTariffActorCriticPolicy)
    bias = model.policy.mlp_extractor.action_bias
    torch.testing.assert_close(bias, torch.zeros(77), rtol=0, atol=0)
    observation = np.zeros((77, 186), dtype=np.float32)
    observation[:, -1] = 1
    before = probabilities(model, observation)
    torch.testing.assert_close(before, torch.full((1, 77), 1 / 77), rtol=1e-6, atol=1e-8)
    with torch.no_grad():
        bias[7] = 1
    after = probabilities(model, observation)
    assert after[0, 7] > after[0, 6]
    assert after[0, 6] == after[0, 8]


def test_menu_bias_is_in_the_categorical_log_probability_and_respects_masks():
    model = build(FlexTariffActorCriticPolicy)
    observation, _ = model.get_env().envs[0].reset()
    tensor, _ = model.policy.obs_to_tensor(observation)
    distribution = model.policy.get_distribution(tensor)
    loss = -distribution.log_prob(torch.tensor([7])).mean()
    model.policy.optimizer.zero_grad()
    loss.backward()
    gradient = model.policy.mlp_extractor.action_bias.grad
    assert gradient[7] < 0 and gradient[6] > 0
    with torch.no_grad():
        model.policy.mlp_extractor.action_bias[7] = 100
    observation[7, -1] = 0
    assert probabilities(model, observation)[0, 7] == 0
    observation[:, -1] = 0
    expected = torch.zeros((1, 77)); expected[0, 0] = 1
    torch.testing.assert_close(probabilities(model, observation), expected, rtol=0, atol=0)


def test_warmstart_preserves_old_probabilities_context_weights_adam_and_counters():
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        source = build(CandidateTariffActorCriticPolicy)
        source.learn(total_timesteps=4)
        source_weights = {name: parameter.detach().clone() for name, parameter in source.policy.named_parameters()}
        observation = CandidateEnv().observation
        before = probabilities(source, observation)
        target = warm_start_with_action_bias(source, CandidateEnv())
        torch.testing.assert_close(probabilities(target, observation), before, rtol=0, atol=0)
        assert target.num_timesteps == source.num_timesteps == 4
        assert target._n_updates == source._n_updates == 1
        target_parameters = dict(target.policy.named_parameters())
        for name, original in source.policy.named_parameters():
            current = target_parameters[name]
            torch.testing.assert_close(current, original, rtol=0, atol=0)
            torch.testing.assert_close(original, source_weights[name], rtol=0, atol=0)
            original_state = source.policy.optimizer.state.get(original, {})
            current_state = target.policy.optimizer.state.get(current, {})
            assert original_state.keys() == current_state.keys()
            for field, value in original_state.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(current_state[field], value, rtol=0, atol=0)
                    assert current_state[field].data_ptr() != value.data_ptr()
                else:
                    assert current_state[field] == value
        bias = target.policy.mlp_extractor.action_bias
        assert bias not in target.policy.optimizer.state
        torch.testing.assert_close(bias, torch.zeros(77), rtol=0, atol=0)
        assert any(parameter is bias for group in target.policy.optimizer.param_groups for parameter in group["params"])
        target.learn(total_timesteps=4, reset_num_timesteps=False)
        assert target.num_timesteps == 8
        assert bias in target.policy.optimizer.state
        assert not torch.equal(bias.detach(), torch.zeros(77))
    finally:
        torch.set_num_threads(threads)


def test_warmstart_construction_preserves_the_callers_random_streams():
    source = build(CandidateTariffActorCriticPolicy)
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    warm_start_with_action_bias(source, CandidateEnv())
    assert random.getstate() == python_state
    actual_np = np.random.get_state()
    assert actual_np[0] == numpy_state[0]
    np.testing.assert_array_equal(actual_np[1], numpy_state[1])
    assert actual_np[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)


def test_new_policy_and_bias_round_trip_through_ppo_save_load(tmp_path):
    source = build(CandidateTariffActorCriticPolicy)
    target = warm_start_with_action_bias(source, CandidateEnv())
    with torch.no_grad():
        target.policy.mlp_extractor.action_bias.copy_(torch.linspace(-1, 1, 77))
    target.save(tmp_path / "flex_policy")
    restored = PPO.load(tmp_path / "flex_policy.zip", device="cpu")
    assert isinstance(restored.policy, FlexTariffActorCriticPolicy)
    torch.testing.assert_close(probabilities(restored, CandidateEnv().observation),
                               probabilities(target, CandidateEnv().observation), rtol=0, atol=0)


def test_warmstart_rejects_different_candidate_spaces():
    source = build(CandidateTariffActorCriticPolicy)
    changed = CandidateEnv()
    changed.action_space = spaces.Discrete(76)
    with pytest.raises(ValueError, match="same fixed"):
        warm_start_with_action_bias(source, changed)
