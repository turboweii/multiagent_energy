"""Contract scores follow option features and participate in real PPO updates."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from energy_net.coordination.learning import ActorEnv, PolicyBundle
from energy_net.coordination.market import MarketEpisode, encode_platform_action
from energy_net.coordination.policies import ContractActorCriticPolicy
from energy_net.coordination.scenario import make_scenario


@pytest.fixture(scope="module")
def scenario():
    return make_scenario(0)


def observation(scenario):
    episode = MarketEpisode(scenario)
    episode.set_platform_action(encode_platform_action(np.full((3, 4), .15)))
    return episode.current_user_observation()


def policy(economic_only=False):
    torch.manual_seed(7)
    return ContractActorCriticPolicy(
        spaces.Box(-np.inf, np.inf, shape=(92,), dtype=np.float32), spaces.Discrete(10),
        lambda fraction: 3e-4,
        economic_only=economic_only,
    )


@pytest.mark.parametrize("economic_only", [False, True])
def test_actor_scores_and_probabilities_follow_permuted_contracts(scenario, economic_only):
    actor = policy(economic_only)
    original = torch.as_tensor(observation(scenario)[None, :])
    permuted = original.clone()
    permutation = torch.tensor([4, 2, 9, 0, 6, 8, 1, 7, 3, 5])
    permuted[:, 6:76] = original[:, 6:76].reshape(1, 10, 7)[:, permutation].reshape(1, 70)
    with torch.no_grad():
        first = actor.mlp_extractor.forward_actor(original)
        second = actor.mlp_extractor.forward_actor(permuted)
        first_prob = actor.get_distribution(original).distribution.probs
        second_prob = actor.get_distribution(permuted).distribution.probs
    torch.testing.assert_close(second, first[:, permutation])
    torch.testing.assert_close(second_prob, first_prob[:, permutation])


@pytest.mark.parametrize("economic_only", [False, True])
def test_missing_or_precheck_infeasible_contracts_have_zero_probability(scenario, economic_only):
    actor = policy(economic_only)
    sample = torch.as_tensor(observation(scenario)[None, :]).clone()
    options = sample[:, 6:76].reshape(1, 10, 7)
    options[:, 2, 0] = 0.0
    options[:, 3, 1] = 0.0
    with torch.no_grad():
        probability = actor.get_distribution(sample).distribution.probs[0]
        empty = actor.get_distribution(torch.zeros_like(sample)).distribution.probs[0]
    assert probability[2] == 0 and probability[3] == 0
    assert probability.sum() == pytest.approx(1.0)
    assert empty[0] == 1.0 and empty[1:].sum() == 0.0


@pytest.mark.parametrize("economic_only", [False, True])
def test_price_is_a_differentiable_input_to_its_shared_contract_score(scenario, economic_only):
    actor = policy(economic_only)
    sample = torch.as_tensor(observation(scenario)[None, :]).clone().requires_grad_(True)
    logits = actor.mlp_extractor.forward_actor(sample)
    # Use the guaranteed original contract; no assumption about the initial
    # economic direction of a randomly initialized network is built in.
    logits[0, 0].backward()
    assert abs(sample.grad[0, 6 + 2].item()) > 1e-8
    assert torch.count_nonzero(sample.grad[0, 6 + 7:76]) == 0
    changed = sample.detach().clone()
    changed[0, 6 + 2] += .2
    with torch.no_grad():
        other = actor.mlp_extractor.forward_actor(changed)
    assert abs(other[0, 0] - logits.detach()[0, 0]) > 1e-6
    torch.testing.assert_close(other[:, 1:], logits.detach()[:, 1:])


class FixedPolicy:
    def __init__(self, action):
        self.action = action

    def predict(self, observation, deterministic=False):
        return self.action, None


@pytest.mark.parametrize("economic_only", [False, True])
def test_real_ppo_updates_shared_actor_and_save_load_preserves_predictions(scenario, tmp_path, economic_only):
    bundle = PolicyBundle(
        FixedPolicy(encode_platform_action(np.full((3, 4), .15))),
        {owner: FixedPolicy(0) for owner in range(3)},
    )
    env = ActorEnv(lambda seed: MarketEpisode(scenario), bundle, 0, [0])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = PPO(ContractActorCriticPolicy, env, n_steps=8, batch_size=8, n_epochs=2,
                    policy_kwargs=dict(economic_only=economic_only),
                    seed=7, device="cpu", verbose=0)
        before = [value.detach().clone() for value in
                  model.policy.mlp_extractor.option_score.parameters()]
        optimizer_parameters = {id(value) for group in model.policy.optimizer.param_groups
                                for value in group["params"]}
        assert all(id(value) in optimizer_parameters
                   for value in model.policy.mlp_extractor.option_score.parameters())
        model.learn(16)
        after = list(model.policy.mlp_extractor.option_score.parameters())
        assert any(not torch.equal(left, right) for left, right in zip(before, after))
        sample = observation(scenario)
        tensor, _ = model.policy.obs_to_tensor(sample)
        with torch.no_grad():
            expected = model.policy.get_distribution(tensor).distribution.probs
        model.save(tmp_path / "contract")
        restored = PPO.load(tmp_path / "contract", device="cpu")
        assert restored.policy.economic_only is economic_only
        with torch.no_grad():
            actual = restored.policy.get_distribution(tensor).distribution.probs
        torch.testing.assert_close(actual, expected)
        np.testing.assert_array_equal(restored.predict(sample, deterministic=True)[0],
                                      model.predict(sample, deterministic=True)[0])
    finally:
        torch.set_num_threads(previous_threads)
        env.close()


def test_economic_actor_uses_costs_without_region_or_delay_shortcuts(scenario):
    actor = policy(economic_only=True)
    sample = torch.as_tensor(observation(scenario)[None, :]).clone()
    changed = sample.clone()
    changed[:, :6] += 1.0
    changed[:, 6:76].reshape(1, 10, 7)[:, :, 4:] += .5
    changed[:, 76:] -= .25
    with torch.no_grad():
        original_logits = actor.mlp_extractor.forward_actor(sample)
        changed_logits = actor.mlp_extractor.forward_actor(changed)
    torch.testing.assert_close(changed_logits, original_logits)
