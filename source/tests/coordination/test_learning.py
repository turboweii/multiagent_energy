"""Real-market actor turns, private rewards and short independent PPO updates."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from energy_net.coordination.learning import ActorEnv, PolicyBundle, alternating_train
from energy_net.coordination.market import MarketEpisode, encode_platform_action
from energy_net.coordination.scenario import make_scenario


@pytest.fixture(scope="module")
def scenario():
    return make_scenario(0)


class RecordingPolicy:
    """An unchanged opponent that records inference, never receives updates."""

    def __init__(self, action):
        self.action = action
        self.calls = []

    def predict(self, observation, deterministic=False):
        self.calls.append((np.array(observation, copy=True), deterministic))
        return self.action, None


def discount_action(scenario):
    return encode_platform_action(np.full((3, 4), 0.15), config=scenario.config)


@pytest.mark.parametrize("actor", ["platform", 0, 1, 2])
def test_real_actor_interfaces_keep_opponent_rewards_and_steps_separate(scenario, actor):
    platform = RecordingPolicy(discount_action(scenario))
    users = {owner: RecordingPolicy(1) for owner in range(3)}
    bundle = PolicyBundle(platform, users)
    env = ActorEnv(lambda seed: MarketEpisode(scenario), bundle, actor, [0])
    observation, reset_info = env.reset(seed=17)
    assert reset_info["scenario_seed"] == 0
    assert env.observation_space.contains(observation)
    assert observation.shape == ((485,) if actor == "platform" else (92,))
    assert env.action_space.shape == ((15,) if actor == "platform" else ())

    rewards = []
    while True:
        # Users actively select their retained original contracts. Opponents
        # reprice identical execution via option 1 and earn positive utility.
        action = discount_action(scenario) if actor == "platform" else 0
        observation, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(observation)
        assert not truncated
        rewards.append(reward)
        if actor != "platform":
            assert info["user_id"] == actor
            assert info["chosen"] == 0
            assert reward == pytest.approx(0.0)
        if terminated:
            break

    assert env.episode.position == len(scenario.jobs) == 12
    assert env.completed_days == 1
    assert env.actor_steps == len(rewards) == (1 if actor == "platform" else 4)
    metrics = info["market_metrics"]
    assert metrics["completion_rate"] == pytest.approx(1.0)
    assert metrics["deadline_compliance"] == 1.0
    np.testing.assert_array_equal(observation, np.zeros_like(observation))
    if actor == "platform":
        assert not platform.calls  # The action passed to step is executed.
        assert rewards == pytest.approx([env.episode.platform_reward])
        assert rewards[0] < 0  # Discounts reduce profit with unchanged physics.
        assert metrics["user_gain_yuan"] > 0
    else:
        assert len(platform.calls) == 1
        assert not users[actor].calls  # The active user's model is bypassed.
        assert metrics[f"user_{actor}_gain_yuan"] == pytest.approx(0.0)
        assert metrics["user_gain_yuan"] > 0
        assert env.episode.platform_reward < 0
    for owner, policy in users.items():
        assert len(policy.calls) == (0 if owner == actor else 4)
        assert all(deterministic for _, deterministic in policy.calls)
    assert all(deterministic for _, deterministic in platform.calls)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(discount_action(scenario) if actor == "platform" else 0)


def _parameters(model):
    return [parameter.detach().cpu().clone() for parameter in model.policy.parameters()]


def _same_parameters(first, second):
    return all(torch.equal(left, right) for left, right in zip(first, second))


def test_tiny_actual_ppo_updates_all_four_models_and_freezes_each_opponent(scenario, monkeypatch):
    """Four brief learning calls verify updates, not training quality."""
    original_init = PPO.__init__
    initialized = {}

    def record_initial_parameters(model, *args, **kwargs):
        original_init(model, *args, **kwargs)
        initialized[id(model)] = _parameters(model)

    monkeypatch.setattr(PPO, "__init__", record_initial_parameters)
    bundle = PolicyBundle()
    environments = {
        actor: ActorEnv(lambda seed: MarketEpisode(scenario), bundle, actor, [0])
        for actor in ["platform", 0, 1, 2]
    }
    previous = {}
    phases = []

    def check_phase(record, models):
        active = "platform" if record["actor"] == "platform" else int(record["actor"])
        for actor, model in models.items():
            before = previous.get(actor, initialized[id(model)])
            after = _parameters(model)
            if actor == active:
                assert not _same_parameters(before, after), f"{actor} did not learn"
            else:
                assert _same_parameters(before, after), f"Opponent {actor} was updated"
            previous[actor] = after
            # Opponent market turns never count as another learner's samples.
            assert environments[actor].actor_steps == model.num_timesteps
        assert record["actual_actor_steps"] == (2 if active == "platform" else 4)
        assert record["optimizer_steps"] > 0
        assert record["ppo_training_epochs"] > 0
        phases.append(active)

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        models, records = alternating_train(
            bundle, environments, cycles=1, platform_days=2, user_steps=4,
            seed=123, on_phase=check_phase,
        )
    finally:
        torch.set_num_threads(old_threads)
        for environment in environments.values():
            environment.close()

    assert phases == ["platform", 0, 1, 2]
    assert len(records) == len(models) == 4
    assert bundle.platform is models["platform"]
    assert all(bundle.users[owner] is models[owner] for owner in range(3))
    assert len({id(model) for model in models.values()}) == 4
    storage = [{parameter.data_ptr() for parameter in model.policy.parameters()}
               for model in models.values()]
    for first in range(4):
        for second in range(first):
            assert storage[first].isdisjoint(storage[second])
