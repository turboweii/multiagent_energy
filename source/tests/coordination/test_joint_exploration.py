"""Compact actions, diverse training opponents and deterministic evaluation."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from energy_net.coordination.learning import (
    ActorEnv, PolicyBundle, alternating_train, evaluate_bundle,
)
from energy_net.coordination.market import MarketEpisode
from energy_net.coordination.scenario import make_scenario
from energy_net.coordination.tariffs import TariffCatalogue


class RecordingPolicy:
    def __init__(self, action):
        self.action = action
        self.calls = []

    def predict(self, observation, deterministic=False):
        self.calls.append(bool(deterministic))
        return self.action, None


class RecordingCatalogue(TariffCatalogue):
    def __init__(self):
        super().__init__()
        self.sample_calls = 0

    def sample(self, episode, rng):
        self.sample_calls += 1
        return super().sample(episode, rng)


@pytest.fixture(scope="module")
def scenario():
    return make_scenario(0)


def test_compact_platform_env_executes_supplied_index(scenario):
    platform = RecordingPolicy(0)
    users = {owner: RecordingPolicy(1) for owner in range(3)}
    catalogue = TariffCatalogue()
    bundle = PolicyBundle(platform, users, platform_codec=catalogue,
                          training=True, sample_opponents=True)
    env = ActorEnv(lambda _: MarketEpisode(scenario), bundle, "platform", [0])
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 55
    env.reset(seed=0)
    expected = catalogue.decode(env.episode, 54)
    _, reward, terminated, truncated, info = env.step(np.array(54))
    np.testing.assert_array_equal(env.episode.platform_action, expected)
    assert terminated and not truncated
    assert reward == pytest.approx(env.episode.platform_reward)
    assert info["market_metrics"]["completion_rate"] == pytest.approx(1)
    assert platform.calls == []
    assert all(policy.calls == [False] * 4 for policy in users.values())


def test_user_resets_cover_quotes_and_sample_frozen_opponents(scenario):
    platform = RecordingPolicy(0)
    users = {owner: RecordingPolicy(0) for owner in range(3)}
    catalogue = RecordingCatalogue()
    bundle = PolicyBundle(platform, users, platform_codec=catalogue,
                          training=True, sample_opponents=True,
                          price_exploration_probability=1,
                          exploration_rng=np.random.default_rng(19))
    env = ActorEnv(lambda _: MarketEpisode(scenario), bundle, 0, [0])
    quotes = []
    for trial in range(40):
        env.reset(seed=trial)
        quotes.append(tuple(env.episode.platform_action))
        while True:
            _, _, done, _, _ = env.step(0)
            if done:
                break
    assert catalogue.sample_calls == 40
    assert len(set(quotes)) > 15
    assert platform.calls == []
    assert users[0].calls == []
    for owner in (1, 2):
        assert len(users[owner].calls) == 40 * 4
        assert not any(users[owner].calls)


@pytest.mark.parametrize("training_before", [False, True])
def test_evaluation_disables_all_exploration_and_restores_mode(scenario, training_before):
    catalogue = RecordingCatalogue()
    platform = RecordingPolicy(54)
    users = {owner: RecordingPolicy(0) for owner in range(3)}
    bundle = PolicyBundle(platform, users, platform_codec=catalogue,
                          training=training_before, sample_opponents=True,
                          opponent_sampling_probability=.1,
                          price_exploration_probability=1,
                          exploration_rng=np.random.default_rng(17))
    state_before = bundle.exploration_rng.bit_generator.state
    rows = evaluate_bundle(bundle, lambda _: MarketEpisode(scenario), [10000, 10001])
    assert bundle.training is training_before
    assert bundle.sample_opponents is True
    assert bundle.opponent_sampling_probability == .1
    assert bundle.price_exploration_probability == 1
    assert bundle.exploration_rng.bit_generator.state == state_before
    assert catalogue.sample_calls == 0
    assert platform.calls == [True, True]
    assert all(policy.calls == [True] * 8 for policy in users.values())
    expected = catalogue.decode(MarketEpisode(scenario), 54)
    assert len(rows) == 2
    for row in rows:
        np.testing.assert_array_equal(row["platform_action"], expected)
        assert row["completion_rate"] == pytest.approx(1)
        assert row["deadline_compliance"] == pytest.approx(1)


@pytest.mark.parametrize("sampling_probability", [0., 1., .1])
def test_opponent_sampling_probability_is_applied_and_reproducible(scenario, sampling_probability):
    """Check observable inference requests without expecting an exact draw count."""
    episode = MarketEpisode(scenario)

    def draw_sequence():
        platform = RecordingPolicy(0)
        users = {owner: RecordingPolicy(0) for owner in range(3)}
        bundle = PolicyBundle(
            platform, users, platform_codec=TariffCatalogue(), training=True,
            sample_opponents=True, opponent_sampling_probability=sampling_probability,
            price_exploration_probability=0,
            exploration_rng=np.random.Generator(np.random.PCG64(23)),
        )
        for _ in range(200):
            bundle.platform_action(episode)
            bundle.user_action(episode)
        return platform.calls, users[episode.current_user_id].calls

    first, repeated = draw_sequence(), draw_sequence()
    assert first == repeated
    for inference_flags in first:
        assert len(inference_flags) == 200
        if sampling_probability == 0:
            assert all(inference_flags)
        elif sampling_probability == 1:
            assert not any(inference_flags)
        else:
            assert set(inference_flags) == {False, True}


def test_evaluation_restores_training_after_policy_failure(scenario):
    class FailingPolicy:
        def predict(self, observation, deterministic=False):
            assert deterministic
            raise RuntimeError("test policy failure")

    bundle = PolicyBundle(FailingPolicy(), platform_codec=TariffCatalogue(),
                          training=True, sample_opponents=True,
                          price_exploration_probability=1)
    with pytest.raises(RuntimeError, match="test policy failure"):
        evaluate_bundle(bundle, lambda _: MarketEpisode(scenario), [0])
    assert bundle.training is True


def _parameters(model):
    return [parameter.detach().clone() for parameter in model.policy.parameters()]


def _same(first, second):
    return all(torch.equal(a, b) for a, b in zip(first, second))


def test_joint_training_reuses_models_and_updates_only_active_actor(scenario, monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    bundle = PolicyBundle(platform_codec=TariffCatalogue(), training=True,
                          sample_opponents=True, price_exploration_probability=.25,
                          exploration_rng=np.random.default_rng(7))
    actors = ["platform", 0, 1, 2]
    envs = {actor: ActorEnv(lambda _: MarketEpisode(scenario), bundle, actor, [0])
            for actor in actors}
    try:
        initial = {actor: PPO("MlpPolicy", envs[actor], n_steps=2, batch_size=2,
                              n_epochs=1, seed=index, device="cpu", verbose=0,
                              policy_kwargs={"net_arch": [8, 8]})
                   for index, actor in enumerate(actors)}
        for index, actor in enumerate(actors):
            initial[actor].num_timesteps = 10 + 2 * index
        previous = {actor: _parameters(model) for actor, model in initial.items()}
        phases = []

        def forbid_new_model(*args, **kwargs):
            raise AssertionError("Joint training must reuse the supplied model")

        monkeypatch.setattr(PPO, "__init__", forbid_new_model)

        def check_phase(record, models):
            active = "platform" if record["actor"] == "platform" else int(record["actor"])
            for actor, model in models.items():
                assert model is initial[actor]
                current = _parameters(model)
                assert _same(previous[actor], current) is (actor != active)
                previous[actor] = current
            assert record["actual_actor_steps"] == 2
            assert record["optimizer_steps"] == 1
            start = 10 + 2 * actors.index(active)
            assert record["cumulative_actor_steps"] == start + 2
            phases.append(active)

        models, records = alternating_train(
            bundle, envs, cycles=1, platform_days=2, user_steps=2,
            seed=0, initial_models=initial, on_phase=check_phase)
        assert phases == actors
        assert len(records) == 4
        assert bundle.platform is initial["platform"]
        assert all(bundle.users[owner] is initial[owner] for owner in range(3))
        assert all(models[actor] is initial[actor] for actor in actors)
    finally:
        torch.set_num_threads(old_threads)
        for env in envs.values():
            env.close()
