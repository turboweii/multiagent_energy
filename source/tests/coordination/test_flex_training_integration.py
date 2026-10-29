"""Actual PPO updates and checkpoint round trips for the sharing contract."""
from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from energy_net.coordination.flex_market import FlexMarketConfig, FlexPlatformEnv, flex_episode_factory
from energy_net.coordination.flex_tariffs import FlexTariffCatalogue, FlexTariffObserver
from energy_net.coordination.learning import ActorEnv, PolicyBundle, alternating_train
from energy_net.coordination.platform_policy import CandidateTariffActorCriticPolicy
from energy_net.coordination.policies import ContractActorCriticPolicy


def parameters(model):
    return [tensor.detach().clone() for tensor in model.policy.parameters()]


def equal_parameters(first, second):
    return all(torch.equal(a, b) for a, b in zip(first, second))


def make_models(config):
    factory = flex_episode_factory(config, maxsize=2)
    catalogue = FlexTariffCatalogue(config)
    observer = FlexTariffObserver(catalogue)
    mean, scale = observer.fit_normalization(factory, [0])
    bundle = PolicyBundle(platform_codec=catalogue, platform_observer=observer, training=True,
                          sample_opponents=True, opponent_sampling_probability=1.,
                          price_exploration_probability=.5, exploration_rng=np.random.default_rng(0))
    actors = ["platform", 0, 1, 2]
    envs = {actor: (FlexPlatformEnv if actor == "platform" else ActorEnv)(
        factory, bundle, actor, [0], factory(0)) for actor in actors}
    models = {}
    for index, actor in enumerate(actors):
        gym.Env.reset(envs[actor], seed=index)
        policy = CandidateTariffActorCriticPolicy if actor == "platform" else ContractActorCriticPolicy
        kwargs = (dict(feature_mean=mean, feature_scale=scale, actor_hidden=(8, 8), value_hidden=(8, 8))
                  if actor == "platform" else dict(economic_only=True, participation_constraint=True,
                                                     contract_hidden_sizes=(8, 8), value_hidden_sizes=(8, 8)))
        models[actor] = PPO(policy, envs[actor], n_steps=2, batch_size=2, n_epochs=1,
                            learning_rate=3e-4, gamma=0., ent_coef=.01,
                            seed=index, policy_kwargs=kwargs, device="cpu", verbose=0)
        if actor == "platform":
            bundle.platform = models[actor]
        else:
            bundle.users[actor] = models[actor]
    return bundle, envs, models, factory


def test_real_four_ppo_alternating_updates_and_save_load_keep_new_observation(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    bundle, envs, initial, factory = make_models(FlexMarketConfig())
    previous = {actor: parameters(model) for actor, model in initial.items()}
    phases = []
    selected_cycle = []
    try:
        def check_phase(record, models):
            active = "platform" if record["actor"] == "platform" else int(record["actor"])
            for actor, model in models.items():
                assert model is initial[actor]
                current = parameters(model)
                if actor != active:
                    assert equal_parameters(previous[actor], current)
                previous[actor] = current
            assert record["actual_actor_steps"] == 2
            assert record["optimizer_steps"] == 1
            phases.append(record["actor"])
            # The training driver's callback uses the same public record type.
            if record["actor"] == "2":
                selected_cycle.append(record["cycle"] + 1)
                for actor, model in models.items():
                    model.save(tmp_path / ("platform" if actor == "platform" else f"user_{actor}"))
        models, records = alternating_train(bundle, envs, cycles=1, platform_days=2,
            user_steps=2, seed=0, initial_models=initial, on_phase=check_phase)
        assert phases == ["platform", "0", "1", "2"]
        assert selected_cycle == [1]
        assert len(records) == 4 and all(model.num_timesteps == 2 for model in models.values())
        bundle.training = False
        prototype = factory(0)
        observation = bundle.platform_observation(prototype)
        assert observation.shape == (77, 186)
        loaded_platform = PPO.load(tmp_path / "platform.zip", device="cpu")
        original_tensor, _ = bundle.platform.policy.obs_to_tensor(observation)
        loaded_tensor, _ = loaded_platform.policy.obs_to_tensor(observation)
        with torch.no_grad():
            original_probs = bundle.platform.policy.get_distribution(original_tensor).distribution.probs
            loaded_probs = loaded_platform.policy.get_distribution(loaded_tensor).distribution.probs
        torch.testing.assert_close(original_probs, loaded_probs, rtol=0, atol=0)
        for owner in range(3):
            loaded = PPO.load(tmp_path / f"user_{owner}.zip", device="cpu")
            assert equal_parameters(parameters(bundle.users[owner]), parameters(loaded))
        assert loaded_platform.observation_space.shape == (77, 186)
        assert loaded_platform.action_space.n == 77
    finally:
        for env in envs.values():
            env.close()
        torch.set_num_threads(previous_threads)


def test_actual_platform_transition_rewards_settled_cash_not_resource_or_internal_valuation():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    # A nonzero private carbon valuation makes cash and internal utility
    # different quantities, so the environment's cash objective is observable.
    config = replace(FlexMarketConfig(), internal_carbon_value_yuan_per_t=100.)
    bundle, envs, _, _ = make_models(config)
    platform_env = envs["platform"]
    found_distinct = False
    try:
        for menu in (4, 8, 12, 16, 20, 24, 28, 32, 40, 44, 48, 60, 76):
            platform_env.reset(seed=0)
            _, reward, done, truncated, info = platform_env.step(menu)
            assert done and not truncated
            metrics = info["market_metrics"]
            assert reward == pytest.approx(metrics["delta_platform_profit_yuan"] / 1000.)
            assert metrics["cash_change_after_transfer_yuan"] == pytest.approx(
                metrics["cash_change_before_transfer_yuan"] + metrics["flexibility_settlement_yuan"])
            assert metrics["surplus_identity_error_yuan"] == pytest.approx(0., abs=1e-7)
            social_return = -metrics["delta_system_cost_yuan"] / 1000.
            internal_return = metrics["delta_platform_objective_yuan"] / 1000.
            if abs(reward - social_return) > 1e-4 and abs(reward - internal_return) > 1e-4:
                found_distinct = True
                break
        assert found_distinct, "The real sampled PPO users should exercise a nonzero flexibility contract"
    finally:
        for env in envs.values():
            env.close()
        torch.set_num_threads(previous_threads)
