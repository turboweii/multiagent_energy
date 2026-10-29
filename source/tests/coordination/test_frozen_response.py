"""Cached days remain actual deterministic-policy outcomes with full state."""
from functools import lru_cache

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from energy_net.coordination.amendment_features import AmendmentTariffCatalogue
from energy_net.coordination.amendment_market import AmendmentConfig, AmendmentPlatformEnv, amendment_episode_factory
from energy_net.coordination.capped_market import capped_episode_factory
from energy_net.coordination.frozen_response import FrozenResponsePlatformEnv
from energy_net.coordination.learning import ActorEnv, PolicyBundle
from energy_net.coordination.market import exact_user_policy
from energy_net.coordination.outside_bookings import OutsideBooking


class CountingPrivateNetwork:
    """Local-cost policy fixture, with observable neural weight versions."""
    def __init__(self):
        self.policy = torch.nn.Linear(1, 1)
        self.calls = 0
        self.deterministic_arguments = []
        self.num_timesteps = 0

    def predict(self, observation, deterministic=True):
        self.calls += 1
        self.deterministic_arguments.append(deterministic)
        return np.asarray(exact_user_policy(observation)), None


@lru_cache(maxsize=3)
def booking(seed):
    old = capped_episode_factory()(seed)
    old.play()
    return OutsideBooking.from_episode(old)


def setup(users=None, maxsize=16384):
    config = AmendmentConfig()
    factory = amendment_episode_factory(lambda seed: booking(seed).scenario,
        lambda seed: booking(seed).copy(), config, maxsize=3)
    catalogue = AmendmentTariffCatalogue(config)
    users = users or {user: CountingPrivateNetwork() for user in range(3)}
    bundle = PolicyBundle(users=users, platform_codec=catalogue, training=True)
    env = FrozenResponsePlatformEnv(factory, bundle, "platform", [0], factory(0), maxsize=maxsize)
    return env, factory, bundle


def user_calls(bundle):
    return sum(model.calls for model in bundle.users.values())


def assert_same_outcome(actual, expected):
    assert actual.metrics == expected.metrics
    assert actual.selected_options == expected.selected_options
    assert actual.requested_actions == expected.requested_actions
    assert actual.bills_yuan == expected.bills_yuan
    assert actual.user_losses_yuan == expected.user_losses_yuan
    assert actual.reservations == expected.reservations
    for name in ("generation_mw", "flows_mw", "lmp_yuan_per_mwh"):
        np.testing.assert_array_equal(getattr(actual.clearing, name), getattr(expected.clearing, name))


def test_hit_restores_actual_complete_contracts_profit_counters_and_terminal_interface(monkeypatch):
    env, _, bundle = setup()
    observation, _ = env.reset(seed=0)
    first = env.step(19)
    expected = env.episode.outcome()
    assert first[2] and not first[3]
    assert expected.metrics["accepted_amendments"] > 0
    assert user_calls(bundle) == 12
    solver = env.episode.grid
    # Cached replay must not copy the solver, call users, or solve another OPF.
    next_observation, _ = env.reset(seed=0)
    np.testing.assert_array_equal(observation, next_observation)
    def no_solve(*args, **kwargs):
        raise AssertionError("A cache hit must restore the existing real clearing")
    monkeypatch.setattr(solver, "solve", no_solve)
    second = env.step(19)
    assert user_calls(bundle) == 12
    assert env.episode.grid is solver
    assert env.episode.done and env.episode.current_user_id is None
    assert env.episode.position == len(env.episode.jobs)
    assert env.actor_steps == env.completed_days == 2
    assert env._finished
    assert env.last_outcome is second[4]["market_metrics"]
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    assert_same_outcome(env.episode.outcome(), expected)
    assert env.episode.platform_reward == pytest.approx(first[1])
    assert env.cache_info().hits == env.cache_info().misses == 1
    with pytest.raises(RuntimeError, match="Call reset"):
        env.step(19)


def test_returned_metrics_arrays_and_episode_state_do_not_mutate_cache():
    env, _, bundle = setup()
    env.reset(seed=0)
    first = env.step(19)
    profit = first[4]["market_metrics"]["platform_profit_yuan"]
    first[4]["market_metrics"]["platform_profit_yuan"] = -999
    first[0][:] = -999
    env.episode.bills[:] = -1
    env.episode.current_clearing.generation_mw[:] = -1
    env.episode.reservations[0]["chosen"] = 999
    env.reset(seed=0)
    hit = env.step(19)
    assert hit[4]["market_metrics"]["platform_profit_yuan"] == profit
    assert np.all(hit[0] == 0)
    assert np.all(env.episode.bills >= 0)
    assert np.all(env.episode.current_clearing.generation_mw >= 0)
    assert env.episode.reservations[0]["chosen"] < 10
    assert user_calls(bundle) == 12


@pytest.mark.parametrize("mutation", ["optimizer_style", "data_content", "training_counter", "replace_model", "buffer"])
def test_user_changes_invalidate_instead_of_reusing_a_previous_response(mutation):
    env, _, bundle = setup()
    env.reset(seed=0); env.step(19)
    if mutation == "optimizer_style":
        with torch.no_grad():
            bundle.users[0].policy.weight.add_(1)
    elif mutation == "data_content":
        bundle.users[0].policy.weight.data.add_(1)  # Does not advance torch's version counter.
    elif mutation == "training_counter":
        bundle.users[0].num_timesteps += 128
    elif mutation == "replace_model":
        bundle.users[0] = CountingPrivateNetwork()
    else:
        bundle.users[0].policy.register_buffer("running_mean", torch.tensor(2.))
    before = user_calls(bundle)
    env.reset(seed=0); env.step(19)
    assert user_calls(bundle) == before + 12
    assert env.cache_info().invalidations == 1
    assert env.cache_info().hits == 0 and env.cache_info().misses == 2


def test_sampled_opponents_bypass_clear_cache_then_rebuild_on_deterministic_return():
    env, _, bundle = setup()
    env.reset(seed=0); env.step(19)
    bundle.sample_opponents = True
    env.reset(seed=0); env.step(19)
    env.reset(seed=0); env.step(19)
    assert user_calls(bundle) == 36
    assert env.cache_info().currsize == 0 and env.cache_info().bypasses == 2
    assert any(not value for model in bundle.users.values() for value in model.deterministic_arguments)
    bundle.sample_opponents = False
    env.reset(seed=0); env.step(19)
    env.reset(seed=0); env.step(19)
    assert user_calls(bundle) == 48
    assert env.cache_info().hits == 1


def test_new_quotes_and_lru_eviction_execute_real_private_responses():
    env, _, bundle = setup(maxsize=1)
    for menu in (19, 20, 19):
        env.reset(seed=0); env.step(menu)
    assert user_calls(bundle) == 36
    assert env.cache_info().misses == 3 and env.cache_info().currsize == 1


def test_different_scenario_seeds_do_not_share_responses():
    env, _, bundle = setup()
    for seed in (0, 1, 0):
        env.scenario_seeds = (seed,)
        env.reset(seed=0); env.step(19)
    assert user_calls(bundle) == 24
    assert env.cache_info().hits == 1 and env.cache_info().misses == 2


def test_cached_and_uncached_platform_interfaces_are_equal_including_reward_scale():
    env, factory, bundle = setup()
    env.reward_scale_yuan = 2500.
    ordinary = AmendmentPlatformEnv(factory, bundle, "platform", [0], factory(0), reward_scale_yuan=2500.)
    env.reset(seed=0); ordinary.reset(seed=0)
    actual, expected = env.step(19), ordinary.step(19)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]
    env.reset(seed=0)
    hit = env.step(19)
    assert hit[1:] == expected[1:]
    assert_same_outcome(env.episode.outcome(), ordinary.episode.outcome())


def test_changing_reward_scale_cannot_replay_an_old_scaled_reward():
    env, _, bundle = setup()
    env.reset(seed=0); first = env.step(19)
    env.reward_scale_yuan = 2000.
    env.reset(seed=0); second = env.step(19)
    assert second[1] == pytest.approx(first[1] / 2)
    assert env.cache_info().hits == 0 and env.cache_info().misses == 2


def test_actual_ppo_predicts_first_day_and_is_not_called_on_exact_replay(monkeypatch):
    _, factory, _ = setup()
    catalogue = AmendmentTariffCatalogue(factory.config)
    bundle = PolicyBundle(platform_codec=catalogue, training=True)
    real_users, calls = {}, []
    for owner in range(3):
        user_env = ActorEnv(factory, bundle, owner, [0], factory(0))
        model = PPO("MlpPolicy", user_env, seed=owner, n_steps=2, batch_size=2,
                    n_epochs=1, device="cpu", verbose=0)
        predict = model.predict
        def tracked(observation, deterministic=True, original=predict, who=owner):
            calls.append((who, deterministic))
            return original(observation, deterministic=deterministic)
        monkeypatch.setattr(model, "predict", tracked)
        real_users[owner] = model
    bundle.users = real_users
    env = FrozenResponsePlatformEnv(factory, bundle, "platform", [0], factory(0))
    env.reset(seed=0); first = env.step(19)
    assert len(calls) == 12 and all(deterministic for _, deterministic in calls)
    env.reset(seed=0); hit = env.step(19)
    assert len(calls) == 12 and env.cache_info().hits == 1
    assert first[1:] == hit[1:]


def test_platform_learning_does_not_invalidate_user_responses_or_expose_cached_rewards():
    env, _, bundle = setup()
    bundle.platform = CountingPrivateNetwork()
    before, _ = env.reset(seed=0)
    env.step(19)
    with torch.no_grad():
        bundle.platform.policy.weight.add_(10)
    bundle.platform.num_timesteps += 128
    after, _ = env.reset(seed=0)
    np.testing.assert_array_equal(before, after)
    env.step(19)
    assert user_calls(bundle) == 12 and env.cache_info().hits == 1


def test_cache_does_not_apply_to_custom_stochastic_user_action_overrides():
    class CustomResponses(PolicyBundle):
        def user_action(self, episode):
            return super().user_action(episode)
    env, _, bundle = setup()
    env.bundle = CustomResponses(users=bundle.users, platform_codec=bundle.platform_codec)
    for _ in range(2):
        env.reset(seed=0); env.step(19)
    assert user_calls(env.bundle) == 24
    assert env.cache_info().hits == 0 and env.cache_info().bypasses == 2
