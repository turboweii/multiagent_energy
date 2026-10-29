"""Sequential ownership, cash units and private/public factory integration."""
from dataclasses import replace

import numpy as np
import pytest

from energy_net.coordination import posted_offer_learning as learning
from energy_net.coordination.flex_market import FlexMarketEpisode
from energy_net.coordination.flex_tariffs import FlexTariffObserver
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue
from energy_net.coordination.learning import PolicyBundle
from energy_net.coordination.market import exact_user_policy
from energy_net.coordination.posted_offer_features import PostedOfferCatalogue, PostedOfferObserver
from energy_net.coordination.preference_scenario import (
    clone_with_preferences, decode_preference_episode_id, encode_preference_episode_id,
)
from energy_net.coordination.varied_scenario import VariedMarketConfig


class OldPrivateCostFixture:
    def __init__(self, owner):
        self.owner, self.calls = owner, 0

    def predict(self, observation, deterministic=True):
        assert deterministic and observation[0] == self.owner / 2
        self.calls += 1
        return np.asarray(exact_user_policy(observation)), None


class BinaryPrivateCostFixture:
    """Myopic response fixture only; production users retain their own PPOs."""
    def __init__(self, owner):
        self.owner, self.calls = owner, 0

    def predict(self, observation, deterministic=True):
        assert deterministic and observation[0] == self.owner / 2
        self.calls += 1
        options = observation[6:76].reshape(10, 7)
        accept = options[1, 0] > .5 and options[1, 1] > .5 and options[1, 2] + options[1, 3] < -1e-8
        return np.asarray(int(accept)), None


class PublicQuoteFixture:
    def __init__(self, menu=7):
        self.menu = menu
        self.observations = []

    def predict(self, observation, deterministic=True):
        assert deterministic
        self.observations.append(observation.copy())
        return np.asarray(self.menu), None


class FundedPublicController:
    """One public offer, without consulting a user's inconvenience or policy."""
    def choose(self, episode):
        candidates = [episode.preview_offer(action) for action in range(1, 61)]
        candidates = [row for row in candidates if row["admissible"]]
        return max(candidates, key=lambda row: (row["rebate_rate"], row["M"]))["action"] if candidates else 0


@pytest.fixture
def system(monkeypatch):
    config = VariedMarketConfig()
    legacy_config = replace(config, sharing_enabled=False, carbon_eligibility=False,
                            conditional_rebates=False, internal_carbon_value_yuan_per_t=0.)
    codec = FixedGpuTariffCatalogue(legacy_config)
    quote = PublicQuoteFixture()
    legacy = PolicyBundle(platform=quote, users={u: OldPrivateCostFixture(u) for u in range(3)},
        platform_codec=codec, platform_observer=FlexTariffObserver(codec), training=False)
    legacy.legacy_config = legacy_config
    monkeypatch.setattr(learning, "load_legacy_bundle", lambda path: legacy)
    factory = learning.posted_episode_factory("unit-legacy", config,
        public_cache_size=2, case_cache_size=8, grid_cache_size=128)
    cases = tuple(encode_preference_episode_id(0, seed) for seed in (11, 12))
    private = {u: BinaryPrivateCostFixture(u) for u in range(3)}
    posted_codec = PostedOfferCatalogue(config)
    bundle = learning.PostedPolicyBundle(platform=FundedPublicController(), users=private,
        platform_codec=posted_codec, platform_observer=PostedOfferObserver(posted_codec), training=False)
    return factory, bundle, cases, legacy, quote


def test_factory_draws_preferences_before_real_legacy_booking_and_reuses_only_public_quote(system):
    factory, _, cases, legacy, quote = system
    first, second = factory(cases[0]), factory(cases[1])
    assert len(quote.observations) == 1
    assert sum(user.calls for user in legacy.users.values()) == 24
    assert first.scenario.grid_spec is second.scenario.grid_spec
    assert first.scenario.carbon_budget_t == second.scenario.carbon_budget_t
    assert first.scenario.background_mw is second.scenario.background_mw
    for episode, case in zip((first, second), cases):
        assert episode.scenario.seed == episode.booking.scenario.seed == case
        for job, selected, loss in zip(factory.scenario(case).jobs,
                                      episode.outside_selected_options, episode.outside_losses):
            assert loss == pytest.approx(job.options[selected].user_loss_yuan)
            expected = job.work_cu * (job.beta_yuan_per_cu_hour * job.options[selected].delay_hours
                + job.migration_loss_yuan_per_cu * job.options[selected].migrated)
            assert loss == pytest.approx(expected)
        for owner in range(3):
            pairs = {(job.beta_yuan_per_cu_hour, job.migration_loss_yuan_per_cu)
                     for job in episode.jobs if job.user_id == owner}
            assert len(pairs) == 1
    assert first.jobs[0].beta_yuan_per_cu_hour != second.jobs[0].beta_yuan_per_cu_hour
    # Compare the actual old public observer on the independent private draws.
    # It sees identical public physics and no private losses before old choices.
    for case in cases:
        probe = FlexMarketEpisode(replace(factory.scenario(case), config=legacy.legacy_config),
                                  grid=factory.grid(0))
        legacy.platform_action(probe)
    np.testing.assert_array_equal(quote.observations[0], quote.observations[1])
    np.testing.assert_array_equal(quote.observations[0], quote.observations[2])
    assert first.public_history == second.public_history == []


def test_platform_takes_twelve_cash_steps_with_only_actual_public_responses(system):
    factory, bundle, cases, _, _ = system
    env = learning.PostedOfferEnv(factory, bundle, "platform", [cases[0]], factory(cases[0]))
    observation, info = env.reset(seed=0)
    assert observation.shape == (61, len(bundle.platform_observer.feature_names))
    assert info["case_id"] == cases[0]  # Logging address, not a policy feature.
    rewards = []
    while not env.episode.done:
        position = env.episode.position
        # Public counterfactual physical/funding previews create no response.
        bundle.platform_observation(env.episode)
        assert len(env.feedback_records) == position
        action = bundle.platform_action(env.episode)
        observation, reward, terminated, truncated, step_info = env.step(action)
        rewards.append(reward)
        assert env.episode.position == position + 1
        assert len(env.feedback_records) == position + 1
        assert step_info["public_feedback"] == env.episode.public_history[-1]
        assert step_info["public_feedback"]["action"] == action
        assert not truncated
        if not terminated:
            assert set(step_info) == {"public_feedback", "platform_step_gain"}
        assert not any(name in step_info for name in ("loss_yuan", "utility_gain_yuan", "reservation_reason"))
    assert len(rewards) == env.actor_steps == 12
    assert sum(user.calls for user in bundle.users.values()) == 12
    assert all(user.calls == 4 for user in bundle.users.values())
    assert env.completed_days == 1 and env._finished
    assert np.all(observation == 0)
    assert env.feedback_records == env.episode.public_history
    assert sum(rewards) * 1000 == pytest.approx(env.last_outcome["platform_gain_vs_outside_yuan"], abs=1e-6)
    assert sum(rewards) * 1000 == pytest.approx(.8 * env.last_outcome["procurement_net_saving_yuan"], abs=1e-6)
    assert env.last_outcome["platform_floor_satisfied"]


@pytest.mark.parametrize("owner", [0, 1, 2])
def test_user_steps_skip_other_owners_and_return_only_own_total_yuan_utility(system, owner):
    factory, bundle, cases, _, _ = system
    env = learning.PostedOfferEnv(factory, bundle, owner, [cases[0]], factory(cases[0]))
    observation, _ = env.reset(seed=0)
    rewards, positions, nonzero = [], [], []
    while not env.episode.done:
        assert env.episode.current_user_id == owner
        positions.append(env.episode.position)
        job = env.episode.jobs[env.episode.position]
        rows = observation[6:76].reshape(10, 7)
        response = int(rows[1, 0] > .5 and rows[1, 2] + rows[1, 3] < -1e-8)
        observation, reward, _, truncated, info = env.step(response)
        assert not truncated and info["user_id"] == owner
        assert reward == pytest.approx(info["utility_gain_yuan"] / 1000)
        if abs(info["utility_gain_yuan"]) > 1e-6:
            nonzero.append(reward)
            assert reward != pytest.approx(info["utility_gain_yuan"] / job.work_cu)
        rewards.append(reward)
    assert len(rewards) == env.actor_steps == 4
    assert positions == sorted(positions) and len(set(positions)) == 4
    assert sum(user.calls for user in bundle.users.values()) == 8  # Only other owners use opponents.
    assert len(env.feedback_records) == len(env.episode.public_history) == 12
    assert sum(rewards) * 1000 == pytest.approx(env.last_outcome[f"user_{owner}_gain_yuan"], abs=1e-6)
    if owner == 0:
        assert nonzero, "Fixture must exercise actual yuan-vs-per-GPUh scaling"


def test_seeds_are_cache_addresses_and_do_not_enter_either_actor_state(system):
    factory, bundle, cases, _, _ = system
    episode = factory(cases[0])
    public = bundle.platform_observation(episode)
    private = episode.current_user_observation()
    episode.scenario = replace(episode.scenario, seed=-(10**16))
    episode.original_scenario = replace(episode.original_scenario, seed=-(10**16))
    np.testing.assert_array_equal(public, bundle.platform_observation(episode))
    np.testing.assert_array_equal(private, episode.current_user_observation())
    assert not any("seed" in name or "case_id" in name for name in bundle.platform_observer.feature_names)
    cases2 = learning.preference_cases([0, 1], 2, 100)
    assert len(cases2) == len(set(cases2)) == 4
    assert [decode_preference_episode_id(case) for case in cases2] == [(0, 100), (0, 101), (1, 102), (1, 103)]


def test_private_step_fields_cannot_change_platform_observation_or_cash(system, monkeypatch):
    factory, bundle, cases, _, _ = system
    env = learning.PostedOfferEnv(factory, bundle, "platform", [cases[0]], factory(cases[0]))
    env.reset(seed=0)
    original = env.episode.step_accept
    def private_diagnostics(action):
        reward, done, info = original(action)
        info["loss_yuan"] = 9e12
        info["utility_gain_yuan"] = 8e12
        info["reservation_reason"] = "PRIVATE_SENTINEL"
        return reward, done, info
    monkeypatch.setattr(env.episode, "step_accept", private_diagnostics)
    observation, reward, _, _, info = env.step(0)
    assert reward == 0
    assert set(info) == {"public_feedback", "platform_step_gain"}
    assert "PRIVATE_SENTINEL" not in str(info)
    assert not np.any(observation == np.float32(9e12))
    assert not np.any(observation == np.float32(8e12))
    assert env.episode.public_history[0]["response"] == "no_offer"
    assert env.episode.public_history[0]["accepted"] is None


def test_posted_policy_bundle_samples_only_funded_actions_without_a_platform(system):
    factory, bundle, cases, _, _ = system
    bundle.platform = None
    for seed in (0, 1, 2):
        bundle.exploration_rng = np.random.default_rng(seed)
        episode = factory(cases[0])
        action = bundle.platform_action(episode)
        assert episode.preview_offer(action)["admissible"]
        assert episode.public_history == []


def test_trainer_wires_gamma_one_before_any_platform_learning(system, tmp_path, monkeypatch):
    from scripts import train_posted_offers as trainer
    factory, bundle, cases, _, _ = system
    monkeypatch.setattr(trainer, "TRAIN", (cases[0],))
    monkeypatch.setattr(trainer, "posted_episode_factory", lambda *args, **kwargs: factory)
    monkeypatch.setattr(trainer, "load_users", lambda path: bundle.users)
    class BeforeLearning(Exception):
        pass
    def constructor(policy, env, **kwargs):
        assert kwargs["gamma"] == 1.
        assert kwargs["gae_lambda"] == .95
        assert env.actor == "platform" and env.action_space.n == 61
        raise BeforeLearning
    monkeypatch.setattr(trainer, "PPO", constructor)
    with pytest.raises(BeforeLearning):
        trainer.train_stage(tmp_path / "platform", "unit-legacy", tmp_path, 2, 0, 0, True)


def test_trainer_wires_gamma_one_and_unscaled_own_episode_return_for_users(system, tmp_path, monkeypatch):
    from scripts import train_posted_offers as trainer
    factory, _, cases, _, _ = system
    monkeypatch.setattr(trainer, "TRAIN", (cases[0],))
    monkeypatch.setattr(trainer, "posted_episode_factory", lambda *args, **kwargs: factory)
    class BeforeLearning(Exception):
        pass
    class LoadOnly:
        @staticmethod
        def load(*args, **kwargs):
            return object()
    def warm(source, env, **kwargs):
        assert kwargs["gamma"] == kwargs["gae_lambda"] == 1.
        assert env.action_space.n == 2 and env.actor == 0
        raise BeforeLearning
    monkeypatch.setattr(trainer, "PPO", LoadOnly)
    monkeypatch.setattr(trainer, "warm_load_binary_user", warm)
    with pytest.raises(BeforeLearning):
        trainer.users_stage(tmp_path / "users", "unit-legacy", tmp_path, 2, 0)
