"""Realized full-grid carbon credit, unchanged cash and private-user stepping."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from energy_net.coordination import posted_offer_learning as learning
from energy_net.coordination.carbon_training import CarbonConstraintConfig, CarbonOfferEnv
from energy_net.coordination.flex_tariffs import FlexTariffObserver
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue
from energy_net.coordination.learning import PolicyBundle
from energy_net.coordination.market import exact_user_policy
from energy_net.coordination.posted_offer_features import PostedOfferCatalogue, PostedOfferObserver
from energy_net.coordination.preference_scenario import encode_preference_episode_id
from energy_net.coordination.varied_scenario import VariedMarketConfig


def test_expected_profit_reference_and_training_units_are_constant():
    default = CarbonConstraintConfig()
    assert (default.q, default.profit_scale, default.emission_scale_t,
            default.lambda_init) == (.95, 1000., 1., 1.)
    assert default.expected_profit_threshold_yuan is None
    assert default.normalized_profit_threshold is None
    config = replace(default, profit_reference_yuan=2000.)
    assert config.expected_profit_threshold_yuan == 1900.
    assert config.normalized_profit_threshold == 1.9
    assert replace(config, q=0.).expected_profit_threshold_yuan == 0.


@pytest.mark.parametrize("changes", [
    {"q": -.1}, {"q": 1.01}, {"q": np.nan},
    {"profit_scale": 0.}, {"profit_scale": np.inf},
    {"emission_scale_t": 0.}, {"emission_scale_t": -1.},
    {"lambda_init": -1.}, {"lambda_init": np.inf},
    {"profit_reference_yuan": -1.}, {"profit_reference_yuan": np.nan},
])
def test_invalid_constraint_units_and_dual_initialization_are_rejected(changes):
    with pytest.raises(ValueError):
        CarbonConstraintConfig(**changes)


class ScriptedEpisode:
    """Adapter fixture with accepted +/- emission changes under a 10t cap.

    Funding and private decisions are scripted here to isolate whether the
    adapter measures actual dispatch, including an unaccepted green offer.
    The integration tests below separately retain the real OPF and ledger.
    """

    changes_t = (2., -1., 4., 7.)
    cash_rewards = (.4, .3, .9, .8)

    def __init__(self):
        self.reset()

    def reset(self):
        self.position = 0
        self.current_clearing = SimpleNamespace(emissions_t=10.)
        self.current_offer = None
        self.public_history = []
        self.cash_yuan = 0.

    @property
    def done(self):
        return self.position == len(self.changes_t)

    def set_offer(self, action):
        assert action in (0, 1)
        self.current_offer = (None if action == 0 else {
            "clearing": SimpleNamespace(emissions_t=(self.current_clearing.emissions_t
                - self.changes_t[self.position])),
        })

    def step_accept(self, action):
        offered = self.current_offer is not None
        accepted = offered and bool(action)
        reward = self.cash_rewards[self.position] if accepted else 0.
        if accepted:
            self.current_clearing = self.current_offer["clearing"]
        public = {"action": int(offered), "accepted": bool(accepted) if offered else None,
                  "response": "accepted" if accepted else "rejected" if offered else "no_offer"}
        self.public_history.append(public)
        self.cash_yuan += reward * 1000.
        self.position += 1
        self.current_offer = None
        return 7e12, self.done, {"platform_step_gain": reward, "public_feedback": public,
            "utility_gain_yuan": 7e15, "loss_yuan": 8e15,
            "reservation_reason": "PRIVATE_SENTINEL"}

    def outcome(self):
        return SimpleNamespace(to_dict=lambda: {
            "platform_gain_vs_outside_yuan": self.cash_yuan,
            "carbon_emissions_t": self.current_clearing.emissions_t,
            "private_user_loss_yuan": 8e15, "private_diagnostics": "PRIVATE_SENTINEL"})


class ScriptedBundle:
    responses = (1, 1, 0, 1)

    def platform_observation(self, episode):
        return np.tile(np.asarray([episode.position, episode.current_clearing.emissions_t],
                                  dtype=np.float32), (61, 1))

    def user_action(self, episode):
        return self.responses[episode.position]


def scripted_env(config=None, base=False):
    constructor = learning.PostedOfferEnv if base else CarbonOfferEnv
    kwargs = {} if base else {"carbon_config": config}
    return constructor(lambda case: ScriptedEpisode(), ScriptedBundle(), "platform", [0], **kwargs)


def test_actual_signed_reductions_telescope_and_rejections_or_exit_credit_zero():
    config = CarbonConstraintConfig(emission_scale_t=2., profit_reference_yuan=10000.)
    env = scripted_env(config)
    env.reset(seed=0)
    cash, carbon, changes, accepted = [], [], [], []
    for action, expected in zip((1, 1, 1, 0), (2., -1., 0., 0.)):
        _, reward, terminated, truncated, info = env.step(action)
        cash.append(reward)
        carbon.append(info["carbon_reduction_reward"])
        changes.append(info["grid_emissions_reduction_t"])
        accepted.append(info["public_feedback"]["accepted"])
        assert not truncated
        assert info["grid_emissions_reduction_t"] == expected
        assert info["carbon_reduction_reward"] == expected / 2.
        assert expected == info["grid_emissions_t_before"] - info["grid_emissions_t_after"]
        assert info["profit_gain_yuan"] == pytest.approx(reward * 1000.)
        assert info["grid_emissions_t_after"] <= 10.
    assert terminated and accepted == [True, True, False, None]
    assert cash == [.4, .3, 0., 0.]
    assert changes == [2., -1., 0., 0.]
    assert sum(carbon) * config.emission_scale_t == pytest.approx(10. - 9.)
    assert info["carbon_episode"] == {"profit_gain_yuan": 700., "grid_emissions_reduction_t": 1.}
    assert sum(cash) * 1000. == pytest.approx(info["carbon_episode"]["profit_gain_yuan"])
    # q applies to an expectation in training, and cannot invalidate one day's
    # already funded contracts or inject a cash top-up to reach the threshold.
    assert info["carbon_episode"]["profit_gain_yuan"] < config.expected_profit_threshold_yuan


def test_dual_or_normalization_cannot_change_gym_cash_observation_or_contracts():
    base = scripted_env(base=True)
    env = scripted_env(CarbonConstraintConfig(lambda_init=100., profit_scale=2000.,
                                             emission_scale_t=.5, profit_reference_yuan=1e9))
    base_obs, base_info = base.reset(seed=0)
    obs, info = env.reset(seed=0)
    np.testing.assert_array_equal(obs, base_obs)
    assert info == base_info
    for action in (1, 1, 1, 0):
        expected_obs, expected_cash, expected_done, expected_truncated, _ = base.step(action)
        obs, cash, done, truncated, info = env.step(action)
        np.testing.assert_array_equal(obs, expected_obs)
        assert (cash, done, truncated) == (expected_cash, expected_done, expected_truncated)
        assert env.episode.public_history == base.episode.public_history
        assert info["platform_step_gain"] == cash  # Already thousands of yuan.
    assert env.last_outcome == base.last_outcome


def test_platform_info_never_contains_private_step_or_terminal_evaluation_accounts():
    env = scripted_env()
    env.reset(seed=0)
    for action in (1, 1, 1, 0):
        observation, _, _, _, info = env.step(action)
        assert "market_metrics" not in info
        assert "PRIVATE_SENTINEL" not in str(info)
        assert not np.any(observation == np.float32(8e15))
        assert not any(fragment in str(info) for fragment in
                       ("loss_yuan", "utility_gain_yuan", "reservation_reason", "private_"))
    assert env.last_outcome["private_diagnostics"] == "PRIVATE_SENTINEL"


def test_reset_clears_day_totals_and_preserves_original_step_lifecycle():
    env = scripted_env()
    with pytest.raises(RuntimeError, match="Reset before stepping"):
        env.step(1)
    env.reset(seed=0)
    for action in (1, 1, 1, 0):
        env.step(action)
    with pytest.raises(RuntimeError, match="Reset before stepping"):
        env.step(1)
    env.reset(seed=0)
    for _ in range(4):
        _, reward, _, _, info = env.step(0)
        assert reward == info["carbon_reduction_reward"] == 0.
    assert info["carbon_episode"] == {"profit_gain_yuan": 0., "grid_emissions_reduction_t": 0.}
    assert env.completed_days == 2


class OldUserFixture:
    def predict(self, observation, deterministic=True):
        return np.asarray(exact_user_policy(observation)), None


class BinaryUserFixture:
    """Private myopic fixture only: no production PPO is replaced here."""
    def predict(self, observation, deterministic=True):
        rows = observation[6:76].reshape(10, 7)
        accept = rows[1, 0] > .5 and rows[1, 1] > .5 and rows[1, 2] + rows[1, 3] < -1e-8
        return np.asarray(int(accept)), None


class OldQuoteFixture:
    def predict(self, observation, deterministic=True):
        return np.asarray(7), None


class PublicFundedFixture:
    def choose(self, episode):
        rows = [episode.preview_offer(action) for action in range(1, 61)]
        rows = [row for row in rows if row["admissible"]]
        return max(rows, key=lambda row: (row["rebate_rate"], row["M"]))["action"] if rows else 0


@pytest.fixture(scope="module")
def real_system():
    config = VariedMarketConfig()
    legacy_config = replace(config, sharing_enabled=False, carbon_eligibility=False,
                            conditional_rebates=False, internal_carbon_value_yuan_per_t=0.)
    old_codec = FixedGpuTariffCatalogue(legacy_config)
    legacy = PolicyBundle(platform=OldQuoteFixture(), users={i: OldUserFixture() for i in range(3)},
        platform_codec=old_codec, platform_observer=FlexTariffObserver(old_codec), training=False)
    legacy.legacy_config = legacy_config
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(learning, "load_legacy_bundle", lambda path: legacy)
        factory = learning.posted_episode_factory("unit-carbon-legacy", config,
            public_cache_size=2, case_cache_size=8, grid_cache_size=128)
    codec = PostedOfferCatalogue(config)
    bundle = learning.PostedPolicyBundle(platform=PublicFundedFixture(),
        users={i: BinaryUserFixture() for i in range(3)}, platform_codec=codec,
        platform_observer=PostedOfferObserver(codec), training=False)
    case = encode_preference_episode_id(0, 11)  # TRAIN physical case, not TEST.
    return factory, bundle, case


def test_real_opf_cash_identity_global_carbon_and_original_floor_are_unchanged(real_system):
    factory, bundle, case = real_system
    env = CarbonOfferEnv(factory, bundle, "platform", [case])
    env.reset(seed=0)
    initial = env.episode.current_clearing.emissions_t
    rewards, emissions = [], []
    while not env.episode.done:
        before = env.episode.current_clearing.emissions_t
        _, reward, _, _, info = env.step(bundle.platform_action(env.episode))
        after = env.episode.current_clearing.emissions_t
        rewards.append(reward)
        emissions.append(info["carbon_reduction_reward"])
        assert info["grid_emissions_reduction_t"] == before - after
        assert after <= env.episode.scenario.carbon_budget_t + 1e-6
        assert "market_metrics" not in info
    metrics = env.last_outcome
    assert len(rewards) == 12 and metrics["accepted_amendments"] > 0
    assert sum(rewards) * 1000. == pytest.approx(metrics["platform_gain_vs_outside_yuan"], abs=1e-6)
    assert sum(rewards) * 1000. == pytest.approx(.8 * metrics["procurement_net_saving_yuan"], abs=1e-6)
    assert sum(emissions) == pytest.approx(initial - metrics["carbon_emissions_t"], abs=1e-9)
    assert info["carbon_episode"]["grid_emissions_reduction_t"] == pytest.approx(
        metrics["grid_emissions_reduction_t"], abs=1e-9)
    assert metrics["platform_floor_satisfied"]
    assert metrics["completion_rate"] == metrics["deadline_compliance"] == metrics["user_ir_rate"] == 1.


@pytest.mark.parametrize("owner", [0, 1, 2])
def test_real_user_actor_still_receives_only_own_four_utility_rewards(real_system, owner):
    factory, bundle, case = real_system
    base = learning.PostedOfferEnv(factory, bundle, owner, [case])
    carbon = CarbonOfferEnv(factory, bundle, owner, [case],
        carbon_config=CarbonConstraintConfig(lambda_init=100., profit_reference_yuan=1e9))
    expected_obs, expected_info = base.reset(seed=0)
    observation, info = carbon.reset(seed=0)
    np.testing.assert_array_equal(observation, expected_obs)
    assert info == expected_info
    rewards = []
    while not carbon.episode.done:
        rows = observation[6:76].reshape(10, 7)
        response = int(rows[1, 0] > .5 and rows[1, 2] + rows[1, 3] < -1e-8)
        expected_obs, expected_reward, expected_done, expected_truncated, expected_info = base.step(response)
        observation, reward, done, truncated, info = carbon.step(response)
        np.testing.assert_array_equal(observation, expected_obs)
        assert (reward, done, truncated) == (expected_reward, expected_done, expected_truncated)
        assert info == expected_info
        assert "carbon_reduction_reward" not in info and "carbon_episode" not in info
        assert reward == pytest.approx(info["utility_gain_yuan"] / 1000.)
        rewards.append(reward)
    assert len(rewards) == carbon.actor_steps == 4
    assert sum(rewards) * 1000. == pytest.approx(
        carbon.last_outcome[f"user_{owner}_gain_yuan"], abs=1e-6)
