"""Numerical/serialization tests on tiny synthetic tasks, without research cases."""
from copy import deepcopy

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from energy_net.coordination.carbon_ppo import (
    CarbonCandidateTariffPolicy, CarbonConstrainedPPO, CarbonRolloutBuffer,
    mixed_advantages,
)
from energy_net.coordination.platform_policy import CandidateTariffActorCriticPolicy


@pytest.fixture(autouse=True)
def single_thread():
    old = th.get_num_threads()
    th.set_num_threads(1)
    yield
    th.set_num_threads(old)


class TinyOffers(gym.Env):
    """Three usable candidate actions, real per-step cash and observed carbon."""
    observation_space = spaces.Box(-100., 100., (61, 7), dtype=np.float32)
    action_space = spaces.Discrete(61)

    def __init__(self, length=3, *, fixed=True, include_carbon=True):
        self.length, self.fixed, self.include_carbon = length, fixed, include_carbon
        self.position = 0

    def obs(self):
        rows = np.zeros((61, 7), dtype=np.float32)
        rows[:3, 0] = np.arange(3)
        rows[:3, 1] = self.position / self.length
        rows[:3, 2] = [.1, .2, .4]
        rows[:3, -1] = 1.
        return rows

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        return self.obs(), {}

    def step(self, action):
        assert 0 <= int(action) < 3
        self.position += 1
        cash = .5 if self.fixed else (.05, .5, .9)[int(action)]
        carbon = .2 if self.fixed else (.01, .25, -.05)[int(action)]
        info = {"carbon_reduction_reward": carbon} if self.include_carbon else {}
        return self.obs(), cash, self.position == self.length, False, info


def model(env=None, **kwargs):
    options = dict(n_steps=2, batch_size=2, n_epochs=1, learning_rate=0.,
        policy_kwargs=dict(actor_hidden=(8,), value_hidden=(8,)), seed=0,
        profit_floor_yuan_per_day=2000., device="cpu")
    options.update(kwargs)
    return CarbonConstrainedPPO(CarbonCandidateTariffPolicy, env or TinyOffers(), **options)


def test_two_gae_paths_match_hand_computation_without_cross_episode_bootstrap():
    env = TinyOffers()
    buffer = CarbonRolloutBuffer(3, env.observation_space, env.action_space,
                                gamma=1., gae_lambda=.95)
    for index, (cash, carbon) in enumerate(zip([1., 2., 3.], [.1, .2, .3])):
        buffer.add(env.obs()[None], np.array([[0]]), np.array([cash]),
                   np.array([index == 0]), th.tensor([.5]), th.tensor([0.]),
                   carbon_reward=np.array([carbon]), carbon_value=th.tensor([.05]))
    buffer.compute_returns_and_advantage(th.tensor([999.]), np.array([True]),
                                         last_carbon_values=th.tensor([888.]))
    # Terminal delta=reward-value; earlier deltas include next value.
    profit_adv = [1 + .95*(2 + .95*2.5), 2 + .95*2.5, 2.5]
    carbon_adv = [.1 + .95*(.2 + .95*.25), .2 + .95*.25, .25]
    np.testing.assert_allclose(buffer.advantages[:, 0], profit_adv, rtol=1e-6)
    np.testing.assert_allclose(buffer.carbon_advantages[:, 0], carbon_adv, rtol=1e-6)
    np.testing.assert_allclose(buffer.returns[:, 0], np.array(profit_adv)+.5, rtol=1e-6)
    np.testing.assert_allclose(buffer.carbon_returns[:, 0], np.array(carbon_adv)+.05, rtol=1e-6)
    batches = list(buffer.get(3))
    assert batches[0].carbon_returns.shape == batches[0].returns.shape == (3,)
    # A second epoch must not flatten the extra arrays a second time.
    assert len(list(buffer.get(1))) == 3


def test_gae_bootstraps_both_heads_at_rollout_boundary():
    env = TinyOffers()
    buffer = CarbonRolloutBuffer(1, env.observation_space, env.action_space, gamma=1., gae_lambda=.95)
    buffer.add(env.obs()[None], np.array([[0]]), np.array([1.]), np.array([True]),
               th.tensor([.4]), th.tensor([0.]), carbon_reward=np.array([.2]),
               carbon_value=th.tensor([.1]))
    buffer.compute_returns_and_advantage(th.tensor([3.]), np.array([False]),
                                         last_carbon_values=th.tensor([2.]))
    np.testing.assert_allclose(buffer.returns, [[4.]])
    np.testing.assert_allclose(buffer.carbon_returns, [[2.2]])


def test_mix_is_normalized_once_after_combining_declared_units():
    carbon, cash = th.tensor([2., 0., -1.]), th.tensor([1., 4., 3.])
    expected = (carbon + 2*cash)/3
    th.testing.assert_close(mixed_advantages(carbon, cash, 2., normalize=False), expected)
    th.testing.assert_close(mixed_advantages(carbon, cash, 2.),
                            (expected-expected.mean())/expected.std())
    th.testing.assert_close(mixed_advantages(carbon, cash, 200., normalize=False,
                                            carbon_objective=False), cash)


def test_whole_days_cross_rollouts_and_dual_uses_day_not_step_cash():
    algo = model()
    algo.learn(6, reset_num_timesteps=False)
    assert [row['complete_days'] for row in algo.dual_history] == [0, 1, 1]
    assert algo.dual_history[0]['mean_profit_scaled'] is None
    assert algo.dual_history[1]['mean_profit_scaled'] == pytest.approx(1.5)
    assert algo.dual_history[1]['mean_carbon'] == pytest.approx(.6)
    assert algo.dual_value == pytest.approx(1.1)
    # Neither floor nor dual coefficient is injected as a per-step reward.
    np.testing.assert_allclose(algo.rollout_buffer.rewards, .5)
    np.testing.assert_allclose(algo.rollout_buffer.carbon_rewards, .2)


def test_vector_days_are_weighted_per_completed_episode():
    env = DummyVecEnv([lambda: TinyOffers(2), lambda: TinyOffers(4)])
    algo = model(env, n_steps=4, batch_size=4, profit_floor_yuan_per_day=1500.)
    algo.learn(8, reset_num_timesteps=False)
    row = algo.dual_history[0]
    assert row['complete_days'] == 3
    assert row['mean_profit_scaled'] == pytest.approx(4/3)
    assert algo.dual_value == pytest.approx(1 + .1*(1.5 - 4/3))


def test_dual_decreases_and_obeys_declared_bounds():
    algo = model(dual_initial=.01, dual_max=.1, dual_lr=.1, profit_floor_yuan_per_day=0.)
    algo._update_dual([1.], [.2])
    assert algo.dual_value == 0.
    algo.profit_floor_yuan_per_day = 10000.
    algo._update_dual([0.], [.2])
    assert algo.dual_value == .1


@pytest.mark.parametrize('timeout', [False, True])
def test_timeouts_bootstrap_both_heads_but_are_not_complete_days(timeout):
    env = gym.wrappers.TimeLimit(TinyOffers(12), max_episode_steps=2) if timeout else TinyOffers(2)
    algo = model(env)
    with th.no_grad():
        algo.policy.value_net.weight.zero_(); algo.policy.value_net.bias.fill_(.7)
        algo.policy.carbon_value_net.weight.zero_(); algo.policy.carbon_value_net.bias.fill_(.9)
    algo.learn(2, reset_num_timesteps=False)
    assert algo.rollout_buffer.returns[-1].item() == pytest.approx(1.2 if timeout else .5)
    assert algo.rollout_buffer.carbon_returns[-1].item() == pytest.approx(1.1 if timeout else .2)
    assert algo.dual_history[0]['complete_days'] == (0 if timeout else 1)
    if timeout:
        assert algo.dual_value == 1.


def test_warm_actor_and_profit_values_are_exact_and_source_is_unchanged():
    env = TinyOffers()
    original = PPO(CandidateTariffActorCriticPolicy, env, n_steps=4, batch_size=4,
        n_epochs=1, gamma=1., seed=7, device='cpu',
        policy_kwargs=dict(actor_hidden=(8,), value_hidden=(8,)))
    original.learn(8)
    before = deepcopy(original.policy.state_dict())
    opt_before = deepcopy(original.policy.optimizer.state_dict())
    warm = CarbonConstrainedPPO.from_profit_ppo(original, TinyOffers(),
        profit_floor_yuan_per_day=1000.)
    for key, value in before.items():
        th.testing.assert_close(warm.policy.state_dict()[key], value, rtol=0, atol=0)
        th.testing.assert_close(original.policy.state_dict()[key], value, rtol=0, atol=0)
    obs = th.tensor(env.obs()[None])
    with th.no_grad():
        th.testing.assert_close(warm.policy.get_distribution(obs).distribution.logits,
                                original.policy.get_distribution(obs).distribution.logits, rtol=0, atol=0)
        th.testing.assert_close(warm.policy.predict_values(obs), original.policy.predict_values(obs), rtol=0, atol=0)
        assert warm.policy.predict_carbon_values(obs).eq(0).all()
    assert warm.num_timesteps == 0
    assert warm.warm_start_metadata['source_num_timesteps'] == original.num_timesteps == 8
    assert warm.policy.optimizer.state_dict()['state'] == {}
    assert set(original.policy.optimizer.state_dict()['state']) == set(opt_before['state'])
    for profit, carbon in zip(warm.policy.mlp_extractor.value_encoder.parameters(),
                              warm.policy.carbon_value_encoder.parameters()):
        assert profit.data_ptr() != carbon.data_ptr()


def test_actor_and_both_critics_update_and_all_parameters_are_in_optimizer():
    algo = model(TinyOffers(12, fixed=False), n_steps=12, batch_size=12,
                 n_epochs=3, learning_rate=1e-3, target_kl=None)
    before = deepcopy(algo.policy.state_dict())
    optimizer_ids = {id(p) for group in algo.policy.optimizer.param_groups for p in group['params']}
    assert optimizer_ids == {id(p) for p in algo.policy.parameters()}
    algo.learn(48, reset_num_timesteps=False)
    after = algo.policy.state_dict()
    for prefix in ('mlp_extractor.scorer', 'value_net', 'carbon_value_net', 'carbon_value_encoder'):
        assert any(not th.equal(value, after[key]) for key, value in before.items() if key.startswith(prefix)), prefix
    assert algo.dual_history[-1]['optimizer_steps'] > 0
    assert np.isfinite(algo.dual_history[-1]['kl'])
    assert np.isfinite(algo.logger.name_to_value['train/carbon_value_loss'])
    assert algo.predict(TinyOffers().obs(), deterministic=True)[0] < 3


def test_force_reset_drops_partial_day_without_false_dual_update():
    algo = model()
    algo.learn(2, reset_num_timesteps=False)
    assert algo._episode_profit_running[0] == 1.
    algo.set_env(TinyOffers(), force_reset=True)
    algo.learn(2, reset_num_timesteps=False)
    assert algo.dual_history[-1]['complete_days'] == 0
    assert algo.dual_value == 1.


def test_save_load_restores_parameters_dual_and_history_but_not_partial_day(tmp_path):
    algo = model()
    algo.learn(4, reset_num_timesteps=False)
    assert algo._episode_profit_running[0] == .5
    algo.save(tmp_path/'carbon')
    loaded = CarbonConstrainedPPO.load(tmp_path/'carbon.zip', env=TinyOffers(), device='cpu')
    assert loaded.dual_value == algo.dual_value
    assert loaded.dual_history == algo.dual_history
    assert loaded.profit_floor_yuan_per_day == 2000.
    assert isinstance(loaded.rollout_buffer, CarbonRolloutBuffer)
    assert loaded._episode_profit_running[0] == 0.
    for key, value in algo.policy.state_dict().items():
        th.testing.assert_close(loaded.policy.state_dict()[key], value, rtol=0, atol=0)
    np.testing.assert_equal(loaded.predict(TinyOffers().obs(), deterministic=True)[0],
                            algo.predict(TinyOffers().obs(), deterministic=True)[0])
    source_state = algo.policy.optimizer.state_dict()['state']
    loaded_state = loaded.policy.optimizer.state_dict()['state']
    assert source_state.keys() == loaded_state.keys()
    for index in source_state:
        for key, value in source_state[index].items():
            th.testing.assert_close(loaded_state[index][key], value, rtol=0, atol=0)
    loaded.learn(2, reset_num_timesteps=False)
    assert loaded.dual_history[-1]['complete_days'] == 0


def test_cash_only_mode_preserves_cash_objective_and_does_not_fit_carbon_head():
    algo = model(TinyOffers(include_carbon=False), carbon_objective=False, learning_rate=1e-3)
    before = deepcopy(algo.policy.carbon_value_net.state_dict())
    algo.learn(6, reset_num_timesteps=False)
    assert algo.dual_value == 1.
    for key, value in before.items():
        th.testing.assert_close(value, algo.policy.carbon_value_net.state_dict()[key], rtol=0, atol=0)


def test_carbon_reward_is_required_and_cash_normalization_is_rejected():
    with pytest.raises(KeyError, match='carbon_reduction_reward'):
        model(TinyOffers(include_carbon=False)).learn(2)
    env = VecNormalize(DummyVecEnv([lambda: TinyOffers()]), norm_obs=False, norm_reward=True)
    with pytest.raises(ValueError, match='normalization'):
        model(env).learn(2)


def test_discounted_profit_or_nonfinite_dual_configuration_is_rejected():
    with pytest.raises(ValueError, match='gamma=1'):
        model(gamma=.99)
    with pytest.raises(ValueError, match='dual_lr'):
        model(dual_lr=np.nan)


def test_kl_guard_stops_before_optimizer_step():
    algo = model(learning_rate=1e-3)
    algo.learn(2, reset_num_timesteps=False)
    # A deliberately stale log probability tests the train-time guard without
    # relying on any particular learning dynamics to create a large update.
    algo.rollout_buffer.log_probs -= 20.
    before = deepcopy(algo.policy.state_dict())
    algo.train()
    assert algo.dual_history[-1]['early_stopped']
    assert algo.dual_history[-1]['optimizer_steps'] == 0
    assert algo.dual_history[-1]['max_kl'] > 1.5 * algo.target_kl
    for key, value in before.items():
        th.testing.assert_close(value, algo.policy.state_dict()[key], rtol=0, atol=0)


def test_terminal_physical_accounts_have_unscaled_cash_and_scaled_carbon():
    class PhysicalTotals(TinyOffers):
        def step(self, action):
            obs, cash, done, truncated, info = super().step(action)
            info['carbon_reduction_reward'] /= 2.
            if done:
                info['carbon_episode'] = dict(profit_gain_yuan=1500.000001,
                                               grid_emissions_reduction_t=.600000002)
            return obs, cash, done, truncated, info
    algo = model(PhysicalTotals(), n_steps=3, batch_size=3, carbon_reward_scale_t=2.)
    algo.learn(3, reset_num_timesteps=False)
    row = algo.dual_history[0]
    assert row['mean_profit_yuan'] == pytest.approx(1500.000001, abs=1e-10)
    assert row['mean_carbon'] == pytest.approx(.300000001, abs=1e-10)
    assert row['mean_grid_emissions_reduction_t'] == pytest.approx(.600000002, abs=1e-10)
    np.testing.assert_allclose(algo.rollout_buffer.rewards, .5)
    np.testing.assert_allclose(algo.rollout_buffer.carbon_rewards, .1)
