"""Independent PPO actors around a shared, rule-cleared pricing game.

Only the selected actor receives training transitions. Opponent parameters stay
frozen; their actions may be sampled during training. Evaluation uses all four
deterministic learned policies. The market defines profit/utility rewards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .market import MarketEpisode, zero_platform_action
from .scenario import MarketConfig, make_scenario


@dataclass
class PolicyBundle:
    """Live references; the trainer updates just one model at a time."""

    platform: object | None = None
    users: dict[int, object] = field(default_factory=dict)
    platform_codec: object | None = None
    platform_observer: object | None = None
    training: bool = False
    sample_opponents: bool = False
    opponent_sampling_probability: float = 1.0
    price_exploration_probability: float = 0.0
    exploration_rng: object = field(default_factory=lambda: np.random.default_rng(0))

    def _sample_opponent(self):
        return bool(self.training and self.sample_opponents
                    and self.exploration_rng.random() < self.opponent_sampling_probability)

    def decode_platform_action(self, episode, action):
        if self.platform_codec is None:
            return action
        return self.platform_codec.decode(episode, int(np.asarray(action).item()))

    def platform_observation(self, episode):
        if self.platform_observer is not None:
            return self.platform_observer(episode, training=self.training)
        return episode.platform_observation()

    def platform_action(self, episode):
        if (self.training and self.platform_codec is not None
                and self.exploration_rng.random() < self.price_exploration_probability):
            return self.platform_codec.sample(episode, self.exploration_rng)
        if self.platform is None:
            return zero_platform_action(episode.config)
        action = self.platform.predict(self.platform_observation(episode),
                                       deterministic=not self._sample_opponent())[0]
        return self.decode_platform_action(episode, action)

    def user_action(self, episode):
        model = self.users.get(episode.current_user_id)
        if model is None:
            return 0  # Safe original contract during model construction only.
        return int(np.asarray(model.predict(
            episode.current_user_observation(),
            deterministic=not self._sample_opponent())[0]).item())


def cached_episode_factory(config: MarketConfig | None = None, maxsize: int = 64,
                           grid_cache_size: int = 256):
    """Cache a bounded number of immutable scenarios, never live episodes.

    Actors sample the finite training seed pool; evaluation uses disjoint seeds.
    Each episode receives its own reservation state. Exact grid results may be
    reused between days with identical load and cap through a bounded cache.
    """
    config = config or MarketConfig()

    @lru_cache(maxsize=maxsize)
    def scenario(seed):
        return make_scenario(int(seed), config)

    @lru_cache(maxsize=maxsize)
    def grid(seed):
        from .cached_grid import MemoizedGridClearing
        return MemoizedGridClearing(scenario(int(seed)).grid_spec, maxsize=grid_cache_size)

    def factory(seed):
        return MarketEpisode(scenario(int(seed)), grid=grid(int(seed)))

    return factory


class ActorEnv(gym.Env):
    """One platform decision/day or one owner's next-task decisions/day.

    A user transition advances every intervening opponent until the same owner
    acts again or the day ends. Opponents' rewards are deliberately discarded.
    """

    metadata = {"render_modes": []}

    def __init__(self, episode_factory: Callable[[int], MarketEpisode],
                 bundle: PolicyBundle, actor: str | int,
                 scenario_seeds: Sequence[int] = tuple(range(64)),
                 prototype: MarketEpisode | None = None):
        super().__init__()
        self.episode_factory = episode_factory
        self.bundle = bundle
        self.actor = actor
        self.scenario_seeds = tuple(int(x) for x in scenario_seeds)
        if not self.scenario_seeds:
            raise ValueError("Provide a nonempty, finite scenario seed pool.")
        probe = prototype or episode_factory(self.scenario_seeds[0])
        if actor == "platform":
            self.action_space = (spaces.Discrete(bundle.platform_codec.n_actions)
                                 if bundle.platform_codec is not None else
                                 spaces.MultiDiscrete(probe.config.platform_nvec))
            sample = bundle.platform_observation(probe)
        else:
            if not isinstance(actor, int) or not 0 <= actor < probe.config.n_users:
                raise ValueError("actor must be 'platform' or a valid owner id")
            self.action_space = spaces.Discrete(probe.config.n_options)
            sample = probe.current_user_observation()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=sample.shape,
                                            dtype=np.float32)
        self.episode = None
        self.completed_days = 0
        self.actor_steps = 0
        self.last_outcome = None
        self._finished = True

    def _advance_opponents(self):
        while not self.episode.done and self.episode.current_user_id != self.actor:
            self.episode.step_user(self.bundle.user_action(self.episode))

    def _observation(self):
        if self.episode.done:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        if self.actor == "platform":
            return self.bundle.platform_observation(self.episode)
        return self.episode.current_user_observation()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        case_seed = int(self.np_random.choice(self.scenario_seeds))
        self.episode = self.episode_factory(case_seed)
        self.episode.reset()
        self._finished = False
        if self.actor != "platform":
            self.episode.set_platform_action(self.bundle.platform_action(self.episode))
            self._advance_opponents()
            if self.episode.done:
                raise ValueError(f"Scenario {case_seed} has no decisions for owner {self.actor}.")
        return self._observation(), {"scenario_seed": case_seed, "actor": str(self.actor)}

    def step(self, action):
        if self.episode is None or self._finished:
            raise RuntimeError("Call reset before stepping a completed actor episode.")
        if self.actor == "platform":
            self.episode.set_platform_action(self.bundle.decode_platform_action(self.episode, action))
            while not self.episode.done:
                self.episode.step_user(self.bundle.user_action(self.episode))
            reward = self.episode.platform_reward
            info = {}
        else:
            if self.episode.current_user_id != self.actor:
                raise RuntimeError("The pending task belongs to another actor.")
            reward, _, info = self.episode.step_user(action)
            self._advance_opponents()
        self.actor_steps += 1
        terminated = bool(self.episode.done)
        info = dict(info)
        if terminated:
            self.completed_days += 1
            self.last_outcome = self.episode.outcome().to_dict()
            info["market_metrics"] = self.last_outcome
            self._finished = True
        return self._observation(), float(reward), terminated, False, info


class OptimizerStepCounter:
    """Count actual torch optimizer steps separately from SB3 epoch counters."""

    def __init__(self, model):
        self.count = 0
        self._handle = model.policy.optimizer.register_step_post_hook(self._step)

    def _step(self, optimizer, args, kwargs):
        self.count += 1

    def close(self):
        self._handle.remove()


def evaluate_bundle(bundle: PolicyBundle, episode_factory, seeds):
    """Deterministic policies on held-out complete days; no parameter updates."""
    rows, previous_training = [], bundle.training
    bundle.training = False
    try:
        for seed in seeds:
            episode = episode_factory(int(seed))
            episode.reset()
            episode.set_platform_action(bundle.platform_action(episode))
            while not episode.done:
                episode.step_user(bundle.user_action(episode))
            row = episode.outcome().to_dict()
            row["evaluation_policy"] = "all_independent_ppo_deterministic"
            row["platform_action"] = episode.platform_action.tolist()
            rows.append(row)
    finally:
        bundle.training = previous_training
    return rows


def alternating_train(bundle, environments, *, cycles, platform_days, user_steps,
                      seed=0, on_phase=None, initial_models=None, ppo_kwargs=None,
                      before_phase=None):
    """Build four independent PPOs and train with opponents frozen per phase.

    PPO may round a requested sample count up to a complete rollout. Both the
    requested count and actual transitions/optimizer updates are recorded.
    """
    from stable_baselines3 import PPO

    if cycles < 1 or platform_days < 1 or user_steps < 1:
        raise ValueError("cycles, platform_days and user_steps must be positive")
    actors = ["platform"] + sorted(x for x in environments if x != "platform")
    models, counters = {}, {}
    for index, actor in enumerate(actors):
        # Seed the underlying Gym RNG before wrappers are recreated per phase.
        # Pending VecEnv seeds alone are lost by set_env(force_reset=True).
        gym.Env.reset(environments[actor], seed=seed + index)
        requested = platform_days if actor == "platform" else user_steps
        rollout_steps = max(2, min(32, requested))
        if initial_models is not None and actor in initial_models:
            model = initial_models[actor]
            model.set_env(environments[actor], force_reset=True)
        else:
            kwargs = dict(n_steps=rollout_steps, batch_size=min(32, rollout_steps),
                          n_epochs=4, learning_rate=3e-4, gamma=0.99, ent_coef=0.01,
                          policy_kwargs={"net_arch": [64, 64]}, device="cpu", verbose=0)
            kwargs.update(ppo_kwargs or {})
            model = PPO("MlpPolicy", environments[actor], seed=seed + index, **kwargs)
        models[actor] = model
        counters[actor] = OptimizerStepCounter(model)
        if actor == "platform":
            bundle.platform = model
        else:
            bundle.users[actor] = model
    records = []
    try:
        for cycle in range(cycles):
            for actor in actors:
                if before_phase is not None:
                    before_phase(cycle, actor, bundle)
                model, env = models[actor], environments[actor]
                requested = platform_days if actor == "platform" else user_steps
                before_steps, before_updates = model.num_timesteps, counters[actor].count
                before_epochs, before_days = model._n_updates, env.completed_days
                # Discard a partially reset day collected against old opponents.
                model.set_env(env, force_reset=True)
                model.learn(total_timesteps=requested, reset_num_timesteps=False,
                            progress_bar=False)
                record = {
                    "cycle": cycle, "actor": str(actor), "seed": seed,
                    "requested_actor_steps": requested,
                    "actual_actor_steps": model.num_timesteps - before_steps,
                    "cumulative_actor_steps": model.num_timesteps,
                    "optimizer_steps": counters[actor].count - before_updates,
                    "cumulative_optimizer_steps": counters[actor].count,
                    "ppo_training_epochs": model._n_updates - before_epochs,
                    "completed_actor_days": env.completed_days - before_days,
                }
                records.append(record)
                if on_phase is not None:
                    on_phase(record, models)
    finally:
        for counter in counters.values():
            counter.close()
    return models, records
