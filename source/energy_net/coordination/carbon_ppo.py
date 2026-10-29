"""PPO with separate cash/carbon critics and a complete-day profit constraint.

Gym rewards remain actual incremental cash divided by 1000. The additional
public info reward is actual dispatched emissions avoided, in the environment's
declared carbon units. No private inconvenience enters this algorithm. The
dual multiplier changes the learning objective, never the settlement ledger.
"""
from __future__ import annotations

from copy import deepcopy
from typing import NamedTuple

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import explained_variance, obs_as_tensor

from .platform_policy import CandidateTariffActorCriticPolicy


class CarbonRolloutSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    old_carbon_values: th.Tensor
    carbon_advantages: th.Tensor
    carbon_returns: th.Tensor


class CarbonRolloutBuffer(RolloutBuffer):
    """The standard profit buffer plus an independently bootstrapped carbon GAE."""

    def reset(self):
        super().reset()
        for name in ("carbon_rewards", "carbon_values", "carbon_returns", "carbon_advantages"):
            setattr(self, name, np.zeros((self.buffer_size, self.n_envs), dtype=np.float32))

    def add(self, *args, carbon_reward, carbon_value, **kwargs):
        self.carbon_rewards[self.pos] = np.asarray(carbon_reward, dtype=np.float32)
        self.carbon_values[self.pos] = carbon_value.detach().cpu().numpy().flatten()
        super().add(*args, **kwargs)

    def compute_returns_and_advantage(self, last_values, dones, *, last_carbon_values):
        super().compute_returns_and_advantage(last_values, dones)
        last = last_carbon_values.detach().cpu().numpy().flatten()
        gae = np.zeros(self.n_envs, dtype=np.float32)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                nonterminal = 1. - dones.astype(np.float32)
                next_value = last
            else:
                nonterminal = 1. - self.episode_starts[step + 1]
                next_value = self.carbon_values[step + 1]
            delta = (self.carbon_rewards[step] + self.gamma * next_value * nonterminal
                     - self.carbon_values[step])
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            self.carbon_advantages[step] = gae
        self.carbon_returns = self.carbon_advantages + self.carbon_values

    def get(self, batch_size=None):
        if not self.generator_ready:
            for name in ("carbon_values", "carbon_advantages", "carbon_returns"):
                setattr(self, name, self.swap_and_flatten(getattr(self, name)))
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env=None):
        profit = super()._get_samples(batch_inds, env)
        extra = [self.to_torch(getattr(self, name)[batch_inds].flatten())
                 for name in ("carbon_values", "carbon_advantages", "carbon_returns")]
        return CarbonRolloutSamples(*profit, *extra)


class CarbonCandidateTariffPolicy(CandidateTariffActorCriticPolicy):
    """Unchanged candidate actor/profit critic, independent zero-output carbon critic."""

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.carbon_value_encoder = deepcopy(self.mlp_extractor.value_encoder)
        self.carbon_value_net = deepcopy(self.value_net)
        nn.init.zeros_(self.carbon_value_net.weight)
        nn.init.zeros_(self.carbon_value_net.bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def predict_carbon_values(self, obs):
        features = BasePolicy.extract_features(self, obs, self.vf_features_extractor)
        extractor = self.mlp_extractor
        rows = extractor._rows(features)
        available = (rows[..., -1:] > .5).to(rows.dtype)
        values = extractor._values(rows)
        count = available.sum(dim=1).clamp_min(1)
        mean = (values * available).sum(dim=1) / count
        peak = values.masked_fill(available < .5, th.finfo(values.dtype).min).max(dim=1).values
        peak = th.where(th.isfinite(peak) & (peak != th.finfo(values.dtype).min), peak,
                        th.zeros_like(peak))
        pooled = th.cat([mean, peak, count / extractor.n_actions], dim=-1)
        return self.carbon_value_net(self.carbon_value_encoder(pooled))


# Full name is also convenient next to the existing policy's public class name.
CarbonCandidateTariffActorCriticPolicy = CarbonCandidateTariffPolicy


def mixed_advantages(carbon, profit, dual_value, *, normalize=True, carbon_objective=True):
    """Combine in declared units, then normalize once, not separately per objective."""
    result = ((carbon + dual_value * profit) / (1. + dual_value)
              if carbon_objective else profit)
    if normalize and len(result) > 1:
        result = (result - result.mean()) / (result.std() + 1e-8)
    return result


class CarbonConstrainedPPO(PPO):
    """Maximize avoided emissions subject to expected daily incremental cash.

    ``profit_floor_yuan_per_day`` is a declared fraction of a TRAIN-only
    reference's *incremental* cash, not a fraction of its absolute profit.
    Complete, non-truncated days update lambda by
    ``dual_lr * (floor/1000 - mean_daily_cash_reward)``. Days spanning rollout
    boundaries are counted once on completion; resetting an env discards its
    unfinished day. This expected-profit objective is distinct from the market's
    per-contract cash floor, which the environment continues to enforce.
    """

    policy_aliases = {
        "CarbonCandidateTariffPolicy": CarbonCandidateTariffPolicy,
        "CarbonCandidateTariffActorCriticPolicy": CarbonCandidateTariffPolicy,
    }

    def __init__(self, policy=CarbonCandidateTariffPolicy, env=None, *,
                 profit_floor_yuan_per_day=0., dual_lr=.1, dual_initial=1., dual_max=100.,
                 carbon_reward_key="carbon_reduction_reward", carbon_reward_scale_t=1.,
                 carbon_vf_coef=.5, carbon_objective=True, **kwargs):
        for name, value in (("profit_floor_yuan_per_day", profit_floor_yuan_per_day),
                            ("dual_lr", dual_lr), ("dual_initial", dual_initial),
                            ("dual_max", dual_max), ("carbon_vf_coef", carbon_vf_coef)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if dual_initial > dual_max or not np.isfinite(carbon_reward_scale_t) or carbon_reward_scale_t <= 0:
            raise ValueError("Need dual_initial <= dual_max and positive carbon_reward_scale_t")
        self.profit_floor_yuan_per_day = float(profit_floor_yuan_per_day)
        self.dual_lr, self.dual_max = float(dual_lr), float(dual_max)
        self.dual_value = float(dual_initial)
        self.carbon_reward_key = str(carbon_reward_key)
        self.carbon_reward_scale_t = float(carbon_reward_scale_t)
        self.carbon_vf_coef = float(carbon_vf_coef)
        self.carbon_objective = bool(carbon_objective)
        self.dual_history = []
        self.warm_start_metadata = {}
        self._episode_profit_running = None
        self._episode_carbon_running = None
        kwargs.setdefault("gamma", 1.)
        kwargs.setdefault("gae_lambda", .95)
        kwargs.setdefault("target_kl", .02)
        if kwargs["gamma"] != 1.:
            raise ValueError("Daily cash and telescoping emissions require gamma=1")
        if kwargs.get("use_sde", False):
            raise ValueError("Candidate tariffs use Discrete actions, not SDE")
        if policy is CandidateTariffActorCriticPolicy:
            policy = CarbonCandidateTariffPolicy
        kwargs.setdefault("rollout_buffer_class", CarbonRolloutBuffer)
        super().__init__(policy, env, **kwargs)

    @property
    def profit_floor_scaled(self):
        return self.profit_floor_yuan_per_day / 1000.

    def _setup_model(self):
        super()._setup_model()
        if not isinstance(self.policy, CarbonCandidateTariffPolicy):
            raise TypeError("CarbonConstrainedPPO needs CarbonCandidateTariffPolicy")
        if not isinstance(self.rollout_buffer, CarbonRolloutBuffer):
            raise TypeError("CarbonConstrainedPPO needs CarbonRolloutBuffer")
        self._clear_incomplete_days()

    def _clear_incomplete_days(self):
        n = getattr(self, "n_envs", 1)
        self._episode_profit_running = np.zeros(n, dtype=np.float64)
        self._episode_carbon_running = np.zeros(n, dtype=np.float64)

    def set_env(self, env, force_reset=True):
        super().set_env(env, force_reset=force_reset)
        if force_reset:
            self._clear_incomplete_days()

    def _setup_learn(self, *args, **kwargs):
        # A requested reset, including SB3's default learn(reset_num_timesteps=True),
        # must not join cash from an old partial episode onto a newly reset day.
        result = super()._setup_learn(*args, **kwargs)
        if np.asarray(self._last_episode_starts).all():
            self._clear_incomplete_days()
        return result

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_episode_profit_running", "_episode_carbon_running"]

    @classmethod
    def from_profit_ppo(cls, original, env, *, profit_floor_yuan_per_day, **overrides):
        """Copy all original actor/profit parameters exactly; start a fresh optimizer/budget."""
        if not isinstance(original.policy, CandidateTariffActorCriticPolicy):
            raise TypeError("Warm source must use CandidateTariffActorCriticPolicy")
        settings = {name: deepcopy(getattr(original, name)) for name in
            ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda",
             "ent_coef", "vf_coef", "max_grad_norm", "normalize_advantage", "clip_range", "clip_range_vf")}
        settings.update(policy_kwargs=deepcopy(original.policy_kwargs), device=original.device,
                        seed=original.seed, verbose=original.verbose)
        settings.update(overrides)
        result = cls(CarbonCandidateTariffPolicy, env,
                     profit_floor_yuan_per_day=profit_floor_yuan_per_day, **settings)
        source = {key: value for key, value in original.policy.state_dict().items()
                  if not key.startswith("carbon_value_")}
        missing, unexpected = result.policy.load_state_dict(source, strict=False)
        if unexpected or any(not key.startswith("carbon_value_") for key in missing):
            raise ValueError(f"Incompatible warm policy: missing={missing}, unexpected={unexpected}")
        result.policy.carbon_value_encoder.load_state_dict(original.policy.mlp_extractor.value_encoder.state_dict())
        result.warm_start_metadata = dict(source_algorithm=type(original).__name__,
            source_num_timesteps=int(original.num_timesteps), source_seed=original.seed,
            actor_and_profit_critic="exact parameter copy", carbon_head="independent encoder; zero output",
            optimizer="fresh", new_training_start_step=0)
        result.num_timesteps = 0
        # Construction of the extra critic must not determine opponent/case RNGs.
        result.set_random_seed(result.seed)
        return result

    def _update_dual(self, completed_profit, completed_carbon):
        before = self.dual_value
        average = float(np.mean(completed_profit)) if completed_profit else None
        carbon = float(np.mean(completed_carbon)) if completed_carbon else None
        gap = self.profit_floor_scaled - average if average is not None else None
        if self.carbon_objective and gap is not None:
            self.dual_value = float(np.clip(before + self.dual_lr * gap, 0., self.dual_max))
        record = dict(timesteps=int(self.num_timesteps), dual_before=before, dual_after=self.dual_value,
                      actor_dual_value=self.dual_value, complete_days=len(completed_profit),
                      mean_profit_scaled=average, mean_profit_yuan=average * 1000 if average is not None else None,
                      mean_carbon=carbon,
                      mean_grid_emissions_reduction_t=carbon * self.carbon_reward_scale_t if carbon is not None else None,
                      profit_gap_scaled=gap, profit_floor_yuan_per_day=self.profit_floor_yuan_per_day,
                      kl=None, optimizer_steps=0, carbon_objective=self.carbon_objective)
        self.dual_history.append(record)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        # This is the SB3 on-policy collector for discrete tariffs, augmented by
        # a second value/bootstrap path and *unbootstrapped* complete-day accounts.
        assert self._last_obs is not None
        normalizer = self.get_vec_normalize_env()
        if normalizer is not None and normalizer.norm_reward:
            raise ValueError("Profit constraint requires actual cash/1000; reward normalization must be off")
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        completed_profit, completed_carbon = [], []
        callback.on_rollout_start()
        for n_steps in range(n_rollout_steps):
            with th.no_grad():
                obs = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs)
                carbon_values = self.policy.predict_carbon_values(obs)
            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            rewards = np.asarray(rewards, dtype=np.float32).copy()
            carbon_rewards = np.array([info.get(self.carbon_reward_key, 0.) for info in infos], dtype=np.float32)
            if self.carbon_objective and any(self.carbon_reward_key not in info for info in infos):
                raise KeyError(f"Every actual platform step needs info[{self.carbon_reward_key!r}]")
            if not np.isfinite(rewards).all() or not np.isfinite(carbon_rewards).all():
                raise ValueError("Cash and carbon rewards must be finite")
            # Before timeout value bootstrap: these sums are physical cash/carbon.
            self._episode_profit_running += rewards
            self._episode_carbon_running += carbon_rewards
            self.num_timesteps += env.num_envs
            for index, done in enumerate(dones):
                if not done:
                    continue
                timeout = infos[index].get("TimeLimit.truncated", False)
                if not timeout:
                    episode = infos[index].get("carbon_episode", {})
                    cash = episode.get("profit_gain_yuan")
                    carbon = episode.get("grid_emissions_reduction_t")
                    completed_profit.append(float(cash) / 1000. if cash is not None
                                            else float(self._episode_profit_running[index]))
                    completed_carbon.append(float(carbon) / self.carbon_reward_scale_t if carbon is not None
                                            else float(self._episode_carbon_running[index]))
                self._episode_profit_running[index] = 0.
                self._episode_carbon_running[index] = 0.
            callback.update_locals(locals())
            if not callback.on_step():
                self._last_obs, self._last_episode_starts = new_obs, dones
                return False
            self._update_info_buffer(infos, dones)
            for index, done in enumerate(dones):
                if (done and infos[index].get("terminal_observation") is not None
                        and infos[index].get("TimeLimit.truncated", False)):
                    terminal = self.policy.obs_to_tensor(infos[index]["terminal_observation"])[0]
                    with th.no_grad():
                        rewards[index] += self.gamma * float(self.policy.predict_values(terminal).flatten()[0])
                        carbon_rewards[index] += self.gamma * float(self.policy.predict_carbon_values(terminal).flatten()[0])
            rollout_buffer.add(self._last_obs, actions.reshape(-1, 1), rewards,
                               self._last_episode_starts, values, log_probs,
                               carbon_reward=carbon_rewards, carbon_value=carbon_values)
            self._last_obs, self._last_episode_starts = new_obs, dones
        with th.no_grad():
            next_obs = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(next_obs)
            carbon_values = self.policy.predict_carbon_values(next_obs)
        rollout_buffer.compute_returns_and_advantage(values, dones, last_carbon_values=carbon_values)
        self._update_dual(completed_profit, completed_carbon)
        n_steps = n_rollout_steps  # Same completed-count callback semantics as SB3.
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None
        entropy_losses, policy_losses, profit_losses, carbon_losses = [], [], [], []
        clip_fractions, all_kls, optimizer_steps = [], [], 0
        continue_training = True
        for epoch in range(self.n_epochs):
            for data in self.rollout_buffer.get(self.batch_size):
                actions = data.actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(data.observations, actions)
                carbon_values = self.policy.predict_carbon_values(data.observations).flatten()
                values = values.flatten()
                advantages = mixed_advantages(data.carbon_advantages, data.advantages, self.dual_value,
                    normalize=self.normalize_advantage, carbon_objective=self.carbon_objective)
                ratio = th.exp(log_prob - data.old_log_prob)
                policy_loss = -th.min(advantages * ratio,
                    advantages * th.clamp(ratio, 1. - clip_range, 1. + clip_range)).mean()
                clip_fractions.append(float((th.abs(ratio - 1) > clip_range).float().mean()))
                if clip_vf is None:
                    predicted, carbon_predicted = values, carbon_values
                else:
                    predicted = data.old_values + th.clamp(values - data.old_values, -clip_vf, clip_vf)
                    carbon_predicted = data.old_carbon_values + th.clamp(carbon_values - data.old_carbon_values, -clip_vf, clip_vf)
                profit_loss = F.mse_loss(data.returns, predicted)
                carbon_loss = F.mse_loss(data.carbon_returns, carbon_predicted)
                entropy_loss = th.mean(log_prob) if entropy is None else -th.mean(entropy)
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * profit_loss
                if self.carbon_objective:
                    loss = loss + self.carbon_vf_coef * carbon_loss
                with th.no_grad():
                    log_ratio = log_prob - data.old_log_prob
                    kl = float(((th.exp(log_ratio) - 1) - log_ratio).mean())
                all_kls.append(kl)
                entropy_losses.append(float(entropy_loss.detach()))
                policy_losses.append(float(policy_loss.detach()))
                profit_losses.append(float(profit_loss.detach()))
                carbon_losses.append(float(carbon_loss.detach()))
                if self.target_kl is not None and kl > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                optimizer_steps += 1
            self._n_updates += 1
            if not continue_training:
                break
        records = {
            "entropy_loss": np.mean(entropy_losses), "policy_gradient_loss": np.mean(policy_losses),
            "value_loss": np.mean(profit_losses), "carbon_value_loss": np.mean(carbon_losses),
            "approx_kl": np.mean(all_kls), "max_approx_kl": max(all_kls),
            "clip_fraction": np.mean(clip_fractions), "loss": float(loss.detach()),
            "explained_variance": explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()),
            "carbon_explained_variance": explained_variance(self.rollout_buffer.carbon_values.flatten(), self.rollout_buffer.carbon_returns.flatten()),
            "clip_range": clip_range, "dual_value": self.dual_value,
            "optimizer_steps": optimizer_steps,
        }
        for key, value in records.items():
            self.logger.record(f"train/{key}", value)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if clip_vf is not None:
            self.logger.record("train/clip_range_vf", clip_vf)
        if self.dual_history:
            self.dual_history[-1].update(kl=float(np.mean(all_kls)), max_kl=float(max(all_kls)),
                optimizer_steps=optimizer_steps, epochs_completed=epoch + 1,
                early_stopped=not continue_training)
