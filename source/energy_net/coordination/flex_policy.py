"""A learned menu intercept plus a contextual shared PPO tariff scorer.

Every intercept starts at zero. They learn only from sampled policy-gradient
returns, with no benchmark rankings or outcome labels inserted into logits.
The stable 77-menu catalogue gives each intercept a persistent contract meaning.
"""
from __future__ import annotations

from copy import deepcopy
import random

import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO

from .platform_policy import CandidateTariffActorCriticPolicy, CandidateTariffExtractor


class FlexTariffExtractor(CandidateTariffExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_bias = nn.Parameter(torch.zeros(self.n_actions))

    def forward_actor(self, features):
        rows = self._rows(features)
        mask = rows[..., -1] > .5
        fallback = torch.zeros_like(mask)
        fallback[:, 0] = True
        mask = torch.where(mask.any(dim=-1, keepdim=True), mask, fallback)
        scores = self.scorer(self._values(rows)).squeeze(-1) + self.action_bias
        return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)


class FlexTariffActorCriticPolicy(CandidateTariffActorCriticPolicy):
    """Logits are ``learned_menu_bias[a] + learned_context_score(state,a)``."""

    def _build_mlp_extractor(self):
        shape = self.observation_space.shape
        if len(shape) != 2 or shape[0] != self.action_space.n:
            raise ValueError("Expected one public feature row for each fixed tariff menu")
        self.mlp_extractor = FlexTariffExtractor(
            shape[0], shape[1], self.actor_hidden, self.value_hidden,
            self.feature_mean, self.feature_scale,
        )


def _inherit_weights_and_optimizer(target, source):
    """Name-based copying avoids shifting Adam states when a parameter is added."""
    source_state = source.policy.state_dict()
    target_state = target.policy.state_dict()
    new_name = "mlp_extractor.action_bias"
    if set(target_state) - set(source_state) not in (set(), {new_name}):
        raise ValueError("Only a new action-bias parameter may differ during warm start")
    if set(source_state) - set(target_state):
        raise ValueError("Source policy has parameters absent from the contextual-intercept policy")
    for name, value in source_state.items():
        if value.shape != target_state[name].shape:
            raise ValueError(f"Policy shape differs for {name}")
    missing, unexpected = target.policy.load_state_dict(source_state, strict=False)
    if set(missing) - {new_name} or unexpected:
        raise ValueError("The contextual policy could not be copied exactly")
    if new_name not in source_state:
        with torch.no_grad():
            target.policy.mlp_extractor.action_bias.zero_()
    original_optimizer, new_optimizer = source.policy.optimizer, target.policy.optimizer
    if len(original_optimizer.param_groups) != len(new_optimizer.param_groups):
        raise ValueError("Warm start requires matching optimizer parameter groups")
    # Hyperparameters of each group remain those used by the source optimizer.
    for old_group, new_group in zip(original_optimizer.param_groups, new_optimizer.param_groups):
        for key, value in old_group.items():
            if key != "params":
                new_group[key] = deepcopy(value)
    original_parameters = dict(source.policy.named_parameters())
    target_parameters = dict(target.policy.named_parameters())
    new_optimizer.state.clear()
    for name, parameter in original_parameters.items():
        if parameter in original_optimizer.state:
            new_optimizer.state[target_parameters[name]] = deepcopy(original_optimizer.state[parameter])
    # A newly added bias intentionally has no Adam moments or step history.


def warm_start_with_action_bias(source_PPO, env):
    """Return a PPO continuing the same policy with a zero-initial menu bias.

    Contextual network weights, critic weights, normalisation buffers, Adam
    history and training counters are preserved. The new environment starts a
    complete day; partially collected transitions are not carried across.
    Construction preserves the caller's Python, NumPy and torch random state.
    """
    if not isinstance(source_PPO.policy, CandidateTariffActorCriticPolicy):
        raise TypeError("Warm start needs a public candidate-tariff PPO policy")
    if env.observation_space.shape != source_PPO.observation_space.shape:
        raise ValueError("Warm start requires the same candidate observation shape")
    if env.action_space.n != source_PPO.action_space.n:
        raise ValueError("Warm start requires the same fixed tariff action space")
    kwargs = dict(source_PPO.policy_kwargs)
    kwargs.update(actor_hidden=source_PPO.policy.actor_hidden,
                  value_hidden=source_PPO.policy.value_hidden,
                  feature_mean=source_PPO.policy.feature_mean,
                  feature_scale=source_PPO.policy.feature_scale)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        target = PPO(
            FlexTariffActorCriticPolicy, env,
            learning_rate=source_PPO.learning_rate,
            n_steps=source_PPO.n_steps, batch_size=source_PPO.batch_size,
            n_epochs=source_PPO.n_epochs, gamma=source_PPO.gamma,
            gae_lambda=source_PPO.gae_lambda, clip_range=source_PPO.clip_range,
            clip_range_vf=source_PPO.clip_range_vf,
            normalize_advantage=source_PPO.normalize_advantage,
            ent_coef=source_PPO.ent_coef, vf_coef=source_PPO.vf_coef,
            max_grad_norm=source_PPO.max_grad_norm, use_sde=source_PPO.use_sde,
            sde_sample_freq=source_PPO.sde_sample_freq, target_kl=source_PPO.target_kl,
            tensorboard_log=source_PPO.tensorboard_log,
            policy_kwargs=kwargs, verbose=source_PPO.verbose,
            seed=source_PPO.seed, device=source_PPO.device,
        )
        _inherit_weights_and_optimizer(target, source_PPO)
        for name in ("num_timesteps", "_n_updates", "_episode_num", "_total_timesteps",
                     "_num_timesteps_at_start", "_current_progress_remaining"):
            if hasattr(source_PPO, name):
                setattr(target, name, deepcopy(getattr(source_PPO, name)))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return target
