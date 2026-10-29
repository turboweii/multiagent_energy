"""Shared learned scores for public candidate tariffs, trained by PPO."""
from __future__ import annotations

from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


def _mlp(size, hidden, output=None):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(size, width), nn.LayerNorm(width), nn.SiLU()))
        size = width
    if output is not None:
        layers.append(nn.Linear(size, output))
    return nn.Sequential(*layers)


class CandidateTariffExtractor(nn.Module):
    def __init__(self, n_actions, n_features, actor_hidden=(64, 32), value_hidden=(64, 32),
                 feature_mean=None, feature_scale=None):
        super().__init__()
        self.n_actions, self.n_features = n_actions, n_features
        mean = np.zeros(n_features - 1) if feature_mean is None else np.asarray(feature_mean)
        scale = np.ones(n_features - 1) if feature_scale is None else np.asarray(feature_scale)
        if (mean.shape != (n_features - 1,) or scale.shape != (n_features - 1,)
                or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                or np.any(scale <= 0)):
            raise ValueError("Feature normalization needs finite per-column means and positive scales.")
        self.register_buffer("feature_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.latent_dim_pi, self.latent_dim_vf = n_actions, value_hidden[-1]
        self.scorer = _mlp(n_features - 1, actor_hidden, 1)
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)
        self.value_encoder = _mlp(2 * (n_features - 1) + 1, value_hidden)

    def _rows(self, features):
        return features.reshape(-1, self.n_actions, self.n_features)

    def _values(self, rows):
        return (rows[..., :-1] - self.feature_mean) / self.feature_scale

    def forward_actor(self, features):
        rows = self._rows(features)
        mask = rows[..., -1] > .5
        fallback = torch.zeros_like(mask)
        fallback[:, 0] = True
        mask = torch.where(mask.any(dim=-1, keepdim=True), mask, fallback)
        scores = self.scorer(self._values(rows)).squeeze(-1)
        return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    def forward_critic(self, features):
        rows = self._rows(features)
        available = (rows[..., -1:] > .5).to(rows.dtype)
        values = self._values(rows)
        count = available.sum(dim=1).clamp_min(1)
        mean = (values * available).sum(dim=1) / count
        peak = values.masked_fill(available < .5, torch.finfo(values.dtype).min).max(dim=1).values
        peak = torch.where(torch.isfinite(peak) & (peak != torch.finfo(values.dtype).min), peak,
                           torch.zeros_like(peak))
        return self.value_encoder(torch.cat([mean, peak, count / self.n_actions], dim=-1))

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class CandidateTariffActorCriticPolicy(ActorCriticPolicy):
    """PPO policy for Box([n_quotes, public_features]) and Discrete(n_quotes).

    Each quote uses identical learned scoring parameters. No profit forecast
    sign, winning quote or preference label is inserted into the logits.
    """

    def __init__(self, *args, actor_hidden=(64, 32), value_hidden=(64, 32),
                 feature_mean=None, feature_scale=None, **kwargs):
        self.actor_hidden, self.value_hidden = tuple(actor_hidden), tuple(value_hidden)
        self.feature_mean = None if feature_mean is None else tuple(feature_mean)
        self.feature_scale = None if feature_scale is None else tuple(feature_scale)
        kwargs.setdefault("ortho_init", False)
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("Candidate tariffs need a discrete action space.")

    def _build_mlp_extractor(self):
        shape = self.observation_space.shape
        if len(shape) != 2 or shape[0] != self.action_space.n:
            raise ValueError("Expected one feature row for each candidate tariff.")
        self.mlp_extractor = CandidateTariffExtractor(
            shape[0], shape[1], self.actor_hidden, self.value_hidden,
            self.feature_mean, self.feature_scale)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        values = super()._get_constructor_parameters()
        values.update(actor_hidden=self.actor_hidden, value_hidden=self.value_hidden)
        values.update(feature_mean=self.feature_mean, feature_scale=self.feature_scale)
        return values
