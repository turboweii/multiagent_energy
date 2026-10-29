"""Platform PPO observations and profit feedback with fixed learning users."""
from __future__ import annotations

from collections import OrderedDict
import numpy as np

from .learning import ActorEnv
from .platform_features import candidate_tariff_observation, feature_names
from .tariffs import TariffCatalogue


def apply_private_participation(model):
    """Retain the owner's learned ranking over economically acceptable offers.

    This is a private reservation-utility constraint, not a platform filter or
    an optimal-choice label. The user's original contract remains available.
    """
    if not hasattr(model.policy.mlp_extractor, "participation_constraint"):
        raise ValueError("Use a contract-scoring user policy for private participation.")
    model.policy.participation_constraint = True
    model.policy.mlp_extractor.participation_constraint = True
    model.policy_kwargs = dict(model.policy_kwargs, participation_constraint=True)
    return model


class CandidateTariffObserver:
    """Cache immutable public features; expose training-only quote curriculum."""

    def __init__(self, catalogue=None, maxsize=512):
        self.catalogue = catalogue or TariffCatalogue()
        self.maxsize = maxsize
        self.training_max_discount = .30
        self._cache = OrderedDict()

    def raw(self, episode):
        key = int(episode.scenario.seed)
        entry = self._cache.get(key)
        if entry is None or entry[0] is not episode.scenario:
            entry = (episode.scenario, candidate_tariff_observation(episode, self.catalogue))
            self._cache[key] = entry
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return entry[1].copy()

    def __call__(self, episode, *, training=False):
        values = self.raw(episode)
        if training:
            for index, entry in enumerate(self.catalogue.entries):
                if entry is not None and entry[1] > self.training_max_discount + 1e-9:
                    values[index, -1] = 0
        return values

    def fit_normalization(self, episode_factory, seeds):
        # Only public inputs from training scenarios, never returns or labels.
        rows = np.concatenate([self.raw(episode_factory(int(seed)))[:, :-1]
                               for seed in seeds], axis=0)
        mean = rows.mean(axis=0, dtype=np.float64)
        scale = np.maximum(rows.std(axis=0, dtype=np.float64), .01)
        return mean.astype(np.float32), scale.astype(np.float32)


class PlatformProfitEnv(ActorEnv):
    """One sampled price choice/day, with profit in fixed thousand-yuan units."""

    def __init__(self, *args, profit_scale_yuan=1000.0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.actor != "platform" or profit_scale_yuan <= 0:
            raise ValueError("PlatformProfitEnv needs the platform actor and a positive scale.")
        self.profit_scale_yuan = float(profit_scale_yuan)

    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)
        reward = info["market_metrics"]["delta_platform_profit_yuan"] / self.profit_scale_yuan
        return observation, float(reward), terminated, truncated, info


def platform_policy_statistics(bundle, episode_factory, seeds):
    """Describe conditional action probabilities without reward look-ahead."""
    import torch

    rows, probabilities = [], []
    previous = bundle.training
    bundle.training = False
    try:
        for seed in seeds:
            episode = episode_factory(int(seed))
            obs = bundle.platform_observation(episode)
            tensor, _ = bundle.platform.policy.obs_to_tensor(obs)
            with torch.no_grad():
                probs = bundle.platform.policy.get_distribution(tensor).distribution.probs.cpu().numpy()[0]
            action = int(np.argmax(probs))
            probabilities.append(probs)
            rows.append(dict(seed=int(seed), action=action,
                             label=bundle.platform_codec.labels[action],
                             probability=float(probs[action]),
                             entropy=float(-np.sum(probs * np.log(np.maximum(probs, 1e-30))))))
    finally:
        bundle.training = previous
    array = np.asarray(probabilities)
    stats = {
        "distinct_actions": len({row["action"] for row in rows}),
        "original_contract_days": sum(row["action"] == 0 for row in rows),
        "mean_entropy": float(np.mean([row["entropy"] for row in rows])),
        "mean_probability_std_across_states": float(array.std(axis=0).mean()),
        "maximum_probability_std_across_states": float(array.std(axis=0).max()),
    }
    return stats, rows
