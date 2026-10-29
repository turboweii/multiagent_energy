"""Exact day memoization while platform PPO faces fixed deterministic users.

Only a day previously executed through the real user policies and OPF can be
reused. Cached returns never enter platform observations. User updates clear
the cache; sampled opponents always run the ordinary environment instead.
"""
from __future__ import annotations

from collections import OrderedDict, namedtuple
from copy import deepcopy
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from .amendment_market import AmendmentPlatformEnv
from .learning import PolicyBundle


CacheInfo = namedtuple("CacheInfo", "hits misses maxsize currsize invalidations bypasses")


def _model_fingerprint(model):
    if model is None:
        return None
    policy = getattr(model, "policy", None)
    if isinstance(policy, torch.nn.Module):
        digest = hashlib.blake2b(digest_size=20)
        for category, values in (("parameter", policy.named_parameters()),
                                 ("buffer", policy.named_buffers())):
            for name, value in values:
                digest.update(repr((category, name, tuple(value.shape), str(value.dtype))).encode())
                data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
                digest.update(data)
        contents = digest.digest()
    elif callable(getattr(model, "frozen_response_fingerprint", None)):
        # Small non-neural constant policies may expose their behaviour version.
        contents = deepcopy(model.frozen_response_fingerprint())
    else:
        raise TypeError("Cached users need a torch policy or frozen_response_fingerprint()")
    return (id(model), id(policy), getattr(model, "num_timesteps", None),
            getattr(model, "_n_updates", None), contents)


@dataclass(frozen=True)
class _CompletedDay:
    # Strong references prevent an evicted source's object ID being reused.
    source: object
    config: object
    state: dict
    terminal_observation: np.ndarray
    reward: float
    info: dict


class FrozenResponsePlatformEnv(AmendmentPlatformEnv):
    """The ordinary platform interface, with exact fixed-user rollout reuse.

    The episode factory must return deterministic, immutable cases for a seed.
    A new quote or a changed user is executed normally. The cache stores only
    outcomes actually sampled by platform PPO; it does not evaluate additional
    quotes, solve user best responses, or change rewards. Joint/user training
    continues to use the original environments.
    """

    _terminal_fields = (
        "position", "selected", "requested", "reservations", "bills", "losses",
        "current_clearing", "amendment_ledger", "platform_action", "discounts",
        "quota_fractions", "tariff_yuan_per_cu", "amendment_open",
    )

    def __init__(self, *args, maxsize=16384, **kwargs):
        super().__init__(*args, **kwargs)
        if int(maxsize) != maxsize or maxsize < 1:
            raise ValueError("The fixed-response cache must hold at least one completed day")
        self.maxsize = int(maxsize)
        self._completed = OrderedDict()
        self._users_fingerprint = None
        self._cache_hits = self._cache_misses = 0
        self._cache_invalidations = self._cache_bypasses = 0

    def cache_info(self):
        return CacheInfo(self._cache_hits, self._cache_misses, self.maxsize,
                         len(self._completed), self._cache_invalidations, self._cache_bypasses)

    def cache_clear(self):
        self._completed.clear()
        self._users_fingerprint = None

    def _fingerprint(self):
        return tuple((owner, _model_fingerprint(model))
                     for owner, model in sorted(self.bundle.users.items()))

    def _cache_allowed(self):
        # The standard bundle guarantees deterministic predict() here. A custom
        # response override can have its own RNG even when the flag is false.
        return (not self.bundle.sample_opponents
                and getattr(self.bundle.user_action, "__func__", None) is PolicyBundle.user_action
                and getattr(self.bundle._sample_opponent, "__func__", None) is PolicyBundle._sample_opponent)

    def _key(self, raw):
        episode = self.episode
        source = getattr(episode, "original_scenario", episode.scenario)
        reference = (
            tuple(getattr(episode, "outside_selected_options", ())),
            tuple(float(value) for value in episode.baseline_bills),
            tuple(float(value) for value in getattr(episode, "outside_losses", ())),
            tuple(getattr(episode, "outside_platform_action", ())),
            episode.scenario.carbon_budget_t,
        )
        return (int(episode.scenario.seed), id(source), id(episode.config), reference,
                float(self.reward_scale_yuan),
                tuple(np.asarray(raw, dtype=np.int64).reshape(-1)))

    def step(self, action):
        if self.episode is None or self._finished:
            raise RuntimeError("Call reset before stepping a completed actor episode.")
        if not self._cache_allowed():
            if self._completed:
                self._cache_invalidations += 1
            self.cache_clear()
            self._cache_bypasses += 1
            return super().step(action)
        fingerprint = self._fingerprint()
        if self._users_fingerprint is not None and fingerprint != self._users_fingerprint:
            self._completed.clear()
            self._cache_invalidations += 1
        self._users_fingerprint = fingerprint
        raw = self.bundle.decode_platform_action(self.episode, action)
        key = self._key(raw)
        entry = self._completed.get(key)
        if entry is not None:
            self._completed.move_to_end(key)
            self._cache_hits += 1
            # Restore only mutable completion state. The current episode keeps
            # its own original inputs, outside accounts, OPF solver and caches.
            for name, value in deepcopy(entry.state).items():
                setattr(self.episode, name, value)
            self.actor_steps += 1
            self.completed_days += 1
            self._finished = True
            info = deepcopy(entry.info)
            self.last_outcome = info["market_metrics"]
            return entry.terminal_observation.copy(), entry.reward, True, False, info
        self._cache_misses += 1
        result = super().step(action)
        observation, reward, terminated, truncated, info = result
        # A response model that updates during predict() cannot make a reusable
        # fixed-user result, even if it did not advertise sampled opponents.
        if self._fingerprint() != fingerprint:
            self._completed.clear()
            self._users_fingerprint = None
            self._cache_invalidations += 1
            return result
        if terminated and not truncated:
            state = deepcopy({name: getattr(self.episode, name) for name in self._terminal_fields
                              if hasattr(self.episode, name)})
            source = getattr(self.episode, "original_scenario", self.episode.scenario)
            self._completed[key] = _CompletedDay(source, self.episode.config, state,
                                                observation.copy(), float(reward), deepcopy(info))
            if len(self._completed) > self.maxsize:
                self._completed.popitem(last=False)
        return result
