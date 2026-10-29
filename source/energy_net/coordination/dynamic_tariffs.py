"""A static, expanded quote library for contextual PPO selection.

Every index always means the same region/time discount matrix. Public grid
observations may affect the policy's choice of index; they never affect this
decoder. The original 78 amendment indices remain unchanged.
"""
from __future__ import annotations

from itertools import product
from types import SimpleNamespace

import numpy as np

from .amendment_features import AmendmentTariffCatalogue
from .gpu_scenario import GpuMarketConfig
from .gpu_tariffs import FixedGpuTariffCatalogue
from .market import encode_platform_action


class ExpandedAmendmentCatalogue:
    """Original quotes, day/night combinations and all-day regional prices.

    Daylight is blocks 1/2; night is blocks 0/3. Each group independently
    selects one target region and a 0.25%, 0.5% or 1% discount. A second
    family independently prices all three regions throughout the day using
    zero, 0.25%, 0.5% or 1%. Duplicate raw quotes are removed at construction,
    preserving the first occurrence and all original indices. The standard
    configuration gives 78 + 72 + 54 = 204 unique actions.

    ``entries[index][1]`` is the largest discount in that matrix, as expected
    by AmendmentTariffObserver. This class applies no curriculum mask; a
    full-action observer call uses ``training=False``.
    """

    daylight_blocks = (1, 2)
    night_blocks = (0, 3)
    compound_amplitudes = (.0025, .005, .01)
    regional_amplitudes = (0., .0025, .005, .01)
    quotas = (1.,)
    original_n_actions = 78

    def __init__(self, config=None):
        self.config = config
        cfg = config or GpuMarketConfig()
        FixedGpuTariffCatalogue._validate(cfg)
        legacy = AmendmentTariffCatalogue(cfg)
        prototype = SimpleNamespace(config=cfg)
        matrices, openings, entries, labels, seen = [], [], [], [], set()

        def append(discounts, opened, entry, label):
            raw = np.r_[encode_platform_action(discounts, np.ones(3), cfg), int(opened)].astype(np.int64)
            key = tuple(int(value) for value in raw)
            if key in seen:
                return False
            seen.add(key)
            # Store the exact values represented by the declared levels,
            # rather than relying on accidental floating-point equality.
            matrix = np.asarray(cfg.discount_levels)[raw[:12]].reshape(3, 4).copy()
            matrix.setflags(write=False)
            matrices.append(matrix); openings.append(int(opened))
            entries.append(entry); labels.append(label)
            return True

        for index in range(legacy.n_actions):
            raw = legacy.decode(prototype, index)
            discounts = np.asarray(cfg.discount_levels)[raw[:12]].reshape(3, 4)
            if not append(discounts, raw[-1], legacy.entries[index], legacy.labels[index]):
                raise ValueError("Original amendment catalogue contains duplicate raw quotes")

        targets = tuple(product(range(3), self.compound_amplitudes))
        for (day_region, day_rate), (night_region, night_rate) in product(targets, repeat=2):
            discounts = np.zeros((3, 4))
            discounts[day_region, self.daylight_blocks] = day_rate
            discounts[night_region, self.night_blocks] = night_rate
            label = (f"daynight_day_r{day_region}_{day_rate:.4f}"
                     f"_night_r{night_region}_{night_rate:.4f}_quota_1.00")
            append(discounts, 1, (label, max(day_rate, night_rate), 1.), label)

        for rates in product(self.regional_amplitudes, repeat=3):
            discounts = np.repeat(np.asarray(rates)[:, None], 4, axis=1)
            label = "regional_dayflat_" + "_".join(
                f"r{region}_{rate:.4f}" for region, rate in enumerate(rates)) + "_quota_1.00"
            append(discounts, 1, (label, max(rates), 1.), label)

        self._discount_matrices = tuple(matrices)
        self._openings = tuple(openings)
        self.entries, self.labels = tuple(entries), tuple(labels)
        self.n_actions = len(entries)

    def decode(self, episode, index):
        """Decode one fixed index using only the declared configuration."""
        value = np.asarray(index)
        if value.size != 1:
            raise ValueError("One scalar expanded amendment menu index is required")
        scalar = value.item()
        if not np.isfinite(scalar) or int(scalar) != scalar or not 0 <= scalar < self.n_actions:
            raise ValueError(f"Expanded amendment menu index must be an integer in [0, {self.n_actions})")
        config = episode.config
        FixedGpuTariffCatalogue._validate(config)
        index = int(scalar)
        raw = encode_platform_action(self._discount_matrices[index], np.ones(3), config)
        return np.r_[raw, self._openings[index]].astype(np.int64)

    def sample(self, episode, rng):
        """Keep the existing catalogue/independent/uniform training mixture."""
        branch = float(rng.random())
        if branch < .60:
            return self.decode(episode, int(rng.integers(self.n_actions)))
        config = episode.config
        FixedGpuTariffCatalogue._validate(config)
        if branch < .80:
            discounts = rng.choice(config.discount_levels, size=(3, 4))
        else:
            discounts = np.full((3, 4), rng.choice(config.discount_levels))
        return np.r_[encode_platform_action(discounts, np.ones(3), config), 1].astype(np.int64)
