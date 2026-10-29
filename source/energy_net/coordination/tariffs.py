"""A small public-information catalogue of daily compute tariff patterns.

The learning platform chooses one catalogue index.  Its meaning is fixed across
days, while the pattern adapts to the original-contract reference electricity
costs.  No candidate is screened using user preferences or realised profit.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from .market import encode_platform_action, zero_platform_action
from .scenario import MarketConfig


def _discount_score(cost):
    """Map lower public energy cost to a larger offered discount."""
    values = np.asarray(cost, dtype=float)
    spread = float(np.ptp(values))
    return np.zeros_like(values) if spread < 1e-12 else (values.max() - values) / spread


class TariffCatalogue:
    """One original tariff plus 54 structured price/quota combinations.

    Indices 1..54 enumerate pattern, amplitude, then quota in that order.
    ``decode(episode, index)`` returns the market's original 15-component action.
    ``sample(episode, rng)`` supplies diverse raw market actions for user training;
    these samples also include unstructured and uniform discount menus.
    """

    patterns = ("time_only", "region_only", "time_region")
    amplitudes = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    quotas = (0.50, 0.75, 1.00)
    n_actions = 55

    def __init__(self, config: MarketConfig | None = None):
        self.config = config or MarketConfig()
        self.entries = (None, *product(self.patterns, self.amplitudes, self.quotas))
        self.labels = ("original", *(
            f"{pattern}_discount_{amplitude:.2f}_quota_{quota:.2f}"
            for pattern, amplitude, quota in self.entries[1:]
        ))

    def pattern_scores(self, episode):
        """Read public reference LMP and facility energy intensity only."""
        cfg = episode.config
        blocks = np.minimum(np.arange(cfg.n_steps) * cfg.n_blocks // cfg.n_steps,
                            cfg.n_blocks - 1)
        unit_cost = (episode.baseline_clearing.lmp_yuan_per_mwh
                     * np.asarray(cfg.energy_mwh_per_cu))
        cost = np.column_stack([
            unit_cost[blocks == block].mean(axis=0) for block in range(cfg.n_blocks)
        ])
        return {
            "time_only": np.broadcast_to(_discount_score(cost.mean(axis=0)), cost.shape),
            "region_only": np.broadcast_to(
                _discount_score(cost.mean(axis=1))[:, None], cost.shape),
            "time_region": _discount_score(cost),
        }

    def decode(self, episode, index):
        """Translate a stable discrete label, without evaluating its outcome."""
        value = np.asarray(index)
        if value.size != 1:
            raise ValueError("A catalogue action is one discrete index.")
        scalar = value.item()
        if not np.isfinite(scalar) or int(scalar) != scalar or not 0 <= scalar < self.n_actions:
            raise ValueError("Catalogue action index must be an integer in [0, 55).")
        index = int(scalar)
        cfg = episode.config
        if index == 0:
            return zero_platform_action(cfg)
        pattern, amplitude, quota = self.entries[index]
        return encode_platform_action(
            amplitude * self.pattern_scores(episode)[pattern],
            np.full(cfg.n_regions, quota), cfg,
        )

    def sample(self, episode, rng):
        """Sample a raw market menu for continual quote coverage.

        60% of draws use the platform catalogue, 20% independently vary the
        twelve discounts, and 20% offer a uniform discount.  Every sample uses
        the existing bounded discounts and quota levels.
        """
        cfg = episode.config
        branch = float(rng.random())
        if branch < 0.60:
            return self.decode(episode, int(rng.integers(self.n_actions)))
        if branch < 0.80:
            discounts = rng.choice(cfg.discount_levels, size=(cfg.n_regions, cfg.n_blocks))
            quotas = rng.choice(cfg.quota_levels, size=cfg.n_regions)
        else:
            discounts = np.full((cfg.n_regions, cfg.n_blocks), rng.choice(cfg.discount_levels))
            quotas = np.ones(cfg.n_regions)
        return encode_platform_action(discounts, quotas, cfg)
