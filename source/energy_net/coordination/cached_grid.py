"""Exact, bounded memoization of fixed-network OPF clearing results."""
from __future__ import annotations

from collections import OrderedDict, namedtuple
from dataclasses import replace

import numpy as np

from .grid import ClearingResult, GridClearing, GridSpec


CacheInfo = namedtuple("CacheInfo", "hits misses maxsize currsize")


def _copy_result(result: ClearingResult) -> ClearingResult:
    """Keep the cache's dispatch and price arrays private from its callers."""
    return replace(result,
                   lmp_yuan_per_mwh=result.lmp_yuan_per_mwh.copy(),
                   generation_mw=result.generation_mw.copy(),
                   flows_mw=result.flows_mw.copy())


class MemoizedGridClearing(GridClearing):
    """Reuse identical solves for one fixed ``GridSpec`` instance.

    Keys contain every float64 load bit, the exact float64 carbon cap (or None),
    and the objective. Inputs are never rounded. Both feasible and infeasible
    results occupy the LRU, and every return copies the mutable result arrays.
    A zero capacity keeps statistics while disabling storage.
    """

    def __init__(self, spec: GridSpec, maxsize: int = 256):
        if int(maxsize) != maxsize or maxsize < 0:
            raise ValueError("maxsize must be a nonnegative integer")
        super().__init__(spec)
        self.maxsize = int(maxsize)
        self.hits = 0
        self.misses = 0
        self._cache = OrderedDict()

    def solve(self, load_mw: np.ndarray, carbon_budget_t: float | None = None,
              objective: str = "cost") -> ClearingResult:
        if objective not in ("cost", "emissions"):
            raise ValueError("objective must be 'cost' or 'emissions'")
        load = self._load(np.asarray(load_mw, dtype=np.float64))
        if carbon_budget_t is not None and not np.isfinite(carbon_budget_t):
            raise ValueError("carbon_budget_t must be finite or None")
        cap_key = (None if carbon_budget_t is None else
                   np.float64(carbon_budget_t).tobytes())
        key = (load.shape, load.tobytes(order="C"), cap_key, objective)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return _copy_result(self._cache[key])
        self.misses += 1
        result = super().solve(load, carbon_budget_t, objective)
        if self.maxsize:
            self._cache[key] = result
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        return _copy_result(result)

    def cache_info(self):
        return CacheInfo(self.hits, self.misses, self.maxsize, len(self._cache))

    def cache_clear(self):
        self._cache.clear()
        self.hits = self.misses = 0
