"""Memoization preserves exact OPF physics, prices and mutable result isolation."""
from dataclasses import fields

import numpy as np
import pytest

from energy_net.coordination.cached_grid import MemoizedGridClearing
from energy_net.coordination.grid import ClearingResult, GridClearing, GridSpec, default_grid


def one_bus_grid():
    return GridSpec(
        n_steps=1, dt_hours=.5, bus_names=("single",),
        generator_bus=np.array([0, 0]),
        generator_capacity_mw=np.full((1, 2), 100.),
        generator_cost_yuan_per_mwh=np.array([100., 300.]),
        generator_emissions_t_per_mwh=np.array([1., .1]),
        line_from=np.array([], dtype=int), line_to=np.array([], dtype=int),
        line_reactance=np.array([]), line_capacity_mw=np.array([]),
    )


def assert_same_result(first, second):
    for field in fields(ClearingResult):
        np.testing.assert_equal(getattr(first, field.name), getattr(second, field.name))


@pytest.mark.parametrize("objective", ["cost", "emissions"])
@pytest.mark.parametrize("budget", [None, 12., 0.])
def test_cached_and_uncached_results_match_all_fields(objective, budget):
    spec = default_grid(np.array([15., 20.]), n_steps=2)
    load = np.tile([5., 10., 8.], (2, 1))
    plain, memo = GridClearing(spec), MemoizedGridClearing(spec)
    expected = plain.solve(load, budget, objective)
    assert_same_result(memo.solve(load, budget, objective), expected)
    assert_same_result(memo.solve(load.copy(), budget, objective), expected)
    assert memo.cache_info() == (1, 1, 256, 1)


def test_exact_float64_load_bits_define_cache_key():
    memo = MemoizedGridClearing(one_bus_grid())
    load = np.array([[20.]])
    memo.solve(load)
    memo.solve(np.array([[20]], dtype=np.float32))  # Same float64 input values.
    assert (memo.hits, memo.misses) == (1, 1)
    perturbed = load.copy()
    perturbed[0, 0] = np.nextafter(perturbed[0, 0], np.inf)
    memo.solve(perturbed)
    assert memo.misses == 2  # Even a one-bit difference is not rounded away.
    load[0, 0] = 21  # The original key does not retain a reference to this array.
    memo.solve(np.array([[20.]]))
    assert memo.hits == 2


def test_carbon_cap_and_objective_cannot_share_an_entry():
    spec = one_bus_grid()
    plain, memo = GridClearing(spec), MemoizedGridClearing(spec)
    load = np.array([[20.]])
    cases = [(None, "cost"), (6., "cost"), (7., "cost"), (None, "emissions")]
    for cap, objective in cases:
        assert_same_result(memo.solve(load, cap, objective), plain.solve(load, cap, objective))
    assert memo.misses == 4
    constrained = memo.solve(load, 6., "cost")
    emissions = memo.solve(load, None, "emissions")
    assert constrained.carbon_price_yuan_per_t > 0
    assert np.isnan(emissions.carbon_price_yuan_per_t)
    assert np.isnan(emissions.lmp_yuan_per_mwh).all()
    assert memo.hits == 2


def test_lru_evicts_least_recently_used_input():
    memo = MemoizedGridClearing(one_bus_grid(), maxsize=2)
    for value in (20., 21., 20., 22., 20., 21.):
        memo.solve(np.array([[value]]))
    assert memo.cache_info() == (2, 4, 2, 2)


def test_mutating_returned_prices_and_dispatch_cannot_pollute_cache():
    spec = default_grid(np.array([10., 20.]), n_steps=2)
    memo = MemoizedGridClearing(spec)
    load = np.tile([5., 10., 8.], (2, 1))
    expected = GridClearing(spec).solve(load)
    first = memo.solve(load)
    first.lmp_yuan_per_mwh[:] = -1234
    first.generation_mw[:] = -5678
    first.flows_mw[:] = 9999
    first.total_generation_cost = -1
    second = memo.solve(load)
    assert_same_result(second, expected)
    assert second is not first
    for field in ("lmp_yuan_per_mwh", "generation_mw", "flows_mw"):
        assert not np.shares_memory(getattr(first, field), getattr(second, field))
    second.generation_mw[:] = -1
    assert_same_result(memo.solve(load), expected)


def test_infeasible_results_are_cached_and_copied():
    memo = MemoizedGridClearing(one_bus_grid())
    load = np.array([[201.]])
    first = memo.solve(load)
    assert not first.feasible
    first.generation_mw[:] = 0
    second = memo.solve(load)
    assert not second.feasible
    assert np.isnan(second.generation_mw).all()
    assert memo.cache_info() == (1, 1, 256, 1)


def test_distinct_network_instances_do_not_share_results():
    first = MemoizedGridClearing(one_bus_grid())
    other_spec = one_bus_grid()
    other_spec.generator_cost_yuan_per_mwh[0] = 200.
    second = MemoizedGridClearing(other_spec)
    load = np.array([[20.]])
    assert first.solve(load).total_generation_cost != second.solve(load).total_generation_cost
    assert first.misses == second.misses == 1
    assert first.hits == second.hits == 0


def test_zero_capacity_and_clear_have_explicit_statistics():
    memo = MemoizedGridClearing(one_bus_grid(), maxsize=0)
    load = np.array([[20.]])
    memo.solve(load)
    memo.solve(load)
    assert memo.cache_info() == (0, 2, 0, 0)
    memo.cache_clear()
    assert memo.cache_info() == (0, 0, 0, 0)
