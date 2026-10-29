"""Carbon mass conservation, flow reversal and realised load-shift effects."""
from dataclasses import replace

import numpy as np
import pytest

from energy_net.coordination.carbon import (
    consumption_emissions_t, nodal_carbon_intensity, reference_lme,
)
from energy_net.coordination.grid import ClearingResult, GridClearing, GridSpec


def two_bus_grid():
    return GridSpec(
        n_steps=2, dt_hours=0.5, bus_names=("coal", "renewable"),
        generator_bus=np.array([0, 1]),
        generator_capacity_mw=np.array([[100., 0.], [100., 20.]]),
        generator_cost_yuan_per_mwh=np.array([100., 0.]),
        generator_emissions_t_per_mwh=np.array([1., 0.]),
        line_from=np.array([0]), line_to=np.array([1]),
        line_reactance=np.array([.001]), line_capacity_mw=np.array([100.]),
    )


def test_flow_tracing_reverses_with_power_and_conserves_hourly_carbon():
    spec = two_bus_grid()
    load = np.full((2, 2), 5.)
    clearing = GridClearing(spec).solve(load)
    assert clearing.flows_mw[0, 0] > 0
    assert clearing.flows_mw[1, 0] < 0
    intensity = nodal_carbon_intensity(spec, clearing, load)
    np.testing.assert_allclose(intensity, [[1., 1.], [0., 0.]])
    np.testing.assert_allclose(np.sum(load * intensity, axis=1),
                               clearing.generation_mw @ spec.generator_emissions_t_per_mwh)
    assert consumption_emissions_t(load, intensity, spec.dt_hours) == pytest.approx(clearing.emissions_t)
    dc_power = np.array([[1., 2.], [1., 1.]])
    assert consumption_emissions_t(dc_power, intensity, spec.dt_hours) == pytest.approx(1.5)


def test_intermediate_node_mixes_local_clean_generation_and_dirty_imports():
    spec = GridSpec(
        n_steps=1, dt_hours=.5, bus_names=("dirty", "mixed", "consumer"),
        generator_bus=np.array([0, 1]), generator_capacity_mw=np.array([[15., 15.]]),
        generator_cost_yuan_per_mwh=np.array([0., 100.]),
        generator_emissions_t_per_mwh=np.array([1., 0.]),
        line_from=np.array([0, 1]), line_to=np.array([1, 2]),
        line_reactance=np.array([.001, .001]), line_capacity_mw=np.array([30., 30.]),
    )
    load = np.array([[5., 10., 15.]])
    clearing = GridClearing(spec).solve(load)
    np.testing.assert_allclose(nodal_carbon_intensity(spec, clearing, load), [[1., .4, .4]])


def test_zero_power_island_gets_zero_intensity_without_inventing_carbon():
    spec = two_bus_grid()
    spec = replace(spec, generator_capacity_mw=np.zeros((2, 2)), line_capacity_mw=np.zeros(1))
    load = np.zeros((2, 2))
    clearing = GridClearing(spec).solve(load)
    np.testing.assert_array_equal(nodal_carbon_intensity(spec, clearing, load), np.zeros((2, 2)))


def test_source_free_circulation_assigns_zero_and_does_not_create_carbon():
    spec = GridSpec(
        n_steps=1, dt_hours=.5, bus_names=("a", "b", "c"),
        generator_bus=np.array([0]), generator_capacity_mw=np.zeros((1, 1)),
        generator_cost_yuan_per_mwh=np.array([1.]),
        generator_emissions_t_per_mwh=np.array([1.]),
        line_from=np.array([0, 1, 2]), line_to=np.array([1, 2, 0]),
        line_reactance=np.full(3, .001), line_capacity_mw=np.full(3, 10.),
    )
    # An external flow result can contain a source-free circulation even though
    # the DC angle model itself cannot sustain this directed cycle.
    clearing = ClearingResult(True, 0., 0., np.zeros((1, 3)), 0.,
                              np.zeros((1, 1)), np.ones((1, 3)), 0., "external")
    np.testing.assert_allclose(nodal_carbon_intensity(spec, clearing, np.zeros((1, 3))), 0.)


def test_same_work_shift_reduces_actual_grid_emissions_not_only_allocated_footprint():
    spec = two_bus_grid()
    grid = GridClearing(spec)
    original = np.full((2, 2), 5.)
    shifted = original.copy()
    shifted[0, 0] -= 1.
    shifted[1, 1] += 1.
    assert shifted.sum() == original.sum()
    old, new = grid.solve(original), grid.solve(shifted)
    assert old.emissions_t - new.emissions_t == pytest.approx(.5)
    lme = reference_lme(grid, original)
    np.testing.assert_allclose(lme, [[1., 1.], [0., 0.]], atol=1e-8)
    predicted_change = np.sum((shifted - original) * lme) * spec.dt_hours
    assert new.emissions_t - old.emissions_t == pytest.approx(predicted_change)


def test_carbon_accounting_rejects_unbalanced_or_infeasible_results():
    spec = two_bus_grid()
    load = np.full((2, 2), 5.)
    clearing = GridClearing(spec).solve(load)
    with pytest.raises(ValueError, match="balance"):
        nodal_carbon_intensity(spec, clearing, load + 1.)
    with pytest.raises(ValueError, match="feasible"):
        nodal_carbon_intensity(spec, replace(clearing, feasible=False), load)
