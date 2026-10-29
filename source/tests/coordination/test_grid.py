"""Economic and physical checks for sparse grid clearing."""
import numpy as np
import pytest

from energy_net.coordination.grid import GridClearing, GridSpec, default_grid


def one_bus_grid(n_steps=2, dt_hours=0.5):
    return GridSpec(
        n_steps=n_steps, dt_hours=dt_hours, bus_names=("single",),
        generator_bus=np.array([0, 0]),
        generator_capacity_mw=np.full((n_steps, 2), 100.0),
        generator_cost_yuan_per_mwh=np.array([100.0, 300.0]),
        generator_emissions_t_per_mwh=np.array([1.0, 0.1]),
        line_from=np.array([], dtype=int), line_to=np.array([], dtype=int),
        line_reactance=np.array([]), line_capacity_mw=np.array([]),
    )


def test_dispatch_obeys_nodal_balance_lines_and_total_carbon_cap():
    spec = default_grid(np.array([10.0, 15.0, 20.0, 10.0]), n_steps=4)
    grid = GridClearing(spec)
    load = np.tile([5.0, 25.0, 20.0], (4, 1))
    baseline = grid.solve(load)
    budget = baseline.emissions_t * 0.95
    result = grid.solve(load, budget)
    assert result.feasible
    assert result.max_balance_error < 1e-7
    assert np.all(result.generation_mw >= -1e-8)
    assert np.all(result.generation_mw <= spec.generator_capacity_mw + 1e-8)
    assert np.all(np.abs(result.flows_mw) <= spec.line_capacity_mw + 1e-8)
    assert result.emissions_t == pytest.approx(budget, abs=1e-6)
    assert result.total_generation_cost > baseline.total_generation_cost
    assert result.carbon_price_yuan_per_t > 0


def test_carbon_shadow_price_matches_budget_relaxation():
    grid = GridClearing(one_bus_grid())
    load = np.full((2, 1), 20.0)
    result = grid.solve(load, carbon_budget_t=12.0)
    relaxed = grid.solve(load, carbon_budget_t=12.001)
    value_of_relaxation = (result.total_generation_cost - relaxed.total_generation_cost) / 0.001
    assert result.feasible and relaxed.feasible
    assert result.carbon_price_yuan_per_t == pytest.approx(200.0 / 0.9)
    assert result.carbon_price_yuan_per_t == pytest.approx(value_of_relaxation, rel=1e-7)


def test_lmp_is_cost_per_mwh_with_half_hour_weight():
    grid = GridClearing(one_bus_grid())
    load = np.full((2, 1), 20.0)
    result = grid.solve(load, carbon_budget_t=12.0)
    perturbed = load.copy()
    perturbed[0, 0] += 0.001
    other = grid.solve(perturbed, carbon_budget_t=12.0)
    marginal_cost_per_mwh = (other.total_generation_cost - result.total_generation_cost) / (0.001 * 0.5)
    assert result.lmp_yuan_per_mwh[0, 0] == pytest.approx(marginal_cost_per_mwh, rel=1e-7)


def test_congestion_creates_locational_price_difference():
    grid = GridClearing(default_grid(np.zeros(3), n_steps=3))
    result = grid.solve(np.tile([5.0, 25.0, 20.0], (3, 1)))
    assert result.feasible
    assert np.any(np.isclose(np.abs(result.flows_mw), grid.spec.line_capacity_mw))
    assert np.all(result.lmp_yuan_per_mwh[:, 2] > result.lmp_yuan_per_mwh[:, 0])


def test_infeasible_cap_and_excess_demand_are_not_silently_shed():
    grid = GridClearing(one_bus_grid())
    impossible_carbon = grid.solve(np.full((2, 1), 20.0), carbon_budget_t=0.0)
    impossible_supply = grid.solve(np.full((2, 1), 201.0))
    for result in (impossible_carbon, impossible_supply):
        assert not result.feasible
        assert "infeasible" in result.status.lower()
        assert np.isnan(result.generation_mw).all()
        assert np.isinf(result.total_generation_cost)


def test_emissions_objective_is_physical_lower_bound_not_currency_prices():
    grid = GridClearing(one_bus_grid())
    result = grid.solve(np.full((2, 1), 20.0), objective="emissions")
    assert result.feasible
    assert result.emissions_t == pytest.approx(2.0)
    assert np.isnan(result.lmp_yuan_per_mwh).all()
    assert np.isnan(result.carbon_price_yuan_per_t)


def test_linear_problem_exposes_background_balance_rows_for_task_oracle():
    spec = default_grid(np.array([20.0, 10.0]), n_steps=2)
    grid = GridClearing(spec)
    load = np.tile([5.0, 10.0, 8.0], (2, 1))
    problem = grid.linear_problem(load, 30.0)
    np.testing.assert_allclose(problem["b_eq"][problem["balance_rows"]], load)
    assert problem["A_eq"].shape == (2 * (3 + 3), 2 * (4 + 3 + 3))
    np.testing.assert_allclose(problem["A_ub"].toarray()[0, problem["generation_slice"]],
                               np.tile(spec.generator_emissions_t_per_mwh, 2) * 0.5)


def test_sparse_solver_matches_pypsa_with_weighted_carbon_constraint():
    pytest.importorskip("pypsa")
    spec = default_grid(np.array([10.0, 15.0, 20.0, 10.0]), n_steps=4)
    grid = GridClearing(spec)
    load = np.tile([5.0, 25.0, 20.0], (4, 1))
    budget = grid.solve(load).emissions_t * 0.95
    fast = grid.solve(load, budget)
    reference = grid.solve_pypsa_reference(load, budget)
    assert reference.feasible
    assert reference.max_balance_error < 1e-7
    assert fast.total_generation_cost == pytest.approx(reference.total_generation_cost, rel=1e-8)
    assert fast.emissions_t == pytest.approx(reference.emissions_t, abs=1e-7)
    # A horizon-wide cap and constant generator costs can leave multiple optimal
    # allocations across hours. Compare their economic values and physical laws,
    # rather than requiring both solvers to choose the same optimal basis.
    assert np.all(reference.generation_mw <= spec.generator_capacity_mw + 1e-7)
    assert np.all(np.abs(reference.flows_mw) <= spec.line_capacity_mw + 1e-7)
    voltage_drops = reference.flows_mw * spec.line_reactance
    np.testing.assert_allclose(voltage_drops[:, 0] + voltage_drops[:, 1],
                               voltage_drops[:, 2], atol=1e-7)
    np.testing.assert_allclose(fast.lmp_yuan_per_mwh, reference.lmp_yuan_per_mwh, atol=1e-6)
    assert fast.carbon_price_yuan_per_t == pytest.approx(reference.carbon_price_yuan_per_t, abs=1e-6)


def test_single_period_pypsa_dispatch_agrees_when_optimum_is_unique():
    pytest.importorskip("pypsa")
    grid = GridClearing(one_bus_grid(n_steps=1))
    load = np.array([[20.0]])
    fast = grid.solve(load, carbon_budget_t=6.0)
    reference = grid.solve_pypsa_reference(load, carbon_budget_t=6.0)
    assert reference.feasible
    np.testing.assert_allclose(fast.generation_mw, reference.generation_mw, atol=1e-7)
    assert fast.total_generation_cost == pytest.approx(reference.total_generation_cost)
    assert fast.carbon_price_yuan_per_t == pytest.approx(reference.carbon_price_yuan_per_t)


def test_pypsa_reactance_uses_110kv_100mva_reference_units():
    pytest.importorskip("pypsa")
    spec = default_grid(np.zeros(2), n_steps=2)
    network = GridClearing(spec).to_pypsa(np.full((2, 3), 5.0))
    np.testing.assert_allclose(network.buses.v_nom, 110.0)
    z_base_ohm = 110.0**2 / 100.0
    np.testing.assert_allclose(network.lines.x.to_numpy() / z_base_ohm,
                               [0.10, 0.12, 0.15])
    np.testing.assert_allclose(network.lines.x.to_numpy() / 110.0**2,
                               spec.line_reactance)
