"""Carbon payments, physical costs and cap scarcity have distinct meanings."""
from dataclasses import replace

import numpy as np
import pytest

from energy_net.coordination.grid import GridClearing, GridSpec, default_grid


def priced_grid(price=0.0):
    return GridSpec(
        n_steps=2, dt_hours=0.5, bus_names=("single",),
        generator_bus=np.array([0, 0]),
        generator_capacity_mw=np.full((2, 2), 100.0),
        generator_cost_yuan_per_mwh=np.array([100.0, 300.0]),
        generator_emissions_t_per_mwh=np.array([1.0, 0.1]),
        line_from=np.array([], dtype=int), line_to=np.array([], dtype=int),
        line_reactance=np.array([]), line_capacity_mw=np.array([]),
        carbon_price_yuan_per_t=price,
    )


@pytest.mark.parametrize("price", [-1.0, np.inf, np.nan])
def test_policy_price_must_be_finite_and_nonnegative(price):
    with pytest.raises(ValueError, match="carbon_price"):
        priced_grid(price)


def test_carbon_price_changes_dispatch_without_a_binding_emissions_cap():
    load = np.full((2, 1), 20.0)
    unpriced = GridClearing(priced_grid()).solve(load)
    priced = GridClearing(priced_grid(500.0)).solve(load)
    assert unpriced.feasible and priced.feasible
    assert unpriced.emissions_t == pytest.approx(20.0)
    assert priced.emissions_t == pytest.approx(2.0)
    assert priced.total_generation_cost == pytest.approx(6000.0)
    assert priced.carbon_payment_yuan == pytest.approx(1000.0)
    assert priced.dispatch_objective_yuan == pytest.approx(7000.0)
    np.testing.assert_allclose(priced.lmp_yuan_per_mwh, 350.0)
    assert priced.policy_carbon_price_yuan_per_t == 500.0
    assert priced.cap_shadow_price_yuan_per_t == 0.0
    assert priced.carbon_price_yuan_per_t == 0.0


def test_lmp_values_perturbation_of_cost_plus_carbon_payment():
    grid = GridClearing(priced_grid(100.0))
    load = np.full((2, 1), 20.0)
    base = grid.solve(load, 12.0)
    perturbed = load.copy()
    perturbed[0, 0] += 0.001
    other = grid.solve(perturbed, 12.0)
    derivative = ((other.dispatch_objective_yuan - base.dispatch_objective_yuan)
                  / (0.001 * 0.5))
    assert base.lmp_yuan_per_mwh[0, 0] == pytest.approx(derivative, rel=1e-7)
    assert base.policy_carbon_price_yuan_per_t == 100.0
    assert base.cap_shadow_price_yuan_per_t == pytest.approx(200 / 0.9 - 100)
    relaxed = grid.solve(load, 12.001)
    assert base.cap_shadow_price_yuan_per_t == pytest.approx(
        (base.dispatch_objective_yuan - relaxed.dispatch_objective_yuan) / 0.001,
        rel=1e-7)


def test_emissions_objective_keeps_resource_and_policy_payment_accounts():
    result = GridClearing(priced_grid(50.0)).solve(
        np.full((2, 1), 20.0), objective="emissions")
    assert result.emissions_t == pytest.approx(2.0)
    assert result.total_generation_cost == pytest.approx(6000.0)
    assert result.carbon_payment_yuan == pytest.approx(100.0)
    assert result.dispatch_objective_yuan == pytest.approx(6100.0)
    assert np.isnan(result.cap_shadow_price_yuan_per_t)


@pytest.mark.parametrize("budget", [None, 75.0])
def test_priced_sparse_dispatch_matches_pypsa_reference(budget):
    pytest.importorskip("pypsa")
    spec = replace(default_grid(np.array([10.0, 15.0, 20.0, 10.0]), n_steps=4),
                   carbon_price_yuan_per_t=100.0)
    grid = GridClearing(spec)
    load = np.tile([5.0, 25.0, 20.0], (4, 1))
    fast = grid.solve(load, budget)
    reference = grid.solve_pypsa_reference(load, budget)
    assert fast.feasible and reference.feasible
    assert fast.total_generation_cost == pytest.approx(reference.total_generation_cost)
    assert fast.emissions_t == pytest.approx(reference.emissions_t)
    assert fast.carbon_payment_yuan == pytest.approx(reference.carbon_payment_yuan)
    assert fast.dispatch_objective_yuan == pytest.approx(reference.dispatch_objective_yuan)
    np.testing.assert_allclose(fast.lmp_yuan_per_mwh, reference.lmp_yuan_per_mwh, atol=1e-7)
    assert fast.cap_shadow_price_yuan_per_t == pytest.approx(reference.cap_shadow_price_yuan_per_t)
    assert reference.policy_carbon_price_yuan_per_t == 100.0


def test_physical_oracle_separates_resource_cost_from_its_carbon_priced_objective():
    from energy_net.coordination.baselines import physical_oracle
    from energy_net.coordination.scenario import MarketConfig, make_scenario

    original = make_scenario(7, MarketConfig(jobs_per_user=2))
    scenario = replace(original, carbon_budget_t=None,
                       grid_spec=replace(original.grid_spec, carbon_price_yuan_per_t=500.0))
    oracle = physical_oracle(scenario, time_limit=10.0)
    metrics, spec = oracle.metrics, scenario.grid_spec
    resource_generation = float(np.sum(oracle.generation_mw
                                      * spec.generator_cost_yuan_per_mwh) * spec.dt_hours)
    emissions = float(np.sum(oracle.generation_mw
                            * spec.generator_emissions_t_per_mwh) * spec.dt_hours)
    assert oracle.optimal
    assert metrics["generation_cost_yuan"] == pytest.approx(resource_generation)
    assert metrics["carbon_payment_yuan"] == pytest.approx(500.0 * emissions)
    assert metrics["system_cost_yuan"] == pytest.approx(
        resource_generation + metrics["compute_migration_cost_yuan"] + metrics["user_loss_yuan"])
    assert oracle.objective_yuan == pytest.approx(
        metrics["system_cost_yuan"] + metrics["carbon_payment_yuan"])
    assert metrics["dispatch_objective_yuan"] == pytest.approx(
        resource_generation + metrics["carbon_payment_yuan"])
    assert metrics["objective_lower_bound_yuan"] == pytest.approx(oracle.objective_yuan)
    assert metrics["oracle_objective_name"] == (
        "generation_resource_plus_policy_carbon_payment_plus_operating_plus_user_loss")

    resource_oracle = physical_oracle(scenario, time_limit=10.0, carbon_objective_weight=0.0)
    assert resource_oracle.optimal
    assert resource_oracle.objective_yuan == pytest.approx(resource_oracle.metrics["system_cost_yuan"])
    assert resource_oracle.objective_yuan <= metrics["system_cost_yuan"] + 1e-6
    assert resource_oracle.metrics["carbon_emissions_t"] >= emissions - 1e-6
    assert resource_oracle.metrics["oracle_carbon_objective_weight_yuan_per_t"] == 0.0
    # The objective override never erases the actual market-policy payment.
    assert resource_oracle.metrics["policy_carbon_price_yuan_per_t"] == 500.0
    assert resource_oracle.metrics["carbon_payment_yuan"] == pytest.approx(
        500.0 * resource_oracle.metrics["carbon_emissions_t"])
    assert resource_oracle.metrics["oracle_objective_name"] == (
        "generation_resource_plus_operating_plus_user_loss")


@pytest.mark.parametrize("weight", [-1.0, np.inf, np.nan])
def test_physical_oracle_rejects_invalid_carbon_objective_weight(weight):
    from energy_net.coordination.baselines import physical_oracle
    from energy_net.coordination.scenario import MarketConfig, make_scenario

    with pytest.raises(ValueError, match="carbon_objective_weight"):
        physical_oracle(make_scenario(7, MarketConfig(jobs_per_user=1)),
                        carbon_objective_weight=weight)

