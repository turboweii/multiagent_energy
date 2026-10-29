"""Fixed-job oracle and finite-candidate participation guarantees."""
import numpy as np
import pytest

from energy_net.coordination.baselines import (
    evaluate_baselines, finite_price_candidates, finite_price_search, physical_oracle,
)
from energy_net.coordination.market import MarketEpisode, encode_platform_action
from energy_net.coordination.scenario import MarketConfig, make_scenario


@pytest.fixture(scope="module")
def scenario():
    return make_scenario(seed=7, config=MarketConfig(jobs_per_user=2))


def test_physical_oracle_keeps_all_jobs_deadlines_and_resource_caps(scenario):
    baseline = MarketEpisode(scenario).play()
    oracle = physical_oracle(scenario)
    assert oracle.optimal, oracle.status
    assert oracle.selected_options is not None
    assert oracle.label == "physical_milp_optimum"
    for job, index, weights in zip(scenario.jobs, oracle.selected_options, oracle.option_weights):
        option = job.options[index]
        assert sum(weights) == pytest.approx(1.0)
        assert option.allocation_cu.sum() == pytest.approx(job.work_cu)
        assert option.start_step >= job.release_step
        assert option.end_step <= job.deadline_step
    cfg = scenario.config
    assert np.all(oracle.allocation_cu <= np.asarray(cfg.capacity_cu_per_hour) * cfg.dt_hours + 1e-6)
    assert np.all(oracle.transfer_gb <= cfg.bandwidth_gb_per_hour * cfg.dt_hours + 1e-6)
    assert np.all(np.abs(oracle.flows_mw) <= scenario.grid_spec.line_capacity_mw + 1e-6)
    assert oracle.metrics["carbon_emissions_t"] <= scenario.carbon_budget_t + 1e-6
    assert oracle.metrics["completion_rate"] == pytest.approx(1.0)
    assert oracle.metrics["max_balance_error_mw"] < 1e-6
    assert oracle.objective_yuan <= baseline.metrics["system_cost_yuan"] + 1e-5


def test_lp_relaxation_is_labelled_and_a_lower_bound(scenario):
    integer = physical_oracle(scenario)
    relaxed = physical_oracle(scenario, relax=True)
    assert relaxed.optimal
    assert relaxed.relaxed
    assert relaxed.label == "physical_lp_relaxation"
    assert relaxed.objective_yuan <= integer.objective_yuan + 1e-5
    assert relaxed.metrics["completion_rate"] == pytest.approx(1.0)


def test_empty_finite_search_retains_feasible_uniform_contract(scenario):
    search = finite_price_search(scenario, candidates=[])
    assert search.label == "finite_candidate_best"
    assert search.best_candidate == "uniform"
    assert len(search.evaluations) == 1
    assert search.best_outcome.clearing.feasible
    assert search.best_outcome.metrics["user_ir_rate"] == 1.0
    assert search.best_outcome.metrics["platform_participates"]
    assert search.best_outcome.metrics["system_cost_yuan"] == pytest.approx(
        search.baseline_outcome.metrics["system_cost_yuan"])


def test_unfunded_discount_cannot_replace_uniform_fallback(scenario):
    # Same-time/place service is available under a new invoice. A uniform
    # discount buys no physical saving and reduces platform profit.
    action = encode_platform_action(np.full((3, 4), 0.30), config=scenario.config)
    search = finite_price_search(scenario, {"unfunded_discount": action})
    discounted = next(e for e in search.evaluations if e["candidate"] == "unfunded_discount")
    assert not discounted["eligible"]
    assert search.best_candidate == "uniform"
    assert search.best_outcome.metrics["platform_profit_yuan"] >= (
        search.baseline_outcome.metrics["platform_profit_yuan"] - 1e-6)


def test_finite_families_complete_same_work_with_participation(scenario):
    candidates = finite_price_candidates(scenario)
    assert 4 <= len(candidates) <= 16
    outcomes = evaluate_baselines(scenario)
    assert set(outcomes) == {"uniform", "time_only", "time_region", "reservation_plan"}
    baseline = outcomes["uniform"].metrics
    for family, outcome in outcomes.items():
        metrics = outcome.metrics
        assert metrics["benchmark"] == family
        assert metrics["benchmark_scope"] == "finite_candidates_only"
        assert outcome.clearing.feasible
        assert metrics["work_cu"] == pytest.approx(scenario.total_work_cu)
        assert metrics["completion_rate"] == pytest.approx(1.0)
        assert metrics["deadline_compliance"] == 1.0
        assert metrics["user_ir_rate"] == 1.0
        assert metrics["platform_profit_yuan"] >= baseline["platform_profit_yuan"] - 1e-6
        assert metrics["system_cost_yuan"] <= baseline["system_cost_yuan"] + 1e-6
