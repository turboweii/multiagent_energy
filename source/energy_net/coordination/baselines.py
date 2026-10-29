"""Physical task oracle and explicitly finite posted-price benchmarks.

The physical oracle has complete information and labels its objective explicitly;
it is not an implementable tariff or a payment mechanism. The tariff search evaluates
only its disclosed finite candidate family with sequential exact task responses;
it does not claim a global bilevel optimum or a global owner best response.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

from .grid import GridClearing
from .market import MarketEpisode, MarketOutcome, encode_platform_action, zero_platform_action
from .scenario import Scenario


@dataclass
class OracleResult:
    metrics: dict
    selected_options: tuple[int, ...] | None
    option_weights: tuple[tuple[float, ...], ...]
    allocation_cu: np.ndarray
    transfer_gb: np.ndarray
    load_mw: np.ndarray
    generation_mw: np.ndarray
    flows_mw: np.ndarray
    optimal: bool
    relaxed: bool
    status: str
    label: str

    @property
    def objective_yuan(self):
        return self.metrics["oracle_objective_yuan"]

    def to_dict(self):
        return dict(self.metrics, benchmark=self.label, oracle_optimal=self.optimal,
                    oracle_relaxed=self.relaxed, solver_status=self.status)


@dataclass
class SearchResult:
    best_outcome: MarketOutcome
    baseline_outcome: MarketOutcome
    best_action: np.ndarray
    best_candidate: str
    evaluations: tuple[dict, ...]
    label: str = "finite_candidate_best"

    def to_dict(self):
        return dict(self.best_outcome.metrics, benchmark=self.label,
                    best_candidate=self.best_candidate, candidates_evaluated=len(self.evaluations))


def _option_columns(scenario):
    """Keep complete, fixed-work options; never optimize task shedding."""
    offsets, flat = [0], []
    for job in scenario.jobs:
        if not job.options:
            raise ValueError(f"Job {job.job_id} has no execution options.")
        for option in job.options:
            allocation = np.asarray(option.allocation_cu, dtype=float)
            if np.any(allocation < 0) or not np.isclose(allocation.sum(), job.work_cu):
                raise ValueError(f"Option for {job.job_id} must preserve its complete work.")
            if (option.start_step < job.release_step or option.end_step > job.deadline_step
                    or np.any(allocation[:job.release_step] > 1e-9)
                    or np.any(allocation[job.deadline_step:] > 1e-9)):
                raise ValueError(f"Option for {job.job_id} violates arrival or deadline.")
            flat.append(option)
        offsets.append(len(flat))
    if not flat:
        raise ValueError("The task oracle requires at least one job.")
    return np.asarray(offsets), flat


def physical_oracle(scenario: Scenario, relax=False, time_limit=5.0,
                    carbon_objective_weight: float | None = None) -> OracleResult:
    """Jointly assign all complete jobs and clear the grid under physical caps.

    Binary variables choose one complete option per job. The optional continuous
    relaxation can split a job across options and is labelled as a lower bound;
    it is not an executable task plan. A time-limited integer incumbent is also
    labelled explicitly, rather than reported as a proved physical optimum.
    Posted platform quotas and payment/participation constraints are excluded.
    By default the objective includes generation-side policy carbon payments,
    while reported system resource cost excludes those transfers. Set
    ``carbon_objective_weight=0.0`` for a pure resource-plus-user-loss physical
    lower bound; a positive explicit value is an emissions weight in yuan/t.
    This changes the oracle's objective, not the market's carbon policy or its
    physical cap. A carbon-weighted optimum is not a proved minimum of the
    unweighted resource cost.
    """
    cfg, spec = scenario.config, scenario.grid_spec
    objective_carbon_price = (spec.carbon_price_yuan_per_t
                              if carbon_objective_weight is None else carbon_objective_weight)
    if (not np.isscalar(objective_carbon_price) or not np.isfinite(objective_carbon_price)
            or objective_carbon_price < 0):
        raise ValueError("carbon_objective_weight must be finite and nonnegative or None")
    t, r = cfg.n_steps, cfg.n_regions
    offsets, options = _option_columns(scenario)
    k = len(options)
    grid = GridClearing(spec)
    problem = grid.linear_problem(scenario.background_mw + scenario.idle_load_mw,
                                  scenario.carbon_budget_t)
    n = len(problem["c"])
    power = np.column_stack([scenario.option_power_mw(o).ravel() for o in options])
    allocation = np.column_stack([o.allocation_cu.ravel() for o in options])
    transfer = np.column_stack([o.transfer_gb.ravel() for o in options])

    # gen - outgoing + incoming - chosen task power = background + idle.
    nonzero = coo_matrix(power)
    load_columns = coo_matrix(
        (-nonzero.data, (problem["balance_rows"].ravel()[nonzero.row], nonzero.col)),
        shape=(problem["A_eq"].shape[0], k),
    ).tocsr()
    job_rows = np.repeat(np.arange(len(scenario.jobs)), np.diff(offsets))
    one_option = coo_matrix((np.ones(k), (job_rows, np.arange(k))),
                           shape=(len(scenario.jobs), k)).tocsr()
    eq = vstack([
        hstack([problem["A_eq"], load_columns]),
        hstack([csr_matrix((len(scenario.jobs), n)), one_option]),
    ], format="csr")
    eq_rhs = np.r_[problem["b_eq"], np.ones(len(scenario.jobs))]
    matrices, lows, highs = [eq], [eq_rhs], [eq_rhs]
    if problem["A_ub"] is not None:
        matrices.append(hstack([problem["A_ub"], csr_matrix((len(problem["b_ub"]), k))]))
        lows.append(np.full(len(problem["b_ub"]), -np.inf))
        highs.append(problem["b_ub"])
    for values, cap in (
        (allocation, np.broadcast_to(np.asarray(cfg.capacity_cu_per_hour)
                                     * cfg.dt_hours, (t, r)).ravel()),
        (transfer, np.full(t * r * r, cfg.bandwidth_gb_per_hour * cfg.dt_hours)),
    ):
        rows = np.any(values != 0, axis=1)
        matrices.append(hstack([csr_matrix((int(rows.sum()), n)), csr_matrix(values[rows])]))
        lows.append(np.full(int(rows.sum()), -np.inf))
        highs.append(cap[rows])
    matrix = vstack(matrices, format="csc")
    low, high = np.concatenate(lows), np.concatenate(highs)
    extra_costs = np.array([o.operating_cost_yuan + o.user_loss_yuan for o in options])
    dispatch_cost = problem["c"].copy()
    dispatch_cost[problem["generation_slice"]] = np.tile(
        spec.generator_cost_yuan_per_mwh
        + objective_carbon_price * spec.generator_emissions_t_per_mwh, t) * cfg.dt_hours
    objective = np.r_[dispatch_cost, extra_costs]
    integrality = np.r_[np.zeros(n, dtype=int), np.full(k, 0 if relax else 1, dtype=int)]
    solver_options = {"mip_rel_gap": 0.0}
    if time_limit is not None:
        if time_limit <= 0:
            raise ValueError("time_limit must be positive or None.")
        solver_options["time_limit"] = float(time_limit)
    result = milp(objective, integrality=integrality,
                  bounds=Bounds(np.r_[problem["lower"], np.zeros(k)],
                                np.r_[problem["upper"], np.ones(k)]),
                  constraints=LinearConstraint(matrix, low, high), options=solver_options)
    if result.x is None:
        raise RuntimeError(f"No feasible task-oracle solution: {result.message}")
    activity = matrix @ result.x
    if np.any(activity < low - 1e-5) or np.any(activity > high + 1e-5):
        raise RuntimeError("Task-oracle solver returned an infeasible incumbent.")
    weights = result.x[n:]
    if not relax and np.max(np.abs(weights - np.rint(weights))) > 1e-5:
        raise RuntimeError("Task-oracle incumbent does not select integral options.")
    chunks = tuple(tuple(float(w) for w in weights[a:b])
                   for a, b in zip(offsets[:-1], offsets[1:]))
    integral = np.max(np.abs(weights - np.rint(weights))) <= 1e-5
    indices = tuple(int(np.argmax(w)) for w in chunks) if integral else None
    task_allocation = (allocation @ weights).reshape(t, r)
    task_transfer = (transfer @ weights).reshape(t, r, r)
    load = scenario.background_mw + scenario.idle_load_mw + (power @ weights).reshape(t, r)
    generation = result.x[problem["generation_slice"]].reshape(t, spec.n_generators)
    flows = result.x[problem["flow_slice"]].reshape(t, spec.n_lines)
    # The optimisation vector includes policy carbon payments. Keep those
    # transfers separate from fuel/variable generation resource consumption.
    gen_cost = float(np.sum(generation * spec.generator_cost_yuan_per_mwh) * cfg.dt_hours)
    opex = float(np.array([o.operating_cost_yuan for o in options]) @ weights)
    loss = float(np.array([o.user_loss_yuan for o in options]) @ weights)
    emissions = float(np.sum(generation * spec.generator_emissions_t_per_mwh) * cfg.dt_hours)
    carbon_payment = float(spec.carbon_price_yuan_per_t * emissions)
    optimal = result.status == 0
    label = ("physical_lp_relaxation" if relax else
             "physical_milp_optimum" if optimal else "physical_milp_incumbent")
    metrics = {
        "seed": scenario.seed, "jobs": len(scenario.jobs),
        "work_cu": float(task_allocation.sum()), "required_work_cu": scenario.total_work_cu,
        "completion_rate": float(task_allocation.sum() / scenario.total_work_cu),
        "deadline_compliance": 1.0, "generation_cost_yuan": gen_cost,
        "compute_migration_cost_yuan": opex, "resource_cost_yuan": gen_cost + opex,
        "user_loss_yuan": loss, "system_cost_yuan": gen_cost + opex + loss,
        "policy_carbon_price_yuan_per_t": spec.carbon_price_yuan_per_t,
        "carbon_payment_yuan": carbon_payment,
        "dispatch_objective_yuan": gen_cost + carbon_payment,
        "oracle_objective_yuan": float(objective @ result.x),
        "oracle_carbon_objective_weight_yuan_per_t": float(objective_carbon_price),
        "oracle_objective_name": (
            "generation_resource_plus_operating_plus_user_loss" if objective_carbon_price == 0
            else "generation_resource_plus_policy_carbon_payment_plus_operating_plus_user_loss"
            if carbon_objective_weight is None else
            "generation_resource_plus_carbon_weighted_emissions_plus_operating_plus_user_loss"),
        "carbon_emissions_t": emissions, "carbon_budget_t": scenario.carbon_budget_t,
        "max_balance_error_mw": float(np.max(np.abs((eq @ result.x - eq_rhs)
                                                    [:len(problem["b_eq"])][problem["balance_rows"]]))),
    }
    if not relax:
        metrics["mip_gap"] = float(getattr(result, "mip_gap", np.nan))
        metrics["objective_lower_bound_yuan"] = float(getattr(result, "mip_dual_bound", np.nan))
    return OracleResult(metrics, indices, chunks, task_allocation, task_transfer, load,
                        generation, flows, optimal, bool(relax), str(result.message), label)


def _scores(cost):
    spread = float(np.ptp(cost))
    return np.zeros_like(cost) if spread < 1e-12 else (np.max(cost) - cost) / spread


def _price_families(scenario, episode):
    """Disclosed nested candidate families, with duplicate actions removed later.

    Discount patterns use public original-contract clearing, not private user
    preferences. Four six-hour blocks are fixed by the scenario. Quotas apply
    to additional reservations while retaining every original contract.
    """
    cfg = scenario.config
    blocks = np.minimum(np.arange(cfg.n_steps) * cfg.n_blocks // cfg.n_steps,
                        cfg.n_blocks - 1)
    unit_cost = episode.baseline_clearing.lmp_yuan_per_mwh * np.asarray(cfg.energy_mwh_per_cu)
    cost = np.column_stack([unit_cost[blocks == b].mean(axis=0) for b in range(cfg.n_blocks)])
    temporal = np.broadcast_to(_scores(cost.mean(axis=0)), cost.shape)
    spatial_temporal = _scores(cost)
    families = {name: {} for name in ("uniform", "time_only", "time_region", "reservation_plan")}
    uniform = zero_platform_action(cfg)
    for family in families.values():
        family["uniform"] = uniform
    for amplitude in (0.10, 0.20, 0.30):
        families["time_only"][f"time_only_{amplitude:.2f}"] = encode_platform_action(
            amplitude * temporal, config=cfg)
        families["time_region"][f"time_region_{amplitude:.2f}"] = encode_platform_action(
            amplitude * spatial_temporal, config=cfg)
    families["time_region"].update(families["time_only"])
    families["reservation_plan"].update(families["time_region"])
    for amplitude in (0.10, 0.20, 0.30):
        for quota in (0.50, 0.75, 1.00):
            name = f"reservation_{amplitude:.2f}_{quota:.2f}"
            families["reservation_plan"][name] = encode_platform_action(
                amplitude * spatial_temporal, np.full(cfg.n_regions, quota), cfg)
    return families


def finite_price_candidates(scenario: Scenario) -> dict[str, np.ndarray]:
    """The explicit finite family, including the original uniform tariff."""
    episode = MarketEpisode(scenario)
    return {name: action.copy() for family in _price_families(scenario, episode).values()
            for name, action in family.items()}


def _eligible(outcome, baseline):
    m = outcome.metrics
    return bool(outcome.clearing.feasible
                and np.isclose(m["work_cu"], m["required_work_cu"])
                and m["deadline_compliance"] >= 1 - 1e-9
                and m["user_ir_rate"] >= 1 - 1e-9
                and m["platform_profit_yuan"] >= baseline.metrics["platform_profit_yuan"] - 1e-6)


def _search(episode, candidates, cache):
    zero = zero_platform_action(episode.config)
    key = tuple(zero)
    if key not in cache:
        episode.reset(zero)
        cache[key] = episode.play()
    baseline = cache[key]
    if not _eligible(baseline, baseline):
        raise ValueError("Uniform original contracts must be a feasible, individually rational fallback.")
    best, best_name, best_action = baseline, "uniform", zero.copy()
    evaluations, seen = [], set()
    for name, action in [("uniform", zero), *candidates.items()]:
        action = np.asarray(action, dtype=int)
        key = tuple(action)
        if key in seen:
            continue
        seen.add(key)
        if key not in cache:
            episode.reset(action)
            cache[key] = episode.play()
        outcome = cache[key]
        eligible = _eligible(outcome, baseline)
        evaluations.append(dict(outcome.metrics, candidate=name, eligible=eligible,
                                platform_action=action.tolist()))
        if eligible and outcome.metrics["system_cost_yuan"] < best.metrics["system_cost_yuan"] - 1e-7:
            best, best_name, best_action = outcome, name, action.copy()
    annotated = replace(best, metrics=dict(best.metrics, benchmark_scope="finite_candidates_only",
                                           best_candidate=best_name,
                                           candidates_evaluated=len(evaluations)))
    return SearchResult(annotated, baseline, best_action, best_name, tuple(evaluations))


def finite_price_search(scenario: Scenario, candidates=None) -> SearchResult:
    """Best eligible member of this finite family, always retaining baseline.

    ``candidates`` may be a name-to-action mapping or an iterable of actions.
    The best member minimizes actual generation + operating + user-loss cost
    subject to all users' IR and platform profit no lower than uniform pricing.
    """
    episode = MarketEpisode(scenario)
    if candidates is None:
        candidates = {name: action for family in _price_families(scenario, episode).values()
                      for name, action in family.items()}
    elif not isinstance(candidates, Mapping):
        candidates = {f"candidate_{i}": action for i, action in enumerate(candidates)}
    return _search(episode, candidates, {})


def evaluate_baselines(scenario: Scenario) -> dict[str, MarketOutcome]:
    """Four finite tariff-family benchmarks using identical tasks and order.

    Each family includes the uniform fallback and uses physical reservation
    admission. ``reservation_plan`` additionally searches public booking quotas;
    it is not a centralized or globally optimal assignment of users.
    """
    episode = MarketEpisode(scenario)
    families = _price_families(scenario, episode)
    cache, outcomes = {}, {}
    for family, candidates in families.items():
        result = _search(episode, candidates, cache)
        outcomes[family] = replace(result.best_outcome, metrics=dict(
            result.best_outcome.metrics, benchmark=family,
            benchmark_scope="finite_candidates_only"))
    return outcomes
