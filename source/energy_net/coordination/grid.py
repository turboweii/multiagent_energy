"""Multi-period DC power flow with a carbon price and optional emissions cap.

Power is in MW, energy in MWh, generation cost in yuan/MWh, emissions in
tonnes/MWh, and ``carbon_budget_t`` in tonnes over the complete horizon.
The fast solver and the PyPSA reference describe the same lossless network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix


@dataclass(frozen=True)
class GridSpec:
    """Physical inputs; capacities have shape [time, generator].

    ``generator_bus`` and line endpoints are integer indices into ``bus_names``.
    Generator costs and emissions have shape [generator]; line inputs have
    shape [line]. ``line_reactance`` is effective reactance in rad/MW:
    ``x_effective = x_ohm / nominal_voltage_kv**2``. Thus DC flow is angle
    difference divided by effective reactance. The PyPSA reference converts
    this input to ohms on its 110-kV buses. ``carbon_price_yuan_per_t`` is an
    exogenous generation-side price added to generator offers. It is separate
    from the optional emissions cap's endogenous shadow price.
    """

    generator_bus: np.ndarray
    generator_capacity_mw: np.ndarray
    generator_cost_yuan_per_mwh: np.ndarray
    generator_emissions_t_per_mwh: np.ndarray
    line_from: np.ndarray
    line_to: np.ndarray
    line_reactance: np.ndarray
    line_capacity_mw: np.ndarray
    dt_hours: float = 0.5
    n_steps: int = 48
    bus_names: tuple[str, ...] = ("bus0", "bus1", "bus2")
    carbon_price_yuan_per_t: float = 0.0

    def __post_init__(self):
        for name in ("generator_bus", "line_from", "line_to"):
            values = np.asarray(getattr(self, name))
            if np.any(values != values.astype(int)):
                raise ValueError(f"{name} must contain integer bus indices")
            object.__setattr__(self, name, values.astype(int, copy=True))
        for name in (
            "generator_capacity_mw", "generator_cost_yuan_per_mwh",
            "generator_emissions_t_per_mwh", "line_reactance", "line_capacity_mw",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, values.copy())
        object.__setattr__(self, "bus_names", tuple(self.bus_names))
        if self.n_steps < 1 or self.dt_hours <= 0 or not np.isfinite(self.dt_hours):
            raise ValueError("n_steps and dt_hours must be positive")
        if (not np.isscalar(self.carbon_price_yuan_per_t)
                or not np.isfinite(self.carbon_price_yuan_per_t)
                or self.carbon_price_yuan_per_t < 0):
            raise ValueError("carbon_price_yuan_per_t must be finite and nonnegative")
        b, g, l = self.n_buses, self.n_generators, self.n_lines
        if not b or len(set(self.bus_names)) != b or not g:
            raise ValueError("provide distinct bus names and at least one generator")
        shapes = {
            "generator_bus": (g,), "generator_capacity_mw": (self.n_steps, g),
            "generator_cost_yuan_per_mwh": (g,),
            "generator_emissions_t_per_mwh": (g,), "line_from": (l,),
            "line_to": (l,), "line_reactance": (l,), "line_capacity_mw": (l,),
        }
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        for name in ("generator_bus", "line_from", "line_to"):
            if np.any((getattr(self, name) < 0) | (getattr(self, name) >= b)):
                raise ValueError(f"{name} contains an unknown bus")
        if np.any(self.generator_capacity_mw < 0) or np.any(self.line_capacity_mw < 0):
            raise ValueError("capacities must be nonnegative")
        if np.any(self.generator_emissions_t_per_mwh < 0):
            raise ValueError("generator emissions must be nonnegative")
        if np.any(self.line_reactance <= 0) or np.any(self.line_from == self.line_to):
            raise ValueError("lines need positive reactance and distinct endpoints")

    @property
    def n_buses(self) -> int:
        return len(self.bus_names)

    @property
    def n_generators(self) -> int:
        return len(self.generator_bus)

    @property
    def n_lines(self) -> int:
        return len(self.line_from)


@dataclass
class ClearingResult:
    """Physical resource cost and carbon payments are reported separately.

    ``carbon_price_yuan_per_t`` retains its original meaning as the cap's
    shadow price. Use ``policy_carbon_price_yuan_per_t`` for the external price.
    """
    feasible: bool
    total_generation_cost: float
    emissions_t: float
    lmp_yuan_per_mwh: np.ndarray
    carbon_price_yuan_per_t: float
    generation_mw: np.ndarray
    flows_mw: np.ndarray
    max_balance_error: float
    status: str
    policy_carbon_price_yuan_per_t: float = 0.0

    @property
    def cap_shadow_price_yuan_per_t(self) -> float:
        """Value of relaxing the cap; the historical carbon-price field."""
        return self.carbon_price_yuan_per_t

    @property
    def carbon_payment_yuan(self) -> float:
        """Generation-side carbon payment, already reflected in market LMPs."""
        return (self.policy_carbon_price_yuan_per_t * self.emissions_t
                if self.feasible else np.inf)

    @property
    def dispatch_objective_yuan(self) -> float:
        """Economic dispatch value; resource cost plus the carbon payment.

        The same economic accounting is available for an emissions-minimising
        solve, whose optimisation objective itself is tonnes, not yuan.
        """
        return self.total_generation_cost + self.carbon_payment_yuan


def default_grid(
    renewable_mw: np.ndarray, n_steps: int = 48, dt_hours: float = 0.5
) -> GridSpec:
    """Three-bus study network; renewable availability has shape [time].

    Effective reactances correspond to 0.10/0.12/0.15 pu on a 100-MVA,
    110-kV base. The MW flow equations use 0.001/0.0012/0.0015 rad/MW.
    """
    renewable_mw = np.asarray(renewable_mw, dtype=float)
    if renewable_mw.shape != (n_steps,):
        raise ValueError(f"renewable_mw must have shape ({n_steps},)")
    return GridSpec(
        generator_bus=np.array([0, 1, 2, 2]),
        generator_capacity_mw=np.column_stack([
            np.full(n_steps, 55.0), np.full(n_steps, 35.0),
            renewable_mw, np.full(n_steps, 15.0),
        ]),
        generator_cost_yuan_per_mwh=np.array([180.0, 450.0, 15.0, 520.0]),
        generator_emissions_t_per_mwh=np.array([0.9, 0.35, 0.0, 0.35]),
        line_from=np.array([0, 1, 0]), line_to=np.array([1, 2, 2]),
        line_reactance=np.array([0.001, 0.0012, 0.0015]),
        line_capacity_mw=np.array([18.0, 15.0, 12.0]),
        n_steps=n_steps, dt_hours=dt_hours,
    )


class GridClearing:
    """Sparse HiGHS LP with a PyPSA construction and reference solver."""

    def __init__(self, spec: GridSpec):
        self.spec = spec
        t, b, g, l = spec.n_steps, spec.n_buses, spec.n_generators, spec.n_lines
        pg, fl, an = t * g, t * l, t * b
        self.generation_slice = slice(0, pg)
        self.flow_slice = slice(pg, pg + fl)
        self.angle_slice = slice(pg + fl, pg + fl + an)
        self.n_variables = pg + fl + an
        self.balance_rows = np.arange(t * b).reshape(t, b)
        rows, cols, vals = [], [], []
        for ti in range(t):
            for gi in range(g):
                rows.append(ti * b + spec.generator_bus[gi])
                cols.append(ti * g + gi)
                vals.append(1.0)
            for li in range(l):
                fi = pg + ti * l + li
                fr, to = spec.line_from[li], spec.line_to[li]
                rows.extend([ti * b + fr, ti * b + to])
                cols.extend([fi, fi])
                vals.extend([-1.0, 1.0])
                # f = (theta_from - theta_to) / x.
                oi = t * b + ti * l + li
                rows.extend([oi, oi, oi])
                cols.extend([fi, pg + fl + ti * b + fr, pg + fl + ti * b + to])
                vals.extend([1.0, -1.0 / spec.line_reactance[li],
                             1.0 / spec.line_reactance[li]])
        self._a_eq = coo_matrix(
            (vals, (rows, cols)), shape=(t * b + t * l, self.n_variables)
        ).tocsr()
        self._lower = np.r_[np.zeros(pg), -np.tile(spec.line_capacity_mw, t),
                            np.full(an, -np.inf)]
        self._upper = np.r_[spec.generator_capacity_mw.ravel(),
                            np.tile(spec.line_capacity_mw, t), np.full(an, np.inf)]
        # The reference angle does not restrict physical power flows.
        reference_angles = pg + fl + np.arange(t) * b
        self._lower[reference_angles] = self._upper[reference_angles] = 0.0
        self._cost = np.r_[np.tile(spec.generator_cost_yuan_per_mwh, t) * spec.dt_hours,
                          np.zeros(fl + an)]
        self._emissions = np.r_[
            np.tile(spec.generator_emissions_t_per_mwh, t) * spec.dt_hours,
            np.zeros(fl + an),
        ]
        # Keep resource costs separate: carbon payments are monetary transfers,
        # while their opportunity cost belongs in offers and therefore LMPs.
        self._dispatch_cost = self._cost + spec.carbon_price_yuan_per_t * self._emissions
        self._carbon_row = csr_matrix(self._emissions.reshape(1, -1))

    def _load(self, load_mw: np.ndarray) -> np.ndarray:
        load = np.asarray(load_mw, dtype=float)
        if load.shape != (self.spec.n_steps, self.spec.n_buses):
            raise ValueError("load_mw must have shape [n_steps, n_buses]")
        if not np.all(np.isfinite(load)):
            raise ValueError("load_mw must be finite")
        return load

    def linear_problem(
        self, load_mw: np.ndarray, carbon_budget_t: float | None = None
    ) -> dict[str, Any]:
        """LP matrices for solving or adding task assignment variables.

        Variables are ``[p[T,G], f[T,L], theta[T,B]]`` flattened in C order.
        Nodal rows use ``generation - outgoing_flow + incoming_flow = load``.
        Append ``-task_load_mw`` in the corresponding balance rows when adding
        task choices to a problem whose right-hand side contains background load.
        """
        load = self._load(load_mw)
        if carbon_budget_t is not None and not np.isfinite(carbon_budget_t):
            raise ValueError("carbon_budget_t must be finite or None")
        return {
            "c": self._dispatch_cost.copy(), "A_eq": self._a_eq.copy(),
            "b_eq": np.r_[load.ravel(), np.zeros(self.spec.n_steps * self.spec.n_lines)],
            "A_ub": self._carbon_row.copy() if carbon_budget_t is not None else None,
            "b_ub": np.array([carbon_budget_t]) if carbon_budget_t is not None else None,
            "lower": self._lower.copy(), "upper": self._upper.copy(),
            "bounds": np.column_stack([self._lower, self._upper]),
            "balance_rows": self.balance_rows.copy(),
            "generation_slice": self.generation_slice, "flow_slice": self.flow_slice,
            "angle_slice": self.angle_slice,
        }

    def _infeasible(self, status: str) -> ClearingResult:
        s = self.spec
        return ClearingResult(
            False, np.inf, np.inf, np.full((s.n_steps, s.n_buses), np.nan),
            np.nan, np.full((s.n_steps, s.n_generators), np.nan),
            np.full((s.n_steps, s.n_lines), np.nan), np.inf, status,
            s.carbon_price_yuan_per_t,
        )

    def solve(
        self, load_mw: np.ndarray, carbon_budget_t: float | None = None,
        objective: str = "cost",
    ) -> ClearingResult:
        """Clear fixed load without load shedding.

        For ``objective='emissions'`` economic prices are NaN: the objective's
        duals have emissions units rather than yuan. Physical generation cost
        is still reported, together with the minimum feasible emissions.
        """
        if objective not in ("cost", "emissions"):
            raise ValueError("objective must be 'cost' or 'emissions'")
        load = self._load(load_mw)
        problem = self.linear_problem(load, carbon_budget_t)
        solution = linprog(
            problem["c"] if objective == "cost" else self._emissions,
            A_eq=problem["A_eq"], b_eq=problem["b_eq"],
            A_ub=problem["A_ub"], b_ub=problem["b_ub"],
            bounds=problem["bounds"], method="highs",
        )
        if not solution.success:
            return self._infeasible(f"HiGHS {solution.status}: {solution.message}")
        s = self.spec
        p = solution.x[self.generation_slice].reshape(s.n_steps, s.n_generators)
        f = solution.x[self.flow_slice].reshape(s.n_steps, s.n_lines)
        errors = (self._a_eq @ solution.x - problem["b_eq"])[self.balance_rows]
        if objective == "cost":
            prices = solution.eqlin.marginals[self.balance_rows] / s.dt_hours
            carbon_price = (-float(solution.ineqlin.marginals[0])
                            if carbon_budget_t is not None else 0.0)
        else:
            prices = np.full((s.n_steps, s.n_buses), np.nan)
            carbon_price = np.nan
        return ClearingResult(
            True, float(self._cost @ solution.x), float(self._emissions @ solution.x),
            prices, carbon_price, p, f, float(np.max(np.abs(errors))),
            f"HiGHS optimal ({objective})",
            s.carbon_price_yuan_per_t,
        )

    def to_pypsa(self, load_mw: np.ndarray, carbon_budget_t: float | None = None):
        """Construct the same network, weights and emissions constraint in PyPSA."""
        import pandas as pd
        import pypsa

        load = self._load(load_mw)
        if carbon_budget_t is not None and not np.isfinite(carbon_budget_t):
            raise ValueError("carbon_budget_t must be finite or None")
        s = self.spec
        network = pypsa.Network()
        snapshots = pd.RangeIndex(s.n_steps, name="snapshot")
        network.set_snapshots(snapshots)
        network.snapshot_weightings = pd.DataFrame(
            float(s.dt_hours), index=snapshots,
            columns=["objective", "stores", "generators"],
        )
        network.add("Carrier", "AC")
        for bi, name in enumerate(s.bus_names):
            network.add("Bus", name, v_nom=110.0, carrier="AC")
            network.add("Load", f"load{bi}", bus=name,
                        p_set=pd.Series(load[:, bi], index=snapshots))
        for gi in range(s.n_generators):
            carrier = f"generator_carrier{gi}"
            network.add("Carrier", carrier,
                        co2_emissions=s.generator_emissions_t_per_mwh[gi])
            nominal = max(1.0, float(np.max(s.generator_capacity_mw[:, gi])))
            network.add(
                "Generator", f"gen{gi}", bus=s.bus_names[s.generator_bus[gi]],
                p_nom=nominal, carrier=carrier, efficiency=1.0,
                p_max_pu=pd.Series(s.generator_capacity_mw[:, gi] / nominal,
                                   index=snapshots),
                marginal_cost=(s.generator_cost_yuan_per_mwh[gi]
                               + s.carbon_price_yuan_per_t * s.generator_emissions_t_per_mwh[gi]),
            )
        for li in range(s.n_lines):
            network.add(
                "Line", f"line{li}", bus0=s.bus_names[s.line_from[li]],
                bus1=s.bus_names[s.line_to[li]],
                x=s.line_reactance[li] * 110.0**2, r=0.0,
                s_nom=s.line_capacity_mw[li], carrier="AC",
            )
        if carbon_budget_t is not None:
            network.add("GlobalConstraint", "carbon_budget", type="primary_energy",
                        carrier_attribute="co2_emissions", sense="<=",
                        constant=float(carbon_budget_t))
        return network

    def solve_pypsa_reference(
        self, load_mw: np.ndarray, carbon_budget_t: float | None = None
    ) -> ClearingResult:
        """Cost-clearing reference using PyPSA's network optimization formulation."""
        network = self.to_pypsa(load_mw, carbon_budget_t)
        status, condition = network.optimize(
            solver_name="highs", log_to_console=False, assign_all_duals=True,
        )
        if status != "ok" or condition != "optimal":
            return self._infeasible(f"PyPSA {status}: {condition}")
        s = self.spec
        p = network.generators_t.p.loc[:, [f"gen{i}" for i in range(s.n_generators)]].to_numpy()
        f = network.lines_t.p0.loc[:, [f"line{i}" for i in range(s.n_lines)]].to_numpy()
        nodal_generation = np.zeros((s.n_steps, s.n_buses))
        for gi, bi in enumerate(s.generator_bus):
            nodal_generation[:, bi] += p[:, gi]
        for li, (fr, to) in enumerate(zip(s.line_from, s.line_to)):
            nodal_generation[:, fr] -= f[:, li]
            nodal_generation[:, to] += f[:, li]
        carbon_price = (-float(network.global_constraints.loc["carbon_budget", "mu"])
                        if carbon_budget_t is not None else 0.0)
        return ClearingResult(
            True, float(np.sum(p * s.generator_cost_yuan_per_mwh) * s.dt_hours),
            float(np.sum(p * s.generator_emissions_t_per_mwh) * s.dt_hours),
            network.buses_t.marginal_price.loc[:, list(s.bus_names)].to_numpy(),
            carbon_price, p, f,
            float(np.max(np.abs(nodal_generation - self._load(load_mw)))),
            "PyPSA optimal (cost)",
            s.carbon_price_yuan_per_t,
        )
