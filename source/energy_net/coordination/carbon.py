"""Consumption carbon accounting and local dispatch-emissions sensitivities.

Flow tracing allocates actual generator emissions to electricity consumers.
It is an average, physical-flow allocation, not a claim about emissions avoided
by moving a task. Actual avoided emissions must compare the two full dispatches.
All intensities here are tonnes/MWh; no carbon cash fee is created by accounting.
"""
from __future__ import annotations

import numpy as np


def nodal_carbon_intensity(spec, clearing, load_mw):
    """Trace the generation mix through directed realised flows, [T, bus].

    At each node, local generation plus incoming power is perfectly mixed.
    Its carbon content supplies local consumption and outgoing lines at the
    same intensity. The lossless accounting therefore conserves generator
    emissions: ``sum(load * intensity) == sum(generation * emission_factor)``
    separately in every period. Line direction follows the solved flow sign.

    An electrically empty node has zero assigned intensity. A source-free
    circulation, if supplied in an externally constructed result, also receives
    zero; the least-squares fallback must still satisfy all mixing balances.
    """
    if not clearing.feasible:
        raise ValueError("Carbon accounting requires a feasible dispatch")
    load = np.asarray(load_mw, dtype=float)
    generation = np.asarray(clearing.generation_mw, dtype=float)
    flows = np.asarray(clearing.flows_mw, dtype=float)
    shapes = ((load, (spec.n_steps, spec.n_buses)),
              (generation, (spec.n_steps, spec.n_generators)),
              (flows, (spec.n_steps, spec.n_lines)))
    if any(array.shape != shape or not np.all(np.isfinite(array))
           for array, shape in shapes):
        raise ValueError("Load, generation and flows need finite grid-shaped arrays")
    if np.any(load < -1e-8) or np.any(generation < -1e-8):
        raise ValueError("Flow tracing requires nonnegative demand and generation")
    load, generation = np.maximum(load, 0), np.maximum(generation, 0)
    intensity = np.zeros_like(load)
    factors = np.asarray(spec.generator_emissions_t_per_mwh)
    for time in range(spec.n_steps):
        local_power = np.bincount(spec.generator_bus, weights=generation[time],
                                  minlength=spec.n_buses)
        local_carbon = np.bincount(spec.generator_bus,
                                   weights=generation[time] * factors,
                                   minlength=spec.n_buses)
        incoming = np.zeros((spec.n_buses, spec.n_buses))  # [origin, destination]
        for line, (origin, destination) in enumerate(zip(spec.line_from, spec.line_to)):
            power = flows[time, line]
            if power >= 0:
                incoming[origin, destination] += power
            else:
                incoming[destination, origin] -= power
        supply = local_power + incoming.sum(axis=0)
        withdrawal = load[time] + incoming.sum(axis=1)
        if not np.allclose(supply, withdrawal, rtol=1e-8, atol=1e-6):
            raise ValueError("Dispatch does not balance power for carbon flow tracing")
        mixing = np.diag(supply) - incoming.T
        empty = supply <= 1e-12
        mixing[empty, empty] = 1.0
        try:
            carbon = np.linalg.solve(mixing, local_carbon)
        except np.linalg.LinAlgError:
            carbon = np.linalg.lstsq(mixing, local_carbon, rcond=None)[0]
        if (not np.allclose(mixing @ carbon, local_carbon, rtol=1e-8, atol=1e-7)
                or np.min(carbon) < -1e-8):
            raise ValueError("Dispatch has no nonnegative carbon-mixing solution")
        carbon = np.maximum(carbon, 0.0)
        if not np.isclose(load[time] @ carbon, local_carbon.sum(),
                          rtol=1e-8, atol=1e-6):
            raise ValueError("Consumption carbon does not conserve generation emissions")
        intensity[time] = carbon
    return intensity


def consumption_emissions_t(power_mw, intensity_t_per_mwh, dt_hours):
    """Electricity-consumption footprint, including any supplied idle/transfer load."""
    power = np.asarray(power_mw, dtype=float)
    intensity = np.asarray(intensity_t_per_mwh, dtype=float)
    if (power.shape != intensity.shape or not np.all(np.isfinite(power))
            or not np.all(np.isfinite(intensity)) or np.any(power < -1e-8)
            or np.any(intensity < -1e-8) or not np.isfinite(dt_hours)
            or dt_hours <= 0):
        raise ValueError("Need equal-shaped nonnegative power/intensity and positive duration")
    return float(np.sum(np.maximum(power, 0) * np.maximum(intensity, 0)) * dt_hours)


def reference_lme(grid, load_mw, *, perturbation_mw=0.001):
    """One-sided local marginal emissions, tonnes/MWh, without a hard carbon cap.

    Each node/period is perturbed independently and the grid is re-cleared.
    Generation-side external carbon prices remain in the grid's offers. The
    derivative is local to the current dispatch regime; it does not predict a
    large migration across generator/line-capacity boundaries. Negative marginal
    emissions are permitted and must not be replaced by average carbon intensity.
    This optional diagnostic needs T*B extra solves and is not a cash price.
    """
    load = np.asarray(load_mw, dtype=float)
    if (load.shape != (grid.spec.n_steps, grid.spec.n_buses)
            or not np.all(np.isfinite(load)) or np.any(load < 0)
            or not np.isfinite(perturbation_mw) or perturbation_mw <= 0):
        raise ValueError("Need grid-shaped nonnegative load and positive perturbation")
    baseline = grid.solve(load, None)
    if not baseline.feasible:
        raise ValueError("Marginal emissions require a feasible reference dispatch")
    result = np.empty_like(load)
    for index in np.ndindex(load.shape):
        changed = load.copy()
        changed[index] += perturbation_mw
        forward = grid.solve(changed, None)
        if forward.feasible:
            result[index] = ((forward.emissions_t - baseline.emissions_t)
                             / (perturbation_mw * grid.spec.dt_hours))
            continue
        if load[index] >= perturbation_mw:
            changed[index] = load[index] - perturbation_mw
            backward = grid.solve(changed, None)
            if backward.feasible:
                result[index] = ((baseline.emissions_t - backward.emissions_t)
                                 / (perturbation_mw * grid.spec.dt_hours))
                continue
        raise ValueError(f"No feasible marginal load perturbation at {index}")
    return result
