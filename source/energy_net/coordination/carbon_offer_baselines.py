"""Public-response carbon/cash controls for the existing posted-offer market.

The response model is fitted on actually published contracts and replies. A
candidate's carbon consequence is the change in total OPF generator emissions,
not an allocated data-centre footprint. Neither the funding mask, settlement,
user preferences nor cash rewards are changed by this controller.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from .posted_offer_baselines import LearnedResponseGreedy
from .posted_offer_features import PostedOfferCatalogue


CARBON_PROFIT_GRID = (0., .125, .25, .5, 1., 2., 4., 8., 16., 32., 64., 256.)
CARBON_LAMBDA_CANDIDATES = (*CARBON_PROFIT_GRID, "cash_limit")
PLATFORM_SHARE = .8
EMISSION_SCALE_T = 1.
PROFIT_SCALE_YUAN = 1000.


def _number(preview, name, alias):
    value = float(preview[name] if name in preview else preview[alias])
    if not np.isfinite(value):
        raise ValueError(f"Public offer field {name} must be finite")
    return value


def candidate_grid_emission_reduction_t(episode, preview):
    """Current total generator emissions minus this physical candidate's total.

    Both clearings are public results of the existing OPF. The current clearing
    already includes earlier accepted offers, so this is the task's actual
    incremental consequence if accepted at the present negotiation state.
    """
    if not preview.get("feasible", False) or preview.get("clearing") is None:
        raise ValueError("An emissions consequence needs a feasible OPF preview")
    reduction = float(episode.current_clearing.emissions_t
                      - preview["clearing"].emissions_t)
    if not np.isfinite(reduction):
        raise ValueError("Public OPF emissions must be finite")
    return reduction


class CarbonResponseGreedy:
    """Choose max p(accept) * (delta grid t + lambda * .8 * (G-R)/1000).

    ``response_model`` is an existing public-only LearnedResponseGreedy prior,
    including its optional within-day MAP update. All 61 static actions keep
    their original meaning and admissibility. Exit has exactly zero value.

    ``cash_limit=True``, ``lambda_profit='cash_limit'`` or positive infinity
    select the limiting expected-cash control. Exact expected-cash ties use
    expected grid reduction, then the lower action, as in the lambda -> inf
    limit of the finite objective. JSON represents this with a flag and null,
    never a non-standard Infinity literal. This class only chooses a quote;
    rejection still produces zero realised cash and zero realised reduction.
    """

    def __init__(self, response_model=None, lambda_profit=0., *, cash_limit=False):
        if response_model is None:
            response_model = LearnedResponseGreedy()
        if not isinstance(response_model, LearnedResponseGreedy):
            raise TypeError("The response model must be a public LearnedResponseGreedy")
        requested_limit = isinstance(lambda_profit, str) and lambda_profit == "cash_limit"
        if requested_limit:
            value = np.inf
        elif lambda_profit is None and cash_limit:
            value = np.inf
        else:
            value = float(lambda_profit)
        if np.isnan(value) or value < 0:
            raise ValueError("The profit weight must be nonnegative or the cash limit")
        self.response_model = response_model
        self.cash_limit = bool(cash_limit or np.isposinf(value))
        self.lambda_profit = None if self.cash_limit else value

    @classmethod
    def fit(cls, records, *, lambda_profit=0., cash_limit=False, **response_options):
        """Fit only the same issued public feedback as LearnedResponseGreedy."""
        response = LearnedResponseGreedy.fit(records, **response_options)
        return cls(response, lambda_profit, cash_limit=cash_limit)

    def prob(self, episode, action):
        """Public prior/MAP probability; no prospective user-policy query."""
        action = PostedOfferCatalogue._index(action)
        return self.response_model.acceptance_probability(episode, action)

    acceptance_probability = prob

    def choose(self, episode):
        owner = episode.jobs[episode.position].user_id
        # Fit at most once for this negotiation state. Both helpers below are
        # the existing model's public-contract feature/MAP implementations.
        weights = self.response_model._owner_weights(episode, owner)
        best_action = 0
        best_key = (0., 0., 0) if self.cash_limit else (0., 0)
        for action in range(1, PostedOfferCatalogue.n_actions):
            preview = episode.preview_offer(action)
            if not preview.get("admissible", False):
                continue
            probability = float(expit(self.response_model._candidate_features(episode, preview) @ weights))
            resource = _number(preview, "resource_saving_yuan", "G")
            rebate = _number(preview, "additional_rebate_yuan", "R")
            # Recompute the real net amount, rather than trusting an unrelated
            # predicted reward or using resource saving before compensation.
            expected_cash = probability * PLATFORM_SHARE * (resource - rebate) / PROFIT_SCALE_YUAN
            expected_reduction = probability * candidate_grid_emission_reduction_t(episode, preview) / EMISSION_SCALE_T
            if self.cash_limit:
                key = (expected_cash, expected_reduction, -action)
            else:
                score = expected_reduction + self.lambda_profit * expected_cash
                if not np.isfinite(score):
                    raise ValueError("The weighted public offer score must be finite")
                key = (score, -action)
            if key > best_key:
                best_action, best_key = action, key
        return int(best_action)

    def to_dict(self):
        return dict(type="CarbonResponseGreedy", version=1,
                    lambda_profit=self.lambda_profit, cash_limit=self.cash_limit,
                    emission_scale_t=EMISSION_SCALE_T,
                    profit_scale_yuan=PROFIT_SCALE_YUAN, platform_share=PLATFORM_SHARE,
                    response_model=self.response_model.to_dict(),
                    carbon_metric="current OPF total generator emissions minus candidate OPF total generator emissions")

    def to_json(self, path=None):
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, values):
        if values.get("type") != "CarbonResponseGreedy" or values.get("version") != 1:
            raise ValueError("Unsupported carbon-response control format")
        expected = dict(emission_scale_t=EMISSION_SCALE_T,
                        profit_scale_yuan=PROFIT_SCALE_YUAN, platform_share=PLATFORM_SHARE)
        if any(values.get(name) != value for name, value in expected.items()):
            raise ValueError("Carbon/cash objective units do not match")
        cash_limit, weight = values.get("cash_limit"), values.get("lambda_profit")
        if not isinstance(cash_limit, bool) or (cash_limit and weight is not None):
            raise ValueError("The cash limit must use a flag and null profit weight")
        if not cash_limit and (weight is None or not np.isfinite(float(weight))):
            raise ValueError("A finite-weight control needs a finite profit weight")
        return cls(LearnedResponseGreedy.from_dict(values["response_model"]),
                   weight, cash_limit=cash_limit)

    @classmethod
    def from_json(cls, source):
        if isinstance(source, Path) or not str(source).lstrip().startswith("{"):
            text = Path(source).read_text(encoding="utf-8")
        else:
            text = str(source)
        return cls.from_dict(json.loads(text))


def select_carbon_candidate(rows, profit_reference_yuan, profit_fraction=.95):
    """Select by realised TRAIN means, keeping the declared cash floor fixed.

    Each row must contain ``mean_platform_gain_yuan`` and
    ``mean_grid_emissions_reduction_t`` in yuan/day and t/day. Additional row
    fields (lambda, controller JSON, query counts) pass through unchanged.
    Feasible candidates maximise mean actual grid reduction, then cash, then
    lower input index. If none meets the exact floor, return ``feasible=False``
    and the maximum-cash candidate (ties: larger grid reduction, lower index),
    together with the actual shortfall. No threshold or outcome is substituted.
    """
    reference, fraction = float(profit_reference_yuan), float(profit_fraction)
    if not np.isfinite(reference) or reference < 0:
        raise ValueError("The incremental-profit reference must be finite and nonnegative")
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("The retained-profit fraction must lie in [0,1]")
    candidates = [dict(row) for row in rows]
    if not candidates:
        raise ValueError("Carbon selection needs at least one training candidate")
    profits, reductions = [], []
    for row in candidates:
        profit = float(row["mean_platform_gain_yuan"])
        reduction = float(row["mean_grid_emissions_reduction_t"])
        if not np.isfinite([profit, reduction]).all():
            raise ValueError("Realised training means must be finite")
        profits.append(profit)
        reductions.append(reduction)
    required = fraction * reference
    feasible = [index for index, profit in enumerate(profits) if profit >= required]
    if feasible:
        selected = max(feasible, key=lambda index: (reductions[index], profits[index], -index))
        reason = "maximum_actual_grid_reduction_subject_to_profit_floor"
    else:
        selected = max(range(len(candidates)), key=lambda index: (profits[index], reductions[index], -index))
        reason = "profit_floor_unattained_maximum_profit_then_grid_reduction"
    return dict(feasible=bool(feasible), selected_index=selected,
                selected=candidates[selected], required_profit_yuan=required,
                profit_reference_yuan=reference, profit_fraction=fraction,
                selected_mean_platform_gain_yuan=profits[selected],
                selected_mean_grid_emissions_reduction_t=reductions[selected],
                profit_shortfall_yuan=max(0., required - profits[selected]),
                feasible_candidate_count=len(feasible), candidate_count=len(candidates),
                selection_reason=reason)
