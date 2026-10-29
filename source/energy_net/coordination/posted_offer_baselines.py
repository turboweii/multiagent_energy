"""Public-information controls for sequential, single posted contracts.

Route selection in the greedy controls is explicit. No controller calls a
user policy or substitutes another route for a static raw action. Response
probabilities use only already-published offers and their acceptance bits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .posted_offer_features import PostedOfferCatalogue


RATES = PostedOfferCatalogue.rates
RESPONSE_FEATURES = ("intercept", "rebate_yuan_per_gpuh_div_0.1",
                     "additional_delay_hours_div_3", "additional_migrated")


def nearest_rebate_rate(value):
    """Nearest disclosed rebate fraction; ties take the smaller fraction.

    This is not a repair for an unfunded quote: the selected rate is checked
    afterward, and an inadmissible route is skipped rather than repriced.
    """
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("A rebate target must be finite")
    distances = np.abs(np.asarray(RATES) - value)
    closest = float(distances.min())
    return min(rate for rate, distance in zip(RATES, distances)
               if distance <= closest + 1e-12)


def _money(preview, name, alias):
    return float(preview.get(name, preview.get(alias, 0.)))


def _legal(episode):
    return [(action, preview) for action in range(61)
            if (preview := episode.preview_offer(action)).get("admissible", False)]


def _routes(episode):
    """One zero-rebate public preview for each available physical route."""
    codec = PostedOfferCatalogue()
    job = episode.jobs[episode.position]
    for region in codec.regions:
        for delay in codec.delays_hours:
            preview = episode.preview_offer(codec.encode(region, delay, 0.))
            index = preview.get("option_index", preview.get("index"))
            if index is None or not preview.get("feasible", False):
                continue
            option = job.options[int(index)]
            outside = job.options[0]
            changed = any(not np.array_equal(getattr(option, name), getattr(outside, name))
                          for name in ("allocation_cu", "extra_energy_mwh", "transfer_gb"))
            if changed:
                yield region, delay, option, preview


def _best_positive(candidates, score):
    # Exit is an available zero-payoff action. In a tie its index zero wins.
    best_action, best_score = 0, 0.
    for action, preview in candidates:
        if not preview.get("admissible", False):
            continue
        value = float(score(action, preview))
        if not np.isfinite(value):
            raise ValueError("A public control score must be finite")
        tied = np.isclose(value, best_score, rtol=1e-12, atol=1e-9)
        if (value > best_score and not tied) or (tied and action < best_action):
            best_action, best_score = int(action), value
    return best_action


class RandomFundedOffer:
    """Uniform over public admissible actions, including keeping the booking."""

    def __init__(self, rng):
        self.rng = rng

    def choose(self, episode):
        actions = [action for action, _ in _legal(episode)]
        return int(self.rng.choice(actions)) if actions else 0


class FixedRawOffer:
    """Post this exact static action if available, otherwise retain the old job."""

    def __init__(self, action):
        self.action = PostedOfferCatalogue._index(action)

    def choose(self, episode):
        return self.action if episode.preview_offer(self.action).get("admissible", False) else 0


class FixedMatrixGreedy:
    """Fixed regional/time rebates and public maximum-net-surplus routing.

    Average the matrix over the candidate's actual GPU-hour allocation, then
    quantise once to the nearest offered rebate (tie: smaller rebate). The
    chosen matrix stays fixed across days; routing uses the current public M.
    """

    def __init__(self, matrix):
        values = np.asarray(matrix, dtype=float)
        if values.shape != (3, 4) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("A fixed rebate matrix must be a finite [3,4] array in [0,1]")
        self.matrix = values.copy()
        self.matrix.setflags(write=False)

    def choose(self, episode):
        codec = PostedOfferCatalogue()
        config = episode.config
        blocks = np.minimum(np.arange(config.n_steps) * 4 // config.n_steps, 3)
        quoted = self.matrix[:, blocks].T
        candidates = []
        for region, delay, option, _ in _routes(episode):
            work = float(np.sum(option.allocation_cu))
            target = float(np.sum(option.allocation_cu * quoted) / work)
            action = codec.encode(region, delay, nearest_rebate_rate(target))
            candidates.append((action, episode.preview_offer(action)))
        return _best_positive(candidates, lambda action, preview: _money(preview, "net_surplus_yuan", "M"))


class ValueShareGreedy:
    """Offer rho times public resource saving, quantised to the five rates."""

    def __init__(self, rho):
        self.rho = float(rho)
        if not np.isfinite(self.rho) or not 0 <= self.rho <= 1:
            raise ValueError("The proposed compensation share must lie in [0,1]")

    def choose(self, episode):
        codec = PostedOfferCatalogue()
        outside_invoice = float(episode.baseline_bills[episode.position])
        candidates = []
        for region, delay, _, zero_preview in _routes(episode):
            resource = _money(zero_preview, "resource_saving_yuan", "G")
            if resource < 0:
                continue
            # An already-free invoice cannot fund a percentage rebate.
            target = self.rho * resource / outside_invoice if outside_invoice > 0 else 0.
            action = codec.encode(region, delay, nearest_rebate_rate(target))
            candidates.append((action, episode.preview_offer(action)))
        return _best_positive(candidates, lambda action, preview: _money(preview, "net_surplus_yuan", "M"))


def _label(value):
    if value is None:
        return None
    if isinstance(value, str):
        values = {"true": 1., "false": 0., "1": 1., "0": 0.}
        return values.get(value.strip().lower())
    if isinstance(value, (bool, int, float, np.number)):
        number = float(value)
        if np.isfinite(number) and number in (0., 1.):
            return number
    return None


def _public_feedback(record):
    # Training code may pass either the public record itself or its envelope.
    public = record.get("public_feedback", record)
    if not isinstance(public, dict) or _label(public.get("offered")) != 1.:
        return None
    accepted = _label(public.get("accepted"))
    if accepted is None:
        return None  # Includes no-offer and not-yet-answered quotes.
    owner = int(public["user_id"])
    if not 0 <= owner < 3:
        raise ValueError("A public feedback owner must lie in [0,3)")
    features = np.asarray([1., float(public["rebate_yuan_per_gpuh"]) / .1,
                           float(public["additional_delay_hours"]) / 3.,
                           float(public["additional_migrated"])], dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("Public acceptance predictors must be finite")
    return owner, features, accepted


def _fit_logit(features, labels, prior, precision, max_iter):
    if not len(labels):
        return np.asarray(prior, dtype=float).copy()
    x, y, prior = np.asarray(features), np.asarray(labels), np.asarray(prior)

    def objective(weights):
        logits = x @ weights
        difference = weights - prior
        value = np.sum(np.logaddexp(0., logits) - y * logits) + .5 * precision * (difference @ difference)
        gradient = x.T @ (expit(logits) - y) + precision * difference
        return float(value), gradient

    result = minimize(objective, prior.copy(), method="L-BFGS-B", jac=True,
                      options=dict(maxiter=max_iter, ftol=1e-12, gtol=1e-8))
    if not result.success and np.max(np.abs(objective(result.x)[1])) > 1e-5:
        raise RuntimeError(f"Public-response logistic fit did not converge: {result.message}")
    if not np.isfinite(result.x).all():
        raise RuntimeError("Public-response logistic coefficients must be finite")
    return result.x


class LearnedResponseGreedy:
    """Public empirical acceptance probability times the offered net surplus.

    Train one regularised logistic model per owner on already-issued public
    feedback. A refusal is a binary response, including strategic refusal; it
    is never converted into an asserted private disutility inequality. With
    history enabled, the current owner's earlier actual replies update its
    fixed training prior by MAP. Priors themselves are never updated online.
    """

    def __init__(self, prior_weights=None, *, enable_history=False, l2=1.,
                 history_l2=1., max_iter=200, sample_counts=None):
        values = np.zeros((3, 4)) if prior_weights is None else np.asarray(prior_weights, dtype=float)
        if values.shape != (3, 4) or not np.isfinite(values).all():
            raise ValueError("Acceptance priors must be finite [3,4] coefficients")
        self.prior_weights = values.copy()
        self.prior_weights.setflags(write=False)
        self.enable_history = bool(enable_history)
        self.l2, self.history_l2 = float(l2), float(history_l2)
        if not np.isfinite([self.l2, self.history_l2]).all() or min(self.l2, self.history_l2) <= 0:
            raise ValueError("Logistic L2 precisions must be finite and positive")
        if int(max_iter) != max_iter or max_iter < 1:
            raise ValueError("Logistic optimisation needs a positive integer iteration limit")
        self.max_iter = int(max_iter)
        counts = np.zeros(3, dtype=int) if sample_counts is None else np.asarray(sample_counts)
        if counts.shape != (3,) or not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != counts.astype(int)):
            raise ValueError("Sample counts must contain three nonnegative integers")
        self.sample_counts = tuple(int(value) for value in counts)

    @classmethod
    def fit(cls, records, *, enable_history=False, l2=1., history_l2=1., max_iter=200):
        template = cls(enable_history=enable_history, l2=l2, history_l2=history_l2, max_iter=max_iter)
        features, labels = [[] for _ in range(3)], [[] for _ in range(3)]
        for record in records:
            public = _public_feedback(record)
            if public is not None:
                owner, x, y = public
                features[owner].append(x)
                labels[owner].append(y)
        weights = np.vstack([_fit_logit(features[owner], labels[owner], np.zeros(4),
                                        template.l2, template.max_iter) for owner in range(3)])
        return cls(weights, enable_history=enable_history, l2=l2, history_l2=history_l2,
                   max_iter=max_iter, sample_counts=[len(values) for values in labels])

    def _owner_weights(self, episode, owner):
        prior = self.prior_weights[owner]
        if not self.enable_history:
            return prior
        features, labels = [], []
        records = getattr(episode, "public_history", getattr(episode, "history", ()))
        for record in records:
            public = _public_feedback(record)
            if public is not None and public[0] == owner:
                features.append(public[1])
                labels.append(public[2])
        return _fit_logit(features, labels, prior, self.history_l2, self.max_iter)

    @staticmethod
    def _candidate_features(episode, preview):
        job = episode.jobs[episode.position]
        outside = job.options[0]
        option = job.options[int(preview.get("option_index", preview.get("index")))]
        delay = preview.get("additional_delay_hours", option.delay_hours - outside.delay_hours)
        migrated = preview.get("additional_migrated", int(option.migrated) - int(outside.migrated))
        return np.asarray([1., _money(preview, "additional_rebate_yuan", "R") / job.work_cu / .1,
                           float(delay) / 3., float(migrated)])

    def acceptance_probability(self, episode, action):
        if int(action) == 0:
            return 0.
        preview = episode.preview_offer(action)
        if not preview.get("admissible", False):
            return 0.
        owner = episode.jobs[episode.position].user_id
        weights = self._owner_weights(episode, owner)
        return float(expit(self._candidate_features(episode, preview) @ weights))

    def choose(self, episode):
        owner = episode.jobs[episode.position].user_id
        weights = self._owner_weights(episode, owner)
        def expected_net(action, preview):
            if action == 0:
                return 0.
            probability = float(expit(self._candidate_features(episode, preview) @ weights))
            return probability * _money(preview, "net_surplus_yuan", "M")
        return _best_positive(_legal(episode), expected_net)

    def to_dict(self):
        return dict(type="LearnedResponseGreedy", version=1,
                    response_features=list(RESPONSE_FEATURES), prior_weights=self.prior_weights.tolist(),
                    enable_history=self.enable_history, l2=self.l2, history_l2=self.history_l2,
                    max_iter=self.max_iter, sample_counts=list(self.sample_counts),
                    fitting_data="already-issued offers and public accepted bit only; no_offer excluded")

    def to_json(self, path=None):
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, values):
        if values.get("type") != "LearnedResponseGreedy" or values.get("version") != 1:
            raise ValueError("Unsupported public-response control format")
        if tuple(values.get("response_features", ())) != RESPONSE_FEATURES:
            raise ValueError("Public-response predictor units do not match")
        return cls(values["prior_weights"], enable_history=values["enable_history"],
                   l2=values["l2"], history_l2=values["history_l2"],
                   max_iter=values["max_iter"], sample_counts=values["sample_counts"])

    @classmethod
    def from_json(cls, source):
        if isinstance(source, Path) or not str(source).lstrip().startswith("{"):
            text = Path(source).read_text(encoding="utf-8")
        else:
            text = str(source)
        return cls.from_dict(json.loads(text))
