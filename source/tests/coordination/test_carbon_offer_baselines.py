"""Carbon rules use public OPF effects and already-issued acceptance feedback."""
from copy import deepcopy
from types import SimpleNamespace

import json
import numpy as np
import pytest

from energy_net.coordination.amendment_market import AmendmentConfig
from energy_net.coordination.capped_market import capped_episode_factory
from energy_net.coordination.carbon_offer_baselines import (
    CARBON_LAMBDA_CANDIDATES, CARBON_PROFIT_GRID, CarbonResponseGreedy,
    candidate_grid_emission_reduction_t, select_carbon_candidate,
)
from energy_net.coordination.outside_bookings import OutsideBooking
from energy_net.coordination.posted_offer_baselines import LearnedResponseGreedy
from energy_net.coordination.posted_offer_market import PostedOfferEpisode


class PublicEpisode:
    """Only the public interface needed by the wrapped response estimator."""

    def __init__(self, candidates):
        self.jobs = [SimpleNamespace(user_id=0, work_cu=10000., options=(
            SimpleNamespace(delay_hours=0., migrated=False),
            SimpleNamespace(delay_hours=1., migrated=True)))]
        self.position, self.public_history = 0, []
        self.current_clearing = SimpleNamespace(emissions_t=100.)
        self.candidates = candidates

    def preview_offer(self, action):
        if action == 0:
            return dict(admissible=True, feasible=True, clearing=self.current_clearing)
        if action not in self.candidates:
            return dict(admissible=False, feasible=False)
        values = self.candidates[action]
        resource, rebate, reduction = values[:3]
        admissible = values[3] if len(values) > 3 else resource - rebate >= 0.
        return dict(admissible=admissible, feasible=True, option_index=1,
                    G=resource, R=rebate, M=999999.,
                    additional_delay_hours=1., additional_migrated=1.,
                    clearing=SimpleNamespace(emissions_t=100. - reduction))

    @property
    def user_policy(self):
        raise AssertionError("A public rule cannot query a user policy")

    @property
    def beta(self):
        raise AssertionError("A public rule cannot read private preferences")

    def current_user_observation(self):
        raise AssertionError("User utility observations cannot enter this rule")

    def best_response_action(self):
        raise AssertionError("A public rule cannot simulate alternative user responses")


def feedback(owner=0, accepted=False):
    return dict(user_id=owner, offered=True, accepted=accepted,
                rebate_yuan_per_gpuh=.05, additional_delay_hours=1.,
                additional_migrated=1.)


def test_declared_grid_contains_all_finite_weights_and_serialisable_cash_limit():
    assert CARBON_PROFIT_GRID == (0., .125, .25, .5, 1., 2., 4., 8., 16., 32., 64., 256.)
    assert CARBON_LAMBDA_CANDIDATES == (*CARBON_PROFIT_GRID, "cash_limit")
    json.dumps(CARBON_LAMBDA_CANDIDATES, allow_nan=False)


def test_carbon_profit_weight_changes_route_without_changing_funding_mask():
    episode = PublicEpisode({1: (1000., 0., -1.), 2: (100., 0., .5)})
    response = LearnedResponseGreedy()
    assert CarbonResponseGreedy(response, 0.).choose(episode) == 2
    assert CarbonResponseGreedy(response, 1.).choose(episode) == 2
    assert CarbonResponseGreedy(response, 4.).choose(episode) == 1
    assert CarbonResponseGreedy(response, cash_limit=True).choose(episode) == 1
    assert episode.preview_offer(1)["admissible"]


def test_objective_uses_actual_net_compensation_point_eight_share_and_units():
    # R must be deducted even though the preview's unrelated M is huge.
    episode = PublicEpisode({1: (100., 90., 0.), 2: (80., 1., 0.)})
    assert CarbonResponseGreedy(lambda_profit=1.).choose(episode) == 2
    # One tonne and 1000 yuan are the stated scales. With share .8, route2
    # beats route1 at lambda1; dropping .8 would make a lower-action tie.
    episode = PublicEpisode({1: (1000., 0., 0.), 2: (0., 0., 1.)})
    assert CarbonResponseGreedy(lambda_profit=1.).choose(episode) == 2
    assert CarbonResponseGreedy(lambda_profit=2.).choose(episode) == 1


def test_probability_weights_emissions_and_money_without_querying_a_user():
    episode = PublicEpisode({1: (1000., 0., 5.), 2: (1000., 500., 1.)})
    response = LearnedResponseGreedy([[-4., 12., 0., 0.]] * 3)
    control = CarbonResponseGreedy(response, 0.)
    assert control.prob(episode, 1) < .02
    assert control.acceptance_probability(episode, 2) > .88
    assert control.prob(episode, 0) == control.prob(episode, 60) == 0.
    assert control.choose(episode) == 2
    assert episode.public_history == []


def test_exit_has_zero_value_negative_emissions_are_kept_and_invalid_offers_skipped():
    episode = PublicEpisode({1: (1000., 0., -1.), 2: (10000., 0., 100., False)})
    assert CarbonResponseGreedy(lambda_profit=0.).choose(episode) == 0
    assert CarbonResponseGreedy(lambda_profit=4.).choose(episode) == 1
    zero = PublicEpisode({1: (0., 0., 0.)})
    assert CarbonResponseGreedy(lambda_profit=1.).choose(zero) == 0
    assert CarbonResponseGreedy(cash_limit=True).choose(zero) == 0
    equal = PublicEpisode({7: (1000., 0., 1.), 3: (1000., 0., 1.)})
    assert CarbonResponseGreedy(lambda_profit=1.).choose(equal) == 3


def test_cash_limit_ties_follow_finite_objective_limit():
    episode = PublicEpisode({1: (1000., 0., -1.), 2: (1000., 0., 1.)})
    for value in ("cash_limit", np.inf):
        assert CarbonResponseGreedy(lambda_profit=value).choose(episode) == 2
    # When cash is zero, a positive carbon gain still improves the limit.
    assert CarbonResponseGreedy(cash_limit=True).choose(PublicEpisode({1: (0., 0., 1.)})) == 1


def test_true_grid_effect_ignores_allocated_footprint_and_uses_current_prefix():
    episode = PublicEpisode({1: (10., 0., 2.)})
    preview = episode.preview_offer(1)
    preview["dc_footprint_reduction_t"] = -1000.
    assert candidate_grid_emission_reduction_t(episode, preview) == 2.
    episode.current_clearing.emissions_t = 97.
    assert candidate_grid_emission_reduction_t(episode, preview) == -1.
    with pytest.raises(ValueError):
        candidate_grid_emission_reduction_t(episode, dict(feasible=False))


def test_fit_and_online_history_wrap_the_existing_public_model_without_mutating_prior(monkeypatch):
    records = [feedback(accepted=value) for value in [True, False, False, True]]
    records += [dict(offered=False, accepted=None, beta=object()),
                dict(offered=True, accepted=None, user_id=0)]
    control = CarbonResponseGreedy.fit(records, lambda_profit=.25, enable_history=True)
    assert control.response_model.sample_counts == (4, 0, 0)
    episode = PublicEpisode({1: (1000., 0., 1.), 2: (1000., 50., 1.)})
    initial = control.prob(episode, 2)
    prior = control.response_model.prior_weights.copy()
    episode.public_history = [feedback(owner=1)] * 4 + [dict(offered=False, accepted=None)]
    assert control.prob(episode, 2) == initial
    episode.public_history += [dict(feedback(), beta=object(), private_reason="IR")] * 4
    updated = control.prob(episode, 2)
    assert 0 < updated < initial
    episode.public_history[-1] = dict(feedback(), private_reason="strategic")
    assert control.prob(episode, 2) == updated
    original_fit, calls = control.response_model._owner_weights, []
    def count(episode, owner):
        calls.append(owner)
        return original_fit(episode, owner)
    monkeypatch.setattr(control.response_model, "_owner_weights", count)
    control.choose(episode)
    assert calls == [0]
    np.testing.assert_array_equal(control.response_model.prior_weights, prior)
    episode.public_history = []
    assert control.prob(episode, 2) == initial


@pytest.mark.parametrize("weight", [0., .125, 256., "cash_limit", np.inf])
def test_json_roundtrip_preserves_history_probability_and_choice(tmp_path, weight):
    response = LearnedResponseGreedy([[-4., 12., 0., 0.]] * 3,
                                     enable_history=True, sample_counts=(10, 11, 12))
    control = CarbonResponseGreedy(response, weight)
    episode = PublicEpisode({1: (1000., 0., 1.), 2: (1000., 500., 1.)})
    episode.public_history = [feedback()]
    text = control.to_json(tmp_path / "carbon_rule.json")
    assert "Infinity" not in text and "NaN" not in text
    values = json.loads(text)
    if control.cash_limit:
        assert values["cash_limit"] is True and values["lambda_profit"] is None
    for restored in [CarbonResponseGreedy.from_dict(values),
                     CarbonResponseGreedy.from_json(text),
                     CarbonResponseGreedy.from_json(tmp_path / "carbon_rule.json")]:
        assert restored.choose(episode) == control.choose(episode)
        assert restored.prob(episode, 2) == control.prob(episode, 2)
        assert restored.to_dict() == control.to_dict()


def test_real_episode_can_increase_emissions_for_profit_after_prior_carbon_savings():
    config = AmendmentConfig()
    old = capped_episode_factory(config)(0)
    while not old.done:
        old.step_user(0)
    episode = PostedOfferEpisode(old.scenario, OutsideBooking.from_episode(old),
                                 grid=old.grid, config=config)
    # A real accepted prefix releases carbon headroom. All other prior jobs
    # retain their original booking. No alternate user-policy query is used.
    for position in range(10):
        episode.set_offer(52 if position == 4 else 0)
        _, _, info = episode.step_accept(1)
        if position == 4:
            assert info["accepted"]
    response = LearnedResponseGreedy([[-4., 12., 0., 0.]] * 3, enable_history=True)
    carbon, cash = CarbonResponseGreedy(response, 0.), CarbonResponseGreedy(response, 256.)
    assert carbon.choose(episode) == 0
    action = cash.choose(episode)
    preview = episode.preview_offer(action)
    assert action != 0 and preview["admissible"]
    assert candidate_grid_emission_reduction_t(episode, preview) < -1e-6
    assert preview["clearing"].emissions_t <= episode.scenario.carbon_budget_t + 1e-7
    selected, bills = list(episode.selected), episode.bills.copy()
    clearing, ledger = episode.current_clearing, deepcopy(episode.amendment_ledger)
    episode.set_offer(action)
    # Publishing a profitable forecast does not execute it or book its cash.
    assert episode.current_clearing is clearing
    assert episode.amendment_ledger == ledger
    _, _, info = episode.step_accept(0)
    assert info["rejected"] and info["platform_step_gain"] == 0.
    assert info["additional_rebate_yuan"] == info["utility_gain_yuan"] == 0.
    assert episode.current_clearing is clearing and episode.selected == selected
    np.testing.assert_array_equal(episode.bills, bills)
    assert episode.amendment_ledger == ledger


def test_train_selection_maximises_carbon_subject_to_unchanged_profit_floor():
    rows = [dict(mean_platform_gain_yuan=94., mean_grid_emissions_reduction_t=10.),
            dict(mean_platform_gain_yuan=95., mean_grid_emissions_reduction_t=3.),
            dict(mean_platform_gain_yuan=100., mean_grid_emissions_reduction_t=2.),
            dict(mean_platform_gain_yuan=97., mean_grid_emissions_reduction_t=3.)]
    original = deepcopy(rows)
    result = select_carbon_candidate(rows, 100.)
    assert result["feasible"] and result["required_profit_yuan"] == 95.
    assert result["selected_index"] == 3 and result["profit_shortfall_yuan"] == 0.
    assert result["feasible_candidate_count"] == 3 and result["candidate_count"] == 4
    assert rows == original


def test_unattained_floor_returns_maximum_profit_then_carbon_and_actual_gap():
    rows = [dict(mean_platform_gain_yuan=90., mean_grid_emissions_reduction_t=100.),
            dict(mean_platform_gain_yuan=94., mean_grid_emissions_reduction_t=2.),
            dict(mean_platform_gain_yuan=94., mean_grid_emissions_reduction_t=3.)]
    result = select_carbon_candidate(rows, 100.)
    assert result["feasible"] is False and result["selected_index"] == 2
    assert result["required_profit_yuan"] == 95. and result["profit_shortfall_yuan"] == 1.
    assert result["feasible_candidate_count"] == 0
    # Do not silently move the floor by even a small numerical tolerance.
    boundary = select_carbon_candidate([dict(mean_platform_gain_yuan=np.nextafter(95., 0.),
                                             mean_grid_emissions_reduction_t=1.)], 100.)
    assert boundary["feasible"] is False and boundary["profit_shortfall_yuan"] > 0.


@pytest.mark.parametrize("weight", [-1., np.nan, -np.inf])
def test_invalid_profit_weights_raise(weight):
    with pytest.raises(ValueError):
        CarbonResponseGreedy(lambda_profit=weight)


@pytest.mark.parametrize("rows,reference,fraction", [
    ([], 100., .95),
    ([dict(mean_platform_gain_yuan=np.nan, mean_grid_emissions_reduction_t=1.)], 100., .95),
    ([dict(mean_platform_gain_yuan=1., mean_grid_emissions_reduction_t=np.inf)], 100., .95),
    ([dict(mean_platform_gain_yuan=1., mean_grid_emissions_reduction_t=1.)], -1., .95),
    ([dict(mean_platform_gain_yuan=1., mean_grid_emissions_reduction_t=1.)], 100., 1.1),
])
def test_invalid_train_selection_inputs_raise(rows, reference, fraction):
    with pytest.raises(ValueError):
        select_carbon_candidate(rows, reference, fraction)


def test_serialisation_rejects_changed_objective_units_and_raw_infinity():
    values = CarbonResponseGreedy().to_dict()
    for update in [dict(platform_share=1.), dict(emission_scale_t=1000.),
                   dict(cash_limit=False, lambda_profit=np.inf),
                   dict(cash_limit=True, lambda_profit=np.inf)]:
        with pytest.raises(ValueError):
            CarbonResponseGreedy.from_dict(dict(values, **update))
