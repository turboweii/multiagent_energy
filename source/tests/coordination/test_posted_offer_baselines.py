"""Static and learned controls use public contracts, not private user queries."""
from copy import deepcopy
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest

from energy_net.coordination.amendment_market import AmendmentConfig
from energy_net.coordination.capped_market import capped_episode_factory
from energy_net.coordination.outside_bookings import OutsideBooking
from energy_net.coordination.posted_offer_baselines import (
    FixedMatrixGreedy, FixedRawOffer, LearnedResponseGreedy,
    RandomFundedOffer, ValueShareGreedy, nearest_rebate_rate,
)
from energy_net.coordination.posted_offer_features import PostedOfferCatalogue
from energy_net.coordination.posted_offer_market import PostedOfferEpisode


def option(region, delay, start):
    allocation = np.zeros((48, 3))
    allocation[start:start + 4, region] = 2500.
    return SimpleNamespace(region=region, delay_hours=delay, start_step=start,
        allocation_cu=allocation, extra_energy_mwh=np.zeros((48, 3)),
        transfer_gb=np.zeros((48, 3, 3)), migrated=region != 0)


class PublicEpisode:
    def __init__(self, gains=(10., 8.), bill=1000.):
        self.config = SimpleNamespace(n_steps=48, n_blocks=4, dt_hours=.5)
        self.jobs = [SimpleNamespace(user_id=0, work_cu=10000.,
                                   options=(option(0, 0, 8), option(1, 1, 10), option(2, 2, 12)))]
        self.position, self.baseline_bills = 0, np.asarray([bill])
        self.public_history = []
        self.gains, self.calls = gains, []

    def preview_offer(self, action):
        self.calls.append(action)
        decoded = PostedOfferCatalogue().describe(action)
        if decoded is None:
            return dict(admissible=True, feasible=True, M=0.)
        region, delay, rate = decoded
        match = {(1, 1): 1, (2, 2): 2}.get((region, delay))
        if match is None:
            return dict(admissible=False, feasible=False, option_index=None)
        gain = self.gains[match - 1]
        rebate = self.baseline_bills[0] * rate
        net = gain - rebate
        return dict(admissible=net >= 0, feasible=True, changed=True,
                    option_index=match, G=gain, R=rebate, M=net,
                    additional_delay_hours=delay, additional_migrated=1.)

    @property
    def user_policy(self):
        raise AssertionError("Public controls cannot query a user policy")

    def best_response_action(self):
        raise AssertionError("Public controls cannot simulate private responses")


def feedback(owner=0, rebate=.1, delay=1., moved=1., accepted=True, **extra):
    return dict(user_id=owner, offered=True, accepted=accepted,
                rebate_yuan_per_gpuh=rebate, additional_delay_hours=delay,
                additional_migrated=moved, **extra)


def test_nearest_rate_quantization_is_stable_and_ties_take_smaller_rebate():
    assert nearest_rebate_rate(.00375) == .0025
    assert nearest_rebate_rate(.0075) == .005
    assert nearest_rebate_rate(.015) == .01
    assert nearest_rebate_rate(.0085) == .01
    assert nearest_rebate_rate(.7) == .02
    with pytest.raises(ValueError):
        nearest_rebate_rate(np.nan)


def test_fixed_raw_offer_never_substitutes_route_or_rebate():
    episode, codec = PublicEpisode(), PostedOfferCatalogue()
    available = codec.encode(1, 1, .005)
    assert FixedRawOffer(available).choose(episode) == available
    assert FixedRawOffer(codec.encode(0, 3, .005)).choose(episode) == 0
    # A cheaper rebate on this route is funded, but the requested 2% is not.
    assert FixedRawOffer(codec.encode(1, 1, .02)).choose(episode) == 0


def test_random_control_samples_exactly_the_public_admissible_set():
    episode = PublicEpisode()
    control = RandomFundedOffer(np.random.default_rng(7))
    actions = [control.choose(episode) for _ in range(100)]
    assert len(set(actions)) > 3 and 0 in actions
    assert all(episode.preview_offer(action)["admissible"] for action in actions)


def test_fixed_matrix_uses_allocation_weights_then_one_quantized_candidate():
    episode, codec = PublicEpisode(), PostedOfferCatalogue()
    matrix = np.zeros((3, 4))
    matrix[1, 0], matrix[1, 1] = .005, .01
    matrix[2, :] = .02
    control = FixedMatrixGreedy(matrix)
    # Region1's task straddles two blocks equally: .0075 -> .005, M=5.
    # The fixed region2 rebate is unfunded and must not be silently reduced.
    chosen = control.choose(episode)
    assert chosen == codec.encode(1, 1, .005)
    assert codec.encode(2, 2, .02) in episode.calls
    assert codec.encode(2, 2, .005) not in episode.calls
    matrix[:, :] = 1.
    assert control.choose(episode) == chosen  # The fixed template was copied.


def test_value_share_quantization_does_not_repair_an_unfunded_rounding():
    codec = PostedOfferCatalogue()
    episode = PublicEpisode()
    assert ValueShareGreedy(.5).choose(episode) == codec.encode(1, 1, .005)
    assert ValueShareGreedy(.75).choose(episode) == codec.encode(1, 1, .005)
    unfunded_rounding = PublicEpisode(gains=(8.5, -1.))
    # rho=1 gives .0085 -> .01, which spends10 on G8.5; skip, not rate.005.
    assert ValueShareGreedy(1.).choose(unfunded_rounding) == 0
    free_invoice = PublicEpisode(bill=0.)
    assert ValueShareGreedy(.5).choose(free_invoice) == codec.encode(1, 1, 0.)


@pytest.mark.parametrize("control", [FixedMatrixGreedy(np.zeros((3, 4))), ValueShareGreedy(0.)])
def test_greedy_score_ties_use_lower_static_action(control):
    episode = PublicEpisode(gains=(10., 10.))
    assert control.choose(episode) == PostedOfferCatalogue().encode(1, 1, 0.)
    assert control.choose(PublicEpisode(gains=(0., 0.))) == 0


def test_logistic_fit_uses_only_issued_public_offers_and_returns_probabilities():
    records = []
    for owner in range(3):
        for repeat in range(4):
            for rebate in (0., .1, .2, .3):
                record = feedback(owner, rebate=rebate, delay=0., moved=0., accepted=rebate >= .2)
                record["private_user_loss"] = object()
                record["beta"] = object()
                records.append({"public_feedback": record})
    records.append(dict(offered=False, accepted=None, user_id=0))
    records.append(dict(offered=True, accepted=None, user_id=1))
    model = LearnedResponseGreedy.fit(records)
    assert model.sample_counts == (16, 16, 16)
    assert np.all(model.prior_weights[:, 1] > 0)
    assert np.isfinite(model.prior_weights).all()
    low = 1 / (1 + np.exp(-model.prior_weights[0, 0]))
    high = 1 / (1 + np.exp(-(model.prior_weights[0] @ [1, 3, 0, 0])))
    assert low < .3 < .7 < high
    empty = LearnedResponseGreedy.fit([dict(offered=False, accepted=False)])
    np.testing.assert_array_equal(empty.prior_weights, np.zeros((3, 4)))
    assert empty.sample_counts == (0, 0, 0)


def test_logistic_map_uses_own_actual_replies_once_and_never_mutates_priors():
    episode = PublicEpisode()
    control = LearnedResponseGreedy(enable_history=True)
    action = PostedOfferCatalogue().encode(1, 1, .005)
    original = control.prior_weights.copy()
    initial = control.acceptance_probability(episode, action)
    episode.public_history = [dict(user_id=0, offered=False, accepted=None)]
    assert control.acceptance_probability(episode, action) == initial
    episode.public_history = [feedback(owner=1, accepted=False)] * 4
    assert control.acceptance_probability(episode, action) == initial
    episode.public_history += [feedback(owner=0, rebate=.0005, accepted=False)] * 4
    first = control.acceptance_probability(episode, action)
    assert 0 < first < initial
    assert control.acceptance_probability(episode, action) == first
    np.testing.assert_array_equal(control.prior_weights, original)
    # A private IR refusal and a strategic refusal remain the same label.
    episode.public_history[-1]["private_reason"] = "IR"
    ir = control.acceptance_probability(episode, action)
    episode.public_history[-1]["private_reason"] = "strategic"
    assert control.acceptance_probability(episode, action) == ir
    episode.public_history = []
    assert control.acceptance_probability(episode, action) == initial


def test_learned_response_selects_probability_times_net_not_net_alone():
    episode = PublicEpisode(gains=(1000., 800.), bill=100000.)
    control = LearnedResponseGreedy([[-4., 12., 0., 0.]] * 3)
    codec = PostedOfferCatalogue()
    # Zero rebate maximises M, but predicted uptake is1.8%. A0.5% rebate
    # leaves M500 with88% uptake, beating both0.25% and the other route.
    assert control.choose(episode) == codec.encode(1, 1, .005)


def test_no_history_model_ignores_replies_and_json_roundtrip_preserves_choice(tmp_path):
    episode = PublicEpisode()
    model = LearnedResponseGreedy([[0., 20., -1., -1.]] * 3, enable_history=False)
    action = model.choose(episode)
    assert episode.preview_offer(action)["admissible"]
    episode.public_history = [feedback(accepted=False)] * 4
    assert model.choose(episode) == action
    restored = LearnedResponseGreedy.from_json(model.to_json())
    assert restored.choose(episode) == action
    path = tmp_path / "public_response.json"
    model.to_json(path)
    from_file = LearnedResponseGreedy.from_json(path)
    np.testing.assert_array_equal(from_file.prior_weights, model.prior_weights)
    assert from_file.acceptance_probability(episode, action) == model.acceptance_probability(episode, action)


@pytest.fixture(scope="module")
def real_episode_factory():
    config = AmendmentConfig()
    legacy = capped_episode_factory(config)
    @lru_cache(maxsize=4)
    def booking(seed):
        episode = legacy(seed)
        while not episode.done:
            episode.step_user(0)
        return OutsideBooking.from_episode(episode)
    def make(seed):
        case = legacy(seed)
        return PostedOfferEpisode(case.scenario, booking(seed), grid=case.grid, config=config)
    return make


def test_all_controls_return_real_funded_actions_without_public_state_mutation(real_episode_factory):
    controls = [RandomFundedOffer(np.random.default_rng(9)), FixedRawOffer(28),
                FixedMatrixGreedy(np.full((3, 4), .005)), ValueShareGreedy(.5),
                LearnedResponseGreedy(enable_history=True)]
    for seed in (0, 1, 4):
        for control in controls:
            episode = real_episode_factory(seed)
            selected, bills, history = list(episode.selected), episode.bills.copy(), deepcopy(episode.public_history)
            def forbidden(*args, **kwargs):
                raise AssertionError("No private user reaction is observable before posting")
            episode.current_user_observation = forbidden
            episode.best_response_action = forbidden
            action = control.choose(episode)
            assert isinstance(action, int) and 0 <= action < 61
            assert episode.preview_offer(action)["admissible"]
            assert episode.selected == selected and episode.position == 0
            np.testing.assert_array_equal(episode.bills, bills)
            assert episode.public_history == history and episode.current_offer is None


@pytest.mark.parametrize("constructor,args", [
    (FixedRawOffer, (61,)), (FixedMatrixGreedy, (np.zeros((3, 3)),)),
    (ValueShareGreedy, (-.1,)), (ValueShareGreedy, (1.1,)),
    (LearnedResponseGreedy, (np.zeros((2, 4)),)),
])
def test_invalid_control_parameters_raise(constructor, args):
    with pytest.raises(ValueError):
        constructor(*args)
