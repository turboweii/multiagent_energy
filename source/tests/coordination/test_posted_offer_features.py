"""Public pricing state and static contracts, with private user IR separate."""
from copy import copy
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from energy_net.coordination.amendment_market import AmendmentConfig
from energy_net.coordination.capped_market import capped_episode_factory
from energy_net.coordination.outside_bookings import OutsideBooking
from energy_net.coordination.posted_offer_features import (
    PostedOfferCatalogue, PostedOfferObserver, feature_names,
)
from energy_net.coordination.posted_offer_market import PostedOfferEpisode, decode_posted_action


@pytest.fixture(scope="module")
def factory():
    config = AmendmentConfig()
    old_factory = capped_episode_factory(config)

    @lru_cache(maxsize=8)
    def booking(seed):
        old = old_factory(seed)
        while not old.done:
            old.step_user(0)
        return OutsideBooking.from_episode(old)

    def make(seed=0):
        old = old_factory(seed)
        return PostedOfferEpisode(old.scenario, booking(seed), grid=old.grid, config=config)
    return make


def positive_private_offer(episode):
    """Test user applies its own IR; this is never a platform feature."""
    job = episode.jobs[episode.position]
    for action in range(1, 61):
        preview = episode.preview_offer(action)
        if not preview["admissible"]:
            continue
        option = job.options[preview["option_index"]]
        gain = preview["R"] - option.user_loss_yuan + episode.outside_losses[episode.position]
        if gain > 1e-6:
            return action
    return None


def advance_to_positive_offer(episode):
    for _ in episode.jobs:
        action = positive_private_offer(episode)
        if action is not None:
            return action
        episode.set_offer(0)
        episode.step_accept(0)
        if episode.done:
            break
    raise AssertionError("Fixture needs a funded offer acceptable to its private user")


def test_catalogue_static_semantics_match_core_without_substitution(factory):
    catalogue = PostedOfferCatalogue()
    first, second = factory(0), factory(1)
    assert len(catalogue.entries) == len(catalogue.labels) == catalogue.n_actions == 61
    assert catalogue.describe(0) is None
    for index in range(61):
        assert catalogue.describe(index) == decode_posted_action(index)
        assert catalogue.decode(first, index).shape == ()
        assert catalogue.decode(first, index).item() == catalogue.decode(second, index).item() == index
        if index:
            region, delay, rate = catalogue.describe(index)
            assert catalogue.encode(region, delay, rate) == index


@pytest.mark.parametrize("bad", [-1, 61, .5, np.nan, np.inf, [1, 2]])
def test_invalid_action_indices_are_rejected(factory, bad):
    with pytest.raises(ValueError):
        PostedOfferCatalogue().decode(factory(0), bad)


def test_existing_routes_and_outside_are_masked_before_publication(factory):
    episode, catalogue = factory(0), PostedOfferCatalogue()
    values, names = PostedOfferObserver(catalogue).raw(episode), feature_names()
    assert values.shape == (61, 331)
    assert len(names) == len(set(names)) == 331 and names[-1] == "offer_admissible"
    assert np.isfinite(values).all() and values[0, -1] == 1
    job = episode.jobs[episode.position]
    outside = job.options[0]
    for region in range(3):
        missing_delay = 3 if region == job.home_region else 0
        for rate in catalogue.rates:
            index = catalogue.encode(region, missing_delay, rate)
            assert values[index, names.index("candidate_route_exists")] == 0
            assert values[index, -1] == 0
    for rate in catalogue.rates:
        index = catalogue.encode(outside.region, outside.delay_hours, rate)
        assert values[index, names.index("candidate_route_exists")] == 1
        assert values[index, names.index("candidate_changed_execution")] == 0
        assert values[index, -1] == 0
    # Keeping the current booking has the same complete physical/public state.
    prefix = names.index("candidate_exit")
    np.testing.assert_array_equal(values[0, :prefix], values[1, :prefix])
    assert values[0, names.index(f"current_owner_{job.user_id}")] == 1
    assert values[0, names.index("day_work_thousand_gpuh")] > 0
    assert values[0, names.index("candidate_physically_feasible")] == 1
    assert values[0, names.index("candidate_peak_compute_fraction")] > 0


def test_every_candidate_financial_feature_matches_public_prebinding_preview(factory):
    episode = factory(0)
    values, names = PostedOfferObserver().raw(episode), feature_names()
    work = episode.jobs[episode.position].work_cu
    fields = {
        "candidate_resource_saving_yuan_per_gpuh": "G",
        "candidate_platform_bill_saving_yuan_per_gpuh": "a",
        "candidate_rebate_yuan_per_gpuh": "R",
        "candidate_net_surplus_yuan_per_gpuh": "M",
        "candidate_platform_gain_yuan_per_gpuh": "platform_gain_yuan",
        "candidate_buyer_gain_yuan_per_gpuh": "buyer_gain_yuan",
        "candidate_settlement_yuan_per_gpuh": "settlement_yuan",
    }
    for action in range(61):
        preview = episode.preview_offer(action)
        assert values[action, -1] == int(preview["admissible"])
        for feature, field in fields.items():
            assert values[action, names.index(feature)] == pytest.approx(preview[field] / work, abs=1e-7)
    selected, position, bills = list(episode.selected), episode.position, episode.bills.copy()
    PostedOfferObserver().raw(episode)
    assert episode.selected == selected and episode.position == position
    np.testing.assert_array_equal(episode.bills, bills)
    assert not episode.public_history and episode.current_offer is None


def test_random_user_pretraining_quotes_are_legal_not_greedy_replacements(factory):
    episode, catalogue = factory(0), PostedOfferCatalogue()
    rng = np.random.default_rng(5)
    samples = [int(catalogue.sample(episode, rng)) for _ in range(100)]
    assert len(set(samples)) > 2
    assert all(episode.preview_offer(action)["admissible"] for action in samples)
    assert all(catalogue.decode(episode, action).item() == action for action in samples)
    class NoFundedOffer:
        def preview_offer(self, action):
            return {"admissible": False}
    assert catalogue.sample(NoFundedOffer(), rng).item() == 0


def test_no_offer_history_and_no_history_ablation_preserve_other_state(factory):
    episode = factory(0)
    owner = episode.jobs[episode.position].user_id
    episode.set_offer(0)
    episode.step_accept(0)
    names = feature_names()
    full = PostedOfferObserver(enable_history=True).raw(episode)
    ablated = PostedOfferObserver(enable_history=False).raw(episode)
    history = np.array([name.startswith("history_") for name in names])
    np.testing.assert_array_equal(full[:, ~history], ablated[:, ~history])
    assert not np.any(ablated[:, history])
    assert full[0, names.index(f"history_owner{owner}_slot0_observed")] == 1
    assert full[0, names.index(f"history_owner{owner}_slot0_offered")] == 0
    assert full[0, names.index(f"history_owner{owner}_slot0_accepted")] == 0
    assert full[0, names.index("task_position_fraction")] == pytest.approx(1 / 12)


def test_private_ir_refusal_and_strategic_refusal_have_identical_public_state(factory):
    """User knows private IR; the platform sees neither its mask nor reason."""
    strategic = factory(0)
    action = advance_to_positive_offer(strategic)
    private_ir = factory(0)
    while private_ir.position < strategic.position:
        private_ir.set_offer(0)
        private_ir.step_accept(0)
    position = private_ir.position
    job = private_ir.jobs[position]
    options = tuple(replace(option, user_loss_yuan=(option.user_loss_yuan if i == 0 else 1e9))
                    for i, option in enumerate(job.options))
    jobs = list(private_ir.jobs)
    jobs[position] = replace(job, beta_yuan_per_cu_hour=1e6,
                             migration_loss_yuan_per_cu=1e6, options=options)
    private_ir.jobs = tuple(jobs)
    private_ir.scenario = replace(private_ir.scenario, jobs=private_ir.jobs)
    strategic.set_offer(action)
    private_ir.set_offer(action)
    s_obs = strategic.current_user_observation()[6:76].reshape(10, 7)
    p_obs = private_ir.current_user_observation()[6:76].reshape(10, 7)
    assert s_obs[1, 2] + s_obs[1, 3] < 0
    assert p_obs[1, 2] + p_obs[1, 3] > 0
    # The normal policy's private IR mask rules out accepting the bad offer;
    # a positive offer can still be declined to affect later prices.
    strategic.step_accept(0)
    private_ir.step_accept(0)
    assert strategic.public_history == private_ir.public_history
    np.testing.assert_array_equal(PostedOfferObserver().raw(strategic),
                                  PostedOfferObserver().raw(private_ir))


def test_public_features_do_not_read_preferences_losses_or_user_reactions(factory):
    episode = factory(0)
    expected = PostedOfferObserver().raw(episode)
    other = copy(episode)
    other.jobs = tuple(replace(job, beta_yuan_per_cu_hour=777., migration_loss_yuan_per_cu=999.,
        options=tuple(replace(option, user_loss_yuan=1e12) for option in job.options))
        for job in episode.jobs)
    other.scenario = replace(episode.scenario, jobs=other.jobs)
    other.outside_losses = np.full(len(other.jobs), -1e15)
    def forbidden(*args, **kwargs):
        raise AssertionError("No user/private reaction query belongs in platform state")
    other.current_user_observation = forbidden
    other.best_response_action = forbidden
    other.outcome = forbidden
    np.testing.assert_array_equal(PostedOfferObserver().raw(other), expected)


def test_current_grid_and_history_update_after_acceptance_without_stale_cache(factory):
    episode, observer = factory(0), PostedOfferObserver()
    action = advance_to_positive_offer(episode)
    before = observer.raw(episode)
    position = episode.position
    owner = episode.jobs[position].user_id
    episode.set_offer(action)
    _, _, info = episode.step_accept(1)
    assert info["amendment_accepted"]
    current = observer.raw(episode)
    names = feature_names()
    assert not np.array_equal(before, current)
    assert current[0, names.index("current_grid_emissions_thousand_t")] == pytest.approx(
        episode.current_clearing.emissions_t / 1000, abs=1e-7)
    assert current[0, names.index(f"history_owner{owner}_slot0_accepted")] == 1
    assert current[0, names.index("task_position_fraction")] == pytest.approx((position + 1) / 12)
    # Same public seed/position but a different accepted invoice changes R.
    changed = factory(0)
    raw = observer.raw(changed)
    changed.baseline_bills *= .9
    fresh = observer.raw(changed)
    assert np.any(raw[:, names.index("candidate_rebate_yuan_per_gpuh")]
                  != fresh[:, names.index("candidate_rebate_yuan_per_gpuh")])


def test_normalization_only_uses_given_training_states_and_keeps_history_units(factory):
    observer, seen = PostedOfferObserver(), []
    def training(seed):
        seen.append(seed)
        return factory(seed)
    mean, scale = observer.fit_normalization(training, [0, 1])
    assert seen == [0, 1]
    assert mean.shape == scale.shape == (330,)
    assert np.isfinite(mean).all() and np.all(scale > 0)
    for index, name in enumerate(feature_names()[:-1]):
        if name.startswith("history_") or name == "task_position_fraction":
            assert mean[index] == 0 and scale[index] == 1
    np.testing.assert_array_equal(observer(factory(0), training=True), observer(factory(0), training=False))
