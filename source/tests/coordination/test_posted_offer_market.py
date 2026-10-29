"""Published single offers, private acceptance and the unchanged cash ledger."""
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from energy_net.coordination.amendment_market import AmendmentConfig
from energy_net.coordination.capped_market import capped_episode_factory
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue
from energy_net.coordination.outside_bookings import OutsideBooking
from energy_net.coordination.posted_offer_market import (
    N_POSTED_ACTIONS, OFFER_DELAYS_HOURS, OFFER_RATES,
    PostedOfferEpisode, decode_posted_action,
)


@lru_cache(maxsize=6)
def booked(seed=0, old_menu=0):
    old = capped_episode_factory()(seed)
    catalogue = FixedGpuTariffCatalogue(old.config)
    old.set_platform_action(catalogue.decode(old, old_menu))
    old.play()  # Private local-choice fixture, not an evaluated learned policy.
    return OutsideBooking.from_episode(old)


def episode(seed=0, old_menu=0):
    booking = booked(seed, old_menu)
    return PostedOfferEpisode(booking.scenario, booking)


def action_for(region, delay, rate):
    return 1 + (region * len(OFFER_DELAYS_HOURS) + OFFER_DELAYS_HOURS.index(float(delay))) * len(OFFER_RATES) + OFFER_RATES.index(rate)


def find_offer(ep, willing=None, positive_rebate=False):
    """Choose a test fixture to exercise accounting, never a production policy."""
    for action in range(1, N_POSTED_ACTIONS):
        proposal = ep.preview_offer(action)
        if not proposal["admissible"]:
            continue
        if positive_rebate and proposal["R"] <= 0:
            continue
        option = ep.jobs[ep.position].options[proposal["index"]]
        gain = proposal["R"] - option.user_loss_yuan + ep.outside_losses[ep.position]
        if willing is not None and bool(gain > 1e-7) != willing:
            continue
        return action, proposal
    raise AssertionError("Declared test day has no requested funded-offer fixture")


def finish_without_more_offers(ep):
    while not ep.done:
        ep.set_offer(0)
        ep.step_accept(0)
    return ep.outcome()


def test_static_61_actions_and_absent_or_unchanged_routes_stay_unavailable():
    ep = episode()
    assert N_POSTED_ACTIONS == 61 and decode_posted_action(0) is None
    for region in range(3):
        for delay in OFFER_DELAYS_HOURS:
            for rate in OFFER_RATES:
                assert decode_posted_action(action_for(region, delay, rate)) == (region, delay, rate)
    job = ep.jobs[0]
    assert ep.preview_offer(action_for(job.home_region, 0., .005))["reason"] == "unchanged_execution"
    assert ep.preview_offer(action_for(job.home_region, 3., .005))["reason"] == "route_unavailable"
    remote = (job.home_region + 1) % 3
    assert ep.preview_offer(action_for(remote, 0., .005))["reason"] == "route_unavailable"
    assert len(job.options) == 10 and sum(ep.option_available[0]) == 9
    for invalid in (-1, 61, .1, np.nan, [1, 2]):
        with pytest.raises(ValueError):
            decode_posted_action(invalid)


def test_preview_is_public_and_does_not_read_private_preferences_or_losses():
    ep = episode()
    action, before = find_offer(ep)
    old = ep.jobs[0]
    options = tuple(replace(option, user_loss_yuan=1e9 + index)
                    for index, option in enumerate(old.options))
    ep.jobs = (replace(old, options=options, beta_yuan_per_cu_hour=999,
                       migration_loss_yuan_per_cu=999), *ep.jobs[1:])
    ep.scenario = replace(ep.scenario, jobs=ep.jobs)
    ep.outside_losses[0] = -123456
    after = ep.preview_offer(action)
    assert {key: value for key, value in before.items() if key != "clearing"} == {
        key: value for key, value in after.items() if key != "clearing"}
    assert not any(name in after for name in ("user_loss_yuan", "utility_gain_yuan", "beta", "kappa"))


def test_unfunded_proposal_is_rejected_before_publication_and_no_contract_changes():
    ep = episode()
    bad = next(action for action in range(1, 61)
               if ep.preview_offer(action)["reason"] == "negative_distributable_surplus")
    with pytest.raises(ValueError, match="negative_distributable_surplus"):
        ep.set_offer(bad)
    assert ep.current_offer is None and ep.public_history == []
    assert ep.selected == [0] * len(ep.jobs) and ep.amendment_ledger == []


def test_acceptance_binds_published_snapshot_without_rechecking_budget_or_opf(monkeypatch):
    ep = episode()
    action, proposal = find_offer(ep, willing=True, positive_rebate=True)
    old_bill = ep.baseline_bills[0]
    ep.set_offer(action)
    def no_new_screen(*args, **kwargs):
        raise AssertionError("Funding and physical promises were locked before acceptance")
    monkeypatch.setattr(ep, "preview_offer", no_new_screen)
    monkeypatch.setattr(ep.grid, "solve", no_new_screen)
    reward, done, info = ep.step_accept(1)
    assert not done and info["amendment_accepted"]
    assert ep.selected[0] == proposal["index"]
    assert ep.bills[0] == pytest.approx(old_bill - proposal["R"])
    assert reward == pytest.approx(info["utility_gain_yuan"] / ep.jobs[0].work_cu)
    assert ep.platform_step_gain == pytest.approx(.8 * proposal["M"] / 1000)
    assert ep.amendment_ledger[0]["settlement_yuan"] == pytest.approx(
        .8 * proposal["M"] - (proposal["a"] - proposal["R"]))
    assert info["requested"] == info["chosen"] == 1
    assert info["chosen_option_index"] == proposal["index"]


def test_user_sees_only_outside_and_the_unique_offer_in_92_features():
    ep = episode()
    action, proposal = find_offer(ep, willing=True, positive_rebate=True)
    ep.set_offer(action)
    obs = ep.current_user_observation()
    assert obs.shape == (92,) and obs.dtype == np.float32
    rows = obs[6:76].reshape(10, 7)
    np.testing.assert_array_equal(rows[:2, :2], np.ones((2, 2)))
    assert np.all(rows[2:] == 0)
    assert np.all(rows[0, 2:5] == 0)
    assert rows[1, 2] == pytest.approx(-proposal["R"] / ep.jobs[0].work_cu)
    assert ep.option_bill(ep.jobs[0], ep.jobs[0].options[proposal["index"]]) == pytest.approx(
        ep.baseline_bills[0] - proposal["R"])
    assert np.all(obs[-16:] == 0)


def test_negative_utility_and_voluntary_rejection_have_identical_public_feedback():
    forced, voluntary = episode(), episode()
    action, proposal = find_offer(forced, willing=False)
    assert proposal["R"] == 0
    forced.set_offer(action)
    voluntary.set_offer(action)
    reward, _, private = forced.step_accept(1)
    voluntary.step_accept(0)
    assert reward == 0 and private["reservation_reason"] == "user_reservation_utility"
    assert forced.public_history == voluntary.public_history
    public = forced.public_history[0]
    assert public["offered"] and public["accepted"] is False and public["response"] == "rejected"
    assert not any(any(fragment in name for fragment in ("loss", "utility", "beta", "kappa", "reason"))
                   for name in public)
    assert forced.selected[0] == 0 and forced.amendment_ledger == []


def test_user_can_reject_a_positive_offer_and_keep_old_contract_for_later_negotiation():
    ep = episode()
    action, _ = find_offer(ep, willing=True, positive_rebate=True)
    ep.set_offer(action)
    assert ep.best_response_action() == 1
    reward, _, info = ep.step_accept(0)
    assert reward == ep.platform_step_gain == 0
    assert not info["amendment_accepted"]
    assert ep.bills[0] == ep.baseline_bills[0]
    assert ep.losses[0] == ep.outside_losses[0]
    assert ep.current_clearing is ep.baseline_clearing
    assert ep.public_history[0]["accepted"] is False


def test_no_offer_is_not_a_rejected_offer_or_a_false_private_feedback():
    ep = episode(old_menu=7)
    original_loss = ep.outside_losses.sum()
    assert original_loss > 0
    result = finish_without_more_offers(ep)
    assert result.metrics["posted_offers"] == result.metrics["rejected_offers"] == 0
    assert result.metrics["no_offer_jobs"] == len(ep.jobs)
    assert all(row["accepted"] is None and row["response"] == "no_offer" for row in ep.public_history)
    assert result.metrics["user_loss_yuan"] == pytest.approx(original_loss)
    assert result.metrics["platform_gain_vs_outside_yuan"] == pytest.approx(0, abs=1e-6)
    ep.reset()
    assert ep.position == 0 and ep.public_history == [] and ep.current_offer is None
    np.testing.assert_array_equal(ep.losses, ep.outside_losses)


def test_own_four_record_history_is_public_and_other_users_feedback_is_excluded():
    ep = episode()
    owner = ep.current_user_id
    action, _ = find_offer(ep, willing=True, positive_rebate=True)
    ep.set_offer(action)
    ep.step_accept(1)
    record = ep.public_history[0]
    while ep.current_user_id != owner:
        assert np.all(ep.current_user_observation()[-16:] == 0)
        ep.set_offer(0)
        ep.step_accept(0)
    tail = ep.current_user_observation()[-16:].reshape(4, 4)
    np.testing.assert_allclose(tail[0], [record["rebate_yuan_per_gpuh"],
        record["additional_delay_hours"] / 3., record["additional_migrated"], 1.])
    assert np.all(tail[1:] == 0)


@pytest.mark.parametrize("seed", [0, 1, 3])
def test_real_whole_day_net_money_and_social_identity_and_common_cap(seed):
    ep = episode(seed)
    rewards, surplus = [], []
    while not ep.done:
        candidates = []
        for action in range(1, 61):
            proposal = ep.preview_offer(action)
            if not proposal["admissible"]:
                continue
            option = ep.jobs[ep.position].options[proposal["index"]]
            gain = proposal["R"] - option.user_loss_yuan + ep.outside_losses[ep.position]
            if gain >= -1e-7:
                candidates.append(proposal)
        action = max(candidates, key=lambda item: item["M"])["action"] if candidates else 0
        proposal = ep.preview_offer(action)
        ep.set_offer(action)
        ep.step_accept(1 if action else 0)
        rewards.append(ep.platform_step_gain)
        surplus.append(proposal["M"] if action else 0.)
    metrics = ep.outcome().metrics
    assert metrics["completion_rate"] == metrics["deadline_compliance"] == metrics["user_ir_rate"] == 1
    assert metrics["platform_gain_vs_outside_yuan"] == pytest.approx(sum(rewards) * 1000, abs=1e-6)
    assert metrics["procurement_net_saving_yuan"] == pytest.approx(sum(surplus), abs=1e-6)
    assert metrics["platform_gain_vs_outside_yuan"] == pytest.approx(.8 * sum(surplus), abs=1e-6)
    assert metrics["buyer_post_settlement_benefit_yuan"] == pytest.approx(.2 * sum(surplus), abs=1e-6)
    assert metrics["social_saving_vs_outside_yuan"] == pytest.approx(sum(surplus) + metrics["user_gain_yuan"], abs=1e-6)
    assert metrics["surplus_identity_error_yuan"] == pytest.approx(0, abs=1e-6)
    assert metrics["platform_profit_identity_error_yuan"] == pytest.approx(0, abs=1e-6)
    assert metrics["carbon_emissions_t"] <= ep.scenario.carbon_budget_t + 1e-6
    assert metrics["platform_floor_satisfied"]
    assert metrics["posted_offers"] == metrics["accepted_offers"]
    assert ep.outcome().metrics == metrics


def test_published_contract_cannot_be_repriced_and_existing_acceptances_are_never_undone():
    ep = episode()
    action, proposal = find_offer(ep, willing=True, positive_rebate=True)
    ep.set_offer(action)
    with pytest.raises(ValueError, match="Resolve the published offer"):
        ep.set_offer(0)
    ep.step_accept(1)
    accepted_index, invoice = ep.selected[0], ep.bills[0]
    result = finish_without_more_offers(ep)
    assert ep.selected[0] == accepted_index and ep.bills[0] == invoice
    assert result.metrics["accepted_offers"] == 1
    assert result.metrics["platform_gain_vs_outside_yuan"] == pytest.approx(proposal["platform_gain_yuan"], abs=1e-6)
    assert result.metrics["reservation_rejections"] == 0


def test_response_requires_a_published_offer_and_the_binary_action_space():
    ep = episode()
    with pytest.raises(ValueError, match="Publish one offer"):
        ep.step_accept(0)
    ep.set_offer(0)
    with pytest.raises(ValueError, match="accept"):
        ep.step_accept(2)
    ep.step_user(0)
    assert ep.position == 1


def test_route_memo_matches_uncached_five_rate_previews_and_reuses_only_public_work(monkeypatch):
    cached, reference = episode(), episode()
    action, _ = find_offer(cached, willing=True, positive_rebate=True)
    region, delay, _ = decode_posted_action(action)
    cached._route_preview_cache.clear()
    counters = dict(precheck=0, bill=0)
    real_check, real_bill = cached._precheck, cached._energy_bill
    def check(indices):
        counters["precheck"] += 1
        return real_check(indices)
    def bill(clearing, indices):
        counters["bill"] += 1
        return real_bill(clearing, indices)
    monkeypatch.setattr(cached, "_precheck", check)
    monkeypatch.setattr(cached, "_energy_bill", bill)
    results = []
    for rate in OFFER_RATES:
        reference._route_preview_cache.clear()
        expected = reference.preview_offer(action_for(region, delay, rate))
        actual = cached.preview_offer(action_for(region, delay, rate))
        assert {k: v for k, v in actual.items() if k != "clearing"} == {
            k: v for k, v in expected.items() if k != "clearing"}
        np.testing.assert_array_equal(actual["clearing"].generation_mw, expected["clearing"].generation_mw)
        np.testing.assert_array_equal(actual["clearing"].lmp_yuan_per_mwh, expected["clearing"].lmp_yuan_per_mwh)
        results.append(actual)
    assert counters == dict(precheck=1, bill=2)
    assert len({row["G"] for row in results}) == len({row["a"] for row in results}) == 1
    assert len({row["R"] for row in results}) == len({row["M"] for row in results}) == 5


def test_route_memo_recalculates_rebate_from_current_invoice_and_never_caches_ir():
    ep = episode()
    action, _ = find_offer(ep, willing=True, positive_rebate=True)
    before = ep.preview_offer(action)
    ep.baseline_bills[0] *= 1.01
    after = ep.preview_offer(action)
    assert after["G"] == before["G"] and after["a"] == before["a"]
    assert after["R"] == pytest.approx(before["R"] * 1.01)
    assert after["M"] == pytest.approx(after["G"] - after["R"])
    assert after["platform_gain_yuan"] == pytest.approx(.8 * after["M"])
    # A separate unchanged invoice fixture isolates the private IR decision.
    ep = episode()
    action, after = find_offer(ep, willing=True, positive_rebate=True)
    option_index = after["index"]
    old_job = ep.jobs[0]
    changed = tuple(replace(option, user_loss_yuan=1e9) if index == option_index else option
                    for index, option in enumerate(old_job.options))
    ep.jobs = (replace(old_job, options=changed), *ep.jobs[1:])
    ep.scenario = replace(ep.scenario, jobs=ep.jobs)
    assert ep.preview_offer(action)["admissible"]  # Public funding did not learn private IR.
    ep.set_offer(action)
    reward, _, info = ep.step_accept(1)
    assert reward == 0 and info["reservation_reason"] == "user_reservation_utility"


@pytest.mark.parametrize("change", ["position", "selected", "cap", "reset"])
def test_route_memo_state_changes_invalidate_public_route_work(monkeypatch, change):
    ep = episode()
    action, _ = find_offer(ep)
    ep._route_preview_cache.clear()
    checks = []
    real = ep._precheck
    def check(indices):
        checks.append(tuple(indices))
        return real(indices)
    monkeypatch.setattr(ep, "_precheck", check)
    ep.preview_offer(action)
    assert len(checks) == 1
    if change == "position":
        ep.position += 1
        # Find a changed route for the next job; probing fills the new entry.
        for candidate in range(1, 61):
            if ep.preview_offer(candidate)["changed"]:
                break
    elif change == "selected":
        ep.selected[1] = 1
        ep.preview_offer(action)
    elif change == "cap":
        ep.scenario = replace(ep.scenario, carbon_budget_t=ep.scenario.carbon_budget_t + 1.)
        ep.preview_offer(action)
    else:
        ep.reset()
        ep.preview_offer(action)
    assert len(checks) >= 2


def test_changed_carbon_allowance_does_not_reuse_an_old_opf_result():
    ep = episode()
    action, before = find_offer(ep)
    assert before["feasible"]
    ep.scenario = replace(ep.scenario, carbon_budget_t=0.)
    after = ep.preview_offer(action)
    assert not after["feasible"] and after["reason"] == "energy_feasibility"


def test_acceptance_and_rejection_clear_route_memo_without_changing_binding_offer():
    for accept in (0, 1):
        ep = episode()
        action, proposal = find_offer(ep, willing=True, positive_rebate=True)
        ep.set_offer(action)
        region, delay, _ = decode_posted_action(action)
        for rate in OFFER_RATES:
            ep.preview_offer(action_for(region, delay, rate))
        assert ep._route_preview_cache
        assert ep.current_offer["R"] == proposal["R"]
        ep.step_accept(accept)
        assert ep._route_preview_cache == {}
        assert ep.platform_step_gain == pytest.approx(proposal["platform_gain_yuan"] / 1000 if accept else 0.)
