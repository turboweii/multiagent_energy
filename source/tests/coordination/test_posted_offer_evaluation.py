"""Evaluation helpers smoke-test real TRAIN cases, never the formal test panel."""
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import evaluate_posted_offers as evaluation

from energy_net.coordination.cached_grid import MemoizedGridClearing
from energy_net.coordination.capped_market import legacy_capped_config
from energy_net.coordination.flex_market import FlexMarketEpisode
from energy_net.coordination.outside_bookings import OutsideBooking
from energy_net.coordination.posted_offer_baselines import FixedRawOffer, ValueShareGreedy, LearnedResponseGreedy
from energy_net.coordination.posted_offer_market import PostedOfferEpisode
from energy_net.coordination.preference_scenario import clone_with_preferences, decode_preference_episode_id
from energy_net.coordination.varied_scenario import VariedMarketConfig, make_varied_scenario


class MyopicBinaryUser:
    def predict(self, observation, deterministic=True):
        assert deterministic
        options = observation[6:76].reshape(10, 7)
        accepted = options[1, 0] > .5 and options[1, 1] > .5 and options[1, 2] + options[1, 3] < -1e-7
        return np.asarray(int(accepted)), None


@pytest.fixture(scope="module")
def factory():
    config = VariedMarketConfig()
    old_config = legacy_capped_config(config)
    @lru_cache(maxsize=4)
    def public(seed):
        return make_varied_scenario(seed, config)
    @lru_cache(maxsize=4)
    def grid(seed):
        return MemoizedGridClearing(public(seed).grid_spec, maxsize=512)
    @lru_cache(maxsize=8)
    def inputs(case):
        public_seed, private_seed = decode_preference_episode_id(case)
        assert 0 <= public_seed < 64  # All smoke inputs belong to training.
        scenario = clone_with_preferences(public(public_seed), private_seed)
        old = FlexMarketEpisode(replace(scenario, config=old_config), grid=grid(public_seed))
        while not old.done:
            old.step_user(0)
        return scenario, OutsideBooking.from_episode(old)
    def make(case):
        public_seed, _ = decode_preference_episode_id(case)
        scenario, booking = inputs(int(case))
        return PostedOfferEpisode(scenario, booking, grid=grid(public_seed), config=config)
    make.config = config
    return make


@pytest.fixture(scope="module")
def users():
    return {owner: MyopicBinaryUser() for owner in range(3)}


def test_panels_and_exit_template_have_distinct_declared_meanings(factory):
    assert len(evaluation.SELECT_CASES) == len(evaluation.CALIBRATION_CASES) == len(evaluation.TEST_CASES) == 128
    assert set(evaluation.SELECT_CASES).isdisjoint(evaluation.CALIBRATION_CASES)
    assert {decode_preference_episode_id(case)[0] for case in evaluation.SELECT_CASES} == set(range(64))
    assert {decode_preference_episode_id(case)[0] for case in evaluation.TEST_CASES} == set(range(70000, 70032))
    templates = evaluation.matrix_templates(factory.config)
    assert len(templates) == 204
    np.testing.assert_array_equal(templates[0]["matrix"], templates[1]["matrix"])
    assert not templates[0]["open"] and templates[1]["open"]
    assert isinstance(evaluation.matrix_controller(templates[0]), FixedRawOffer)
    assert not isinstance(evaluation.matrix_controller(templates[1]), FixedRawOffer)


@pytest.fixture(scope="module")
def small_selection(factory, users):
    templates = evaluation.matrix_templates(factory.config)
    cases = evaluation.preference_cases(range(1), 1, 110000)
    calibration = evaluation.preference_cases(range(1), 1, 120000)
    return evaluation.select_controls(factory, users, cases, calibration,
        raw_actions=(0, 26, 28), templates=[templates[i] for i in (0, 1, 178)],
        rhos=(.3, .6), seed=0)


def test_small_real_train_selection_fixes_one_parameter_per_family(small_selection):
    report, records, summaries, calibration, feedback, prior, online = small_selection
    assert report["selected"]["raw"]["parameters"]["action"] != 0
    for family in ("raw", "matrix", "value_share"):
        eligible = [row for row in summaries[family] if family != "raw" or row["parameters"]["action"] != 0]
        optimum = max(row[evaluation.OBJECTIVE] for row in eligible)
        assert report["selected"][family]["training_mean_M_yuan"] == pytest.approx(optimum)
        assert len({row["case_id"] for row in records[family]}) == 1
    assert report["all_selection_queries"]["day_case_rollouts"] == 9
    assert report["all_selection_queries"]["resolved_task_responses"] == 108
    assert len(calibration) == 1 and len(feedback) == 12
    assert sum(prior.sample_counts) == sum(bool(row["offered"]) for row in feedback)
    np.testing.assert_array_equal(prior.prior_weights, online.prior_weights)
    assert not prior.enable_history and online.enable_history


def test_public_offer_logs_do_not_credit_rejected_proposals(factory):
    class KeepUser:
        def predict(self, observation, deterministic=True):
            return np.asarray(0), None
    users = {owner: KeepUser() for owner in range(3)}
    cases = evaluation.preference_cases(range(1), 1, 130000)
    rows, offers = evaluation.run_policy(factory, users, cases, "declined_train_smoke",
        controller=ValueShareGreedy(.3), record_offers=True)
    assert rows[0]["actual_G_yuan"] == rows[0]["actual_R_yuan"] == rows[0]["actual_M_yuan"] == 0
    assert all(row["realised_G_yuan"] == row["realised_R_yuan"] == row["realised_M_yuan"] == 0 for row in offers)
    assert any(row["offered_G_yuan"] > 0 for row in offers)


def test_actual_private_utility_diagnostic_stays_outside_public_offer_log(factory):
    class KeepUser:
        def predict(self, observation, deterministic=True):
            return np.asarray(0), None
    users = {owner: KeepUser() for owner in range(3)}
    cases = evaluation.preference_cases(range(1), 1, 135000)
    diagnostics = []
    rows, offers = evaluation.run_policy(factory, users, cases, "private_train_smoke",
        controller=ValueShareGreedy(.6), record_offers=True, private_diagnostics=diagnostics)
    assert diagnostics and all(row["private_evaluation_only"] for row in diagnostics)
    assert len(diagnostics) == sum(bool(row["offered"]) for row in offers)
    assert any(row["positive_utility_rejection"] for row in diagnostics)
    assert all(not row["accepted"] for row in diagnostics)
    assert not any("prospective_user_gain_yuan" in row or "private_evaluation_only" in row for row in offers)
    summary = evaluation.summarise_private_responses(diagnostics, ["private_train_smoke", "empty_exit"])
    assert summary[0]["positive_utility_rejections"] == summary[0]["positive_utility_offers"]
    assert summary[1]["actual_offers"] == 0
    assert rows[0]["positive_utility_rejections"] == summary[0]["positive_utility_rejections"]


def test_frozen_evaluation_does_not_fit_and_zeroes_only_explicit_history(factory, users, small_selection, monkeypatch):
    selection, _, _, _, _, prior, online = small_selection
    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen evaluation must not refit its response model")
    monkeypatch.setattr(LearnedResponseGreedy, "fit", forbidden)
    class Platform:
        def __init__(self):
            self.observations = []
        def predict(self, observation, deterministic=True):
            self.observations.append(observation.copy())
            legal = np.flatnonzero(observation[:, -1] > .5)
            positive = legal[legal > 0]
            return np.asarray(positive[0] if len(positive) else 0), None
    model, independent_no_history = Platform(), Platform()
    cases = evaluation.preference_cases(range(1), 1, 140000)
    rows, offers, summaries = evaluation.test_controls(factory, users, cases, selection, prior, online, model=model,
        extra_models=[("independent_no_history", independent_no_history, False)])
    assert len(rows) == len(summaries) == 9 and len(offers) == 108
    assert len(model.observations) == 24
    with_history, zeroed = model.observations[:12], model.observations[12:]
    assert any(np.any(value[:, 137:305]) for value in with_history[1:])
    assert all(not np.any(value[:, 137:305]) for value in zeroed)
    assert len(independent_no_history.observations) == 12
    assert all(not np.any(value[:, 137:305]) for value in independent_no_history.observations)
    for first, second in zip(with_history, zeroed):
        np.testing.assert_array_equal(first[:, -1], second[:, -1])
        np.testing.assert_array_equal(first[:, :137], second[:, :137])
        np.testing.assert_array_equal(first[:, 305:], second[:, 305:])


def test_accounting_rejects_inconsistent_realised_net(factory, users):
    cases = evaluation.preference_cases(range(1), 1, 150000)
    rows, _ = evaluation.run_policy(factory, users, cases, "train_smoke", controller=FixedRawOffer(0))
    broken = dict(rows[0], procurement_net_saving_yuan=1.)
    with pytest.raises(ValueError, match="accepted G minus paid R"):
        evaluation.enrich_day(broken, "wrong")
