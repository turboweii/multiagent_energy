"""Evaluation contracts on tiny synthetic cases, never the held-out study seeds."""
from dataclasses import replace
from functools import lru_cache
import json

import numpy as np
import pandas as pd
import pytest

from energy_net.coordination.amendment_features import AmendmentTariffCatalogue, AmendmentTariffObserver
from energy_net.coordination.amendment_market import AmendmentConfig, amendment_episode_factory
from energy_net.coordination.capped_market import capped_episode_factory
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue
from energy_net.coordination.market import exact_user_policy
from energy_net.coordination.outside_bookings import OutsideBooking
from scripts import evaluate_amendments as evaluation


class PrivateCostNetworkFixture:
    """Predict-compatible deterministic fixture; no trained-effect claim."""
    def __init__(self):
        self.calls = 0

    def predict(self, observation, deterministic=True):
        assert deterministic
        self.calls += 1
        return np.asarray(exact_user_policy(observation)), None


class PlatformNetworkFixture:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = 0

    def predict(self, observation, deterministic=True):
        assert deterministic
        assert observation.shape[0] == 78
        self.calls += 1
        return np.asarray(next(self.actions)), None


def users():
    return {user: PrivateCostNetworkFixture() for user in range(3)}


@lru_cache(maxsize=5)
def outside(seed):
    assert 0 <= seed <= 4, "Do not consume any formal held-out study case in unit tests"
    old = capped_episode_factory()(seed)
    catalogue = FixedGpuTariffCatalogue(old.config)
    old.set_platform_action(catalogue.decode(old, 7))
    old.play()
    return OutsideBooking.from_episode(old)


def market(order="net"):
    config = AmendmentConfig(settlement_order=order)
    factory = amendment_episode_factory(
        lambda seed: replace(outside(seed).scenario, config=config),
        lambda seed: outside(seed).copy(), config, maxsize=5)
    catalogue = AmendmentTariffCatalogue(config)
    return factory, catalogue, AmendmentTariffObserver(catalogue)


def test_exit_uses_exact_old_contracts_without_calling_new_user_networks():
    factory, catalogue, observer = market()
    networks = users()
    rows, actions, contracts = evaluation.evaluate_policy(factory, catalogue, observer,
        networks, None, range(5), "net", "accepted_old_bookings", constant=0)
    assert sum(model.calls for model in networks.values()) == 0
    assert len(rows) == len(actions) == 5 and len(contracts) == 60
    for row, action in zip(rows, actions):
        old = outside(row["seed"])
        assert row["new_platform_profit_yuan"] == pytest.approx(old.metrics["platform_profit_yuan"])
        assert row["new_system_cost_yuan"] == pytest.approx(old.metrics["system_cost_yuan"])
        assert row["extra_cost_saving_yuan"] == pytest.approx(0, abs=1e-6)
        assert row["extra_platform_profit_yuan"] == pytest.approx(0, abs=1e-6)
        assert row["extra_user_gain_yuan"] == pytest.approx(0, abs=1e-6)
        assert not row["menu_open"] and not action["amendment_open"]
        assert action["raw_menu"][-1] == 0
        assert action["outside_source_options"] == list(old.selected_options)
    assert all(row["requested"] == row["chosen"] == 0 for row in contracts)


def test_free_amendments_are_separate_from_exit_and_all_rules_share_outside_cases():
    records = []
    for order in evaluation.ORDERS:
        factory, catalogue, observer = market(order)
        rows, actions, contracts = evaluation.evaluate_policy(factory, catalogue, observer,
            users(), None, range(5), order, "zero_rebate_fixed_users", constant=1)
        records.extend(rows)
        assert all(action["raw_menu"][-1] == 1 for action in actions)
        assert all(np.count_nonzero(action["raw_menu"][:12]) == 0 for action in actions)
        assert all(row["menu_open"] and not row["nonzero_extra_rebate_quote"] for row in rows)
        for row in rows:
            seed = row["seed"]
            assert row["carbon_budget_t"] == outside(seed).scenario.carbon_budget_t
            assert row["old_system_cost_yuan"] == pytest.approx(outside(seed).metrics["system_cost_yuan"])
            assert row["old_platform_profit_yuan"] == pytest.approx(outside(seed).metrics["platform_profit_yuan"])
            assert row["same_cap_compliant"] and row["old_same_cap_compliant"]
            assert row["all_work_complete"] and row["all_deadlines_met"]
            assert row["platform_old_floor_satisfied"] and row["user_ir_satisfied"]
            assert row["extra_cost_saving_yuan"] == pytest.approx(
                row["extra_platform_profit_yuan"] + row["extra_user_gain_yuan"] + row["extra_buyer_gain_yuan"], abs=1e-6)
        assert all(row["individual_rational"] for row in contracts)
    summary = evaluation.summarise(records)
    assert len(summary) == 3
    assert list(summary.days) == [5, 5, 5]
    assert list(summary.free_amendment_quote_days) == [5, 5, 5]
    assert list(summary.same_cap_compliant_days) == [5, 5, 5]


def test_learned_platform_replay_uses_one_network_prediction_per_day_and_pairs_increments():
    factory, catalogue, observer = market()
    platform = PlatformNetworkFixture([0, 1, 3, 7, 15])
    rows, actions, contracts = evaluation.evaluate_policy(factory, catalogue, observer,
        users(), platform, range(5), "net", "ppo_fixed_users")
    baseline, _, _ = evaluation.evaluate_policy(factory, catalogue, observer,
        users(), None, range(5), "net", "zero_rebate_fixed_users", constant=1)
    assert platform.calls == 5
    assert [row["menu"] for row in rows] == [0, 1, 3, 7, 15]
    assert [action["raw_menu"][-1] for action in actions] == [0, 1, 1, 1, 1]
    assert len(contracts) == 60
    pairs = evaluation.paired_comparisons(baseline + rows)
    assert len(pairs) == 5
    assert {row["comparison"] for row in pairs} == {"net_ppo_vs_zero_rebate_fixed_users"}
    for pair, before, after in zip(pairs, baseline, rows):
        assert pair["extra_cost_saving_yuan"] == pytest.approx(
            after["extra_cost_saving_yuan"] - before["extra_cost_saving_yuan"])
    # Selected actions and a floor supplied by clearing are separate observations.
    assert rows[0]["platform_old_floor_satisfied"] and not rows[0]["has_additional_effect"]


def test_enrichment_rejects_inconsistent_money_or_social_accounts():
    factory, catalogue, _ = market()
    metrics = evaluation.play_menu(factory(0), catalogue, 3, users())
    with pytest.raises(ValueError, match="cash difference"):
        evaluation.enrich(dict(metrics, platform_gain_vs_outside_yuan=metrics["platform_gain_vs_outside_yuan"] + 1),
                          "net", "ppo_fixed_users", 0, 3, catalogue.labels[3])
    with pytest.raises(ValueError, match="resource saving minus additional inconvenience"):
        evaluation.enrich(dict(metrics, contract_system_cost_saving_yuan=metrics["contract_system_cost_saving_yuan"] + 1),
                          "net", "ppo_fixed_users", 0, 3, catalogue.labels[3])


def test_constant_is_selected_from_all_training_menus_once_and_reused_without_new_days(tmp_path, monkeypatch):
    legacy, initial, output = tmp_path / "legacy", tmp_path / "initial", tmp_path / "constant"
    (legacy / "selected").mkdir(parents=True)
    initial.mkdir()
    (legacy / "selection.json").write_text('{"selected": "frozen old fixture"}', encoding="utf-8")
    for user in range(3):
        (initial / f"user_{user}.zip").write_bytes(f"new-user-{user}".encode())
        (legacy / "selected" / f"user_{user}.zip").write_bytes(f"old-user-{user}".encode())
    config = AmendmentConfig()
    factory, _, _ = market()
    seen = []
    def training_factory(seed):
        assert seed in (0, 1)
        seen.append(seed)
        return factory(seed)
    monkeypatch.setattr(evaluation, "make_factories", lambda *args, **kwargs: training_factory)
    monkeypatch.setattr(evaluation, "load_users", lambda directory: users())
    selected = evaluation.fit_constant(output, legacy, config, initial, (0, 1))
    assert seen == [0, 1]
    assert selected["searched_menus"] == selected["eligible_menus"] == 78
    assert selected["extra_rebate_menu"] >= 2
    means = pd.read_csv(output / "training_menu_means.csv")
    expected = means.sort_values(["platform_gain_vs_outside_yuan", "menu"], ascending=[False, True]).iloc[0]
    assert selected["menu"] == int(expected.menu)
    outcomes = pd.read_csv(output / "training_menu_outcomes.csv")
    assert len(outcomes) == 2 * 78 and set(outcomes.seed) == {0, 1}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_scenario_seeds"] == [0, 1]
    assert "no test-day selection" in selected["selection"]
    repeated = evaluation.fit_constant(output, legacy, config, initial, (0, 1))
    assert repeated == selected and seen == [0, 1]


def test_optional_finite_menu_reference_is_marked_as_hindsight_not_a_strategy(tmp_path):
    factory, catalogue, _ = market()
    report = evaluation.finite_menu_potential(tmp_path, factory, catalogue, users(), (4,), "net")
    selected = pd.read_csv(tmp_path / "hindsight_by_day.csv")
    candidates = pd.read_csv(tmp_path / "candidate_menus.csv")
    assert len(candidates) == 78 and set(candidates.seed) == {4}
    assert set(selected.phase) == {"hindsight_extra_profit", "hindsight_extra_cost_saving"}
    assert selected.hindsight_only.all()
    assert "no test strategy or model selection" in report["scope"]
