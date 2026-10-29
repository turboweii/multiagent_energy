"""Diverse posted-price coverage and private user-comparison training."""
import numpy as np
import pytest

from energy_net.coordination.curriculum import (
    DiverseQuoteBundle, evaluate_user_responses, pretrain_users,
)
from energy_net.coordination.learning import PolicyBundle, cached_episode_factory


class OriginalContract:
    def predict(self, observation, deterministic=True):
        assert deterministic
        return np.array(0), None


def test_diverse_prices_are_seeded_and_change_contract_costs():
    from energy_net.coordination.tariffs import TariffCatalogue
    factory = cached_episode_factory(maxsize=2)
    episode = factory(0)
    left = DiverseQuoteBundle(TariffCatalogue(episode.config), seed=7)
    right = DiverseQuoteBundle(TariffCatalogue(episode.config), seed=7)
    quotes = []
    for _ in range(32):
        quote = left.platform_action(episode)
        assert np.array_equal(quote, right.platform_action(episode))
        episode.set_platform_action(quote)
        quotes.append(tuple(quote))
    assert len(set(quotes)) > 6


def test_response_evaluation_is_reproducible_and_keeps_complete_original_work():
    factory = cached_episode_factory(maxsize=2)
    bundle = PolicyBundle(users={owner: OriginalContract() for owner in range(3)})
    result = evaluate_user_responses(bundle, factory, [0], quotes_per_day=3, seed=17)
    repeated = evaluate_user_responses(bundle, factory, [0], quotes_per_day=3, seed=17)
    assert result == repeated
    summary = result["summary"]
    assert summary["tasks"] == 36
    assert summary["episodes"] == 3
    assert summary["completion_rate"] == pytest.approx(1)
    assert summary["deadline_compliance"] == pytest.approx(1)
    assert summary["user_ir_rate"] == pytest.approx(1)
    assert set(result["per_user"]) == {0, 1, 2}
    assert all(row["current_task_regret_yuan"] >= 0 for row in result["tasks"])
    assert all(row["requested"] == row["chosen"] == 0 for row in result["tasks"])
    for row in result["tasks"]:
        assert row["generalized_cost_yuan"] - row["current_task_best_cost_yuan"] == pytest.approx(
            row["current_task_regret_yuan"])


def test_tiny_curriculum_updates_three_separate_ppo_users():
    import torch
    torch.set_num_threads(1)
    factory = cached_episode_factory(maxsize=2)
    received = []
    bundle, records = pretrain_users(
        factory, [0, 1], steps_per_user=128, round_steps=128, seed=0,
        on_phase=lambda record, models: received.append(record["actor"]))
    assert bundle.platform is None
    assert set(bundle.users) == {0, 1, 2}
    assert received == ["0", "1", "2"]
    assert len(records) == 3
    for record in records:
        assert record["actual_actor_steps"] == 128
        assert record["stage_actor_steps"] == 128
        assert record["optimizer_steps"] == 6
    parameters = [next(model.policy.parameters()) for model in bundle.users.values()]
    assert len({parameter.data_ptr() for parameter in parameters}) == 3
    assert all(model.gamma == 0 for model in bundle.users.values())


def test_resumed_curriculum_seeds_fresh_envs_without_reinitializing_models():
    import torch
    torch.set_num_threads(1)
    factory = cached_episode_factory(maxsize=4)
    bundle, _ = pretrain_users(factory, list(range(64)), steps_per_user=128,
                              round_steps=128, seed=11)
    identities = {owner: id(model) for owner, model in bundle.users.items()}
    traces = []
    for _ in range(2):
        bundle, _ = pretrain_users(factory, list(range(64)), steps_per_user=128,
                                  round_steps=128, seed=11, users=bundle.users)
        current = {}
        for owner, model in bundle.users.items():
            assert id(model) == identities[owner]
            env = model.get_env().envs[0].unwrapped
            assert env.np_random_seed == 11 + owner + 1
            current[owner] = env.np_random.integers(0, 64, size=12).tolist()
        traces.append(current)
    assert traces[0] == traces[1]
    assert len({tuple(values) for values in traces[0].values()}) == 3
