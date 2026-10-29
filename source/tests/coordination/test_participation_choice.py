"""A private reservation utility retains the owner's learned ranking."""
from dataclasses import replace
from pathlib import Path
import os

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from energy_net.coordination.learning import ActorEnv, PolicyBundle, cached_episode_factory
from energy_net.coordination.market import MarketEpisode, encode_platform_action
from energy_net.coordination.platform_features import candidate_tariff_observation
from energy_net.coordination.platform_learning import apply_private_participation
from energy_net.coordination.policies import ContractActorCriticPolicy
from energy_net.coordination.scenario import make_scenario
from energy_net.coordination.tariffs import TariffCatalogue


def contract_policy():
    return ContractActorCriticPolicy(
        spaces.Box(-np.inf, np.inf, shape=(92,), dtype=np.float32),
        spaces.Discrete(10), lambda fraction: 3e-4,
        participation_constraint=True)


def probabilities(policy, observation):
    tensor, _ = policy.obs_to_tensor(observation)
    with torch.no_grad():
        return policy.get_distribution(tensor).distribution.probs[0].cpu().numpy()


def test_no_discount_excludes_uncompensated_delay_and_migration():
    episode = MarketEpisode(make_scenario(9000))
    observation = episode.current_user_observation()
    options = observation[6:76].reshape(10, 7)
    probability = probabilities(contract_policy(), observation)
    assert np.all(options[2:, 2] + options[2:, 3] > 1.0)
    assert probability[0] > 0 and probability[1] > 0
    assert np.all(probability[2:] == 0)
    assert probability.sum() == pytest.approx(1.0)


def test_constraint_preserves_all_acceptable_contracts_not_only_cheapest():
    observation = np.zeros(92, dtype=np.float32)
    options = observation[6:76].reshape(10, 7)
    options[:, :2] = 1.0
    options[:, 2] = [1.0, .90, .80, .70, .95, .50, .90, .80, .75, .99]
    options[:, 3] = [0.0, 0.0, .10, .10, .10, .60, .20, .30, .30, .10]
    probability = probabilities(contract_policy(), observation)
    # Four acceptable offers include the original and several unequal costs.
    assert np.all(probability[:4] > 0)
    assert np.all(probability[4:] == 0)
    assert probability[3] < 1.0  # The lowest cost was not chosen by a hard argmin.
    assert probability.sum() == pytest.approx(1.0)


def test_private_costs_change_user_constraint_without_changing_platform_features():
    scenario = make_scenario(9000)
    private_jobs = []
    for job in scenario.jobs:
        beta, kappa = job.beta_yuan_per_cu_hour * 3, job.migration_loss_yuan_per_cu * 2
        options = tuple(replace(option, user_loss_yuan=job.work_cu * (
            beta * option.delay_hours + kappa * option.migrated)) for option in job.options)
        private_jobs.append(replace(job, beta_yuan_per_cu_hour=beta,
                                    migration_loss_yuan_per_cu=kappa, options=options))
    altered = replace(scenario, jobs=tuple(private_jobs))
    original, counterfactual = MarketEpisode(scenario), MarketEpisode(altered)
    codec = TariffCatalogue()
    np.testing.assert_array_equal(original.platform_observation(), counterfactual.platform_observation())
    np.testing.assert_array_equal(candidate_tariff_observation(original, codec),
                                  candidate_tariff_observation(counterfactual, codec))
    action = encode_platform_action(np.full((3, 4), .10))
    original.set_platform_action(action)
    counterfactual.set_platform_action(action)
    assert not np.array_equal(original.current_user_observation(),
                              counterfactual.current_user_observation())


def test_enabling_existing_ppo_preserves_weights_and_survives_save_load(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        scenario = make_scenario(9000)
        env = ActorEnv(lambda seed: MarketEpisode(scenario), PolicyBundle(), 0, [9000])
        model = PPO(ContractActorCriticPolicy, env, n_steps=2, batch_size=2,
                    n_epochs=1, seed=0, device="cpu", verbose=0)
        before = {name: tensor.detach().clone() for name, tensor in model.policy.state_dict().items()}
        assert not model.policy.participation_constraint
        assert apply_private_participation(model) is model
        assert model.policy.participation_constraint
        assert model.policy.mlp_extractor.participation_constraint
        assert model.policy_kwargs["participation_constraint"] is True
        for name, tensor in model.policy.state_dict().items():
            assert torch.equal(tensor, before[name])
        model.save(tmp_path / "participation")
        restored = PPO.load(tmp_path / "participation", device="cpu")
        assert restored.policy.participation_constraint
        assert restored.policy.mlp_extractor.participation_constraint
        assert restored.policy_kwargs["participation_constraint"] is True
        sample = MarketEpisode(scenario).current_user_observation()
        np.testing.assert_array_equal(probabilities(model.policy, sample),
                                      probabilities(restored.policy, sample))
        for name, tensor in restored.policy.state_dict().items():
            assert torch.equal(tensor, before[name])
        env.close()
    finally:
        torch.set_num_threads(previous)


def test_trained_users_zero_discount_returns_original_settlement_on_validation_days():
    directory = Path(os.environ.get("COORDINATION_USER_CHECKPOINTS",
        "/Users/turbo/Documents/Codex/2026-10-28/levy-m-segev-i-tuisov-a/outputs/改进训练/seed0-retained/checkpoints"))
    if not all((directory / f"user_{owner}.zip").is_file() for owner in range(3)):
        pytest.skip("Requires the locally trained user checkpoint artifacts.")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        users = {owner: apply_private_participation(PPO.load(
            directory / f"user_{owner}.zip", device="cpu")) for owner in range(3)}
        factory = cached_episode_factory(maxsize=16)
        for seed in range(9000, 9016):
            outcome = factory(seed).play(users)
            assert outcome.metrics["user_ir_rate"] == 1.0
            assert outcome.metrics["migrated_jobs"] == 0
            assert outcome.metrics["delayed_jobs"] == 0
            assert outcome.metrics["delta_platform_profit_yuan"] == pytest.approx(0.0, abs=1e-7)
            assert outcome.metrics["delta_system_cost_yuan"] == pytest.approx(0.0, abs=1e-7)
            assert outcome.metrics["completion_rate"] == pytest.approx(1.0)
            assert outcome.metrics["deadline_compliance"] == 1.0
    finally:
        torch.set_num_threads(previous)
