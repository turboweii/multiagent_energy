"""Implementable, immutable selection of the old-institution outside policy."""
from dataclasses import asdict, FrozenInstanceError
import json

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from energy_net.coordination.capped_market import capped_episode_factory
from energy_net.coordination.flex_market import FlexPlatformEnv
from energy_net.coordination.flex_policy import FlexTariffActorCriticPolicy
from energy_net.coordination.flex_tariffs import FlexTariffObserver
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue
from energy_net.coordination.learning import ActorEnv, PolicyBundle
from energy_net.coordination.legacy_reference import FixedLegacyMenu, load_legacy_bundle
from energy_net.coordination.outside_bookings import make_outside_booking, outside_booking_factory
from energy_net.coordination.policies import ContractActorCriticPolicy


@pytest.fixture
def saved_reference(tmp_path):
    torch.set_num_threads(1)
    factory = capped_episode_factory()
    prototype = factory(0)
    catalogue = FixedGpuTariffCatalogue(factory.config)
    bundle = PolicyBundle(platform_codec=catalogue)
    directory = tmp_path / "selected"; directory.mkdir()
    for user in range(3):
        env = ActorEnv(factory, bundle, user, [0], prototype)
        model = PPO(ContractActorCriticPolicy, env, n_steps=2, batch_size=2,
            n_epochs=1, gamma=0, seed=user+1,
            policy_kwargs=dict(economic_only=True, participation_constraint=True))
        bundle.users[user] = model
        model.save(directory / f"user_{user}")
    bundle.platform_observer = FlexTariffObserver(catalogue)
    env = FlexPlatformEnv(factory, bundle, "platform", [0], prototype)
    platform = PPO(FlexTariffActorCriticPolicy, env, n_steps=2, batch_size=2,
                   n_epochs=1, gamma=0, seed=0)
    platform.save(directory / "platform")
    return tmp_path, factory


def write_selection(path, config, kind, menu=None):
    data = dict(config=asdict(config), policy_kind=kind,
                menu_index=menu, platform_model="selected/platform.zip" if kind=="ppo" else None,
                test_scenario_seeds_used=[], selected_candidate="fixture")
    (path / "selection.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("kind,menu", [("ppo",None),("constant",0),("constant",65)])
def test_selected_reference_loads_frozen_models_and_can_make_real_bookings(saved_reference,kind,menu):
    path, factory = saved_reference
    write_selection(path,factory.config,kind,menu)
    bundle = load_legacy_bundle(path)
    assert not bundle.training and bundle.legacy_config == factory.config
    for model in (*bundle.users.values(), bundle.platform):
        policy = getattr(model,"policy",None)
        if policy is not None:
            assert not policy.training
            assert all(not parameter.requires_grad for parameter in policy.parameters())
    booking = make_outside_booking(0,bundle,factory)
    assert booking.metrics["completion_rate"] == 1
    assert booking.metrics["deadline_compliance"] == 1
    assert booking.metrics["carbon_emissions_t"] <= booking.scenario.carbon_budget_t + 1e-6
    assert booking.metrics["flexibility_settlement_yuan"] == 0
    cached = outside_booking_factory(bundle,factory)
    first,second = cached(0),cached(0)
    assert first.selected_options == second.selected_options


def test_constant_quote_is_immutable_and_does_not_depend_on_observations():
    policy = FixedLegacyMenu(42)
    assert int(policy.predict(np.zeros(485))[0]) == 42
    assert int(policy.predict(np.full((77,186),np.nan))[0]) == 42
    np.testing.assert_array_equal(policy.predict(np.zeros((3,77,186)))[0],[42,42,42])
    with pytest.raises(FrozenInstanceError):
        policy.menu_index = 65
    with pytest.raises(ValueError):
        FixedLegacyMenu(77)


def test_unknown_selection_kind_is_rejected(saved_reference):
    path,factory = saved_reference
    write_selection(path,factory.config,"hindsight_per_test_day")
    with pytest.raises(ValueError,match="Unknown selected"):
        load_legacy_bundle(path)
