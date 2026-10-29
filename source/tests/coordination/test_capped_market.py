"""All institutions inherit the same original-workload physical allowance."""
from dataclasses import dataclass, replace

import numpy as np
import pytest

from energy_net.coordination.capped_market import (
    CAP_TOLERANCE_T, capped_episode_factory, legacy_capped_config, make_capped_scenario,
)
from energy_net.coordination.flex_market import FlexMarketConfig
from energy_net.coordination.gpu_scenario import GpuMarketConfig, make_gpu_scenario
from energy_net.coordination.grid import GridClearing
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue


@pytest.mark.parametrize("seed", range(8))
def test_cap_is_original_unconstrained_emissions_and_preserves_original_physics(seed):
    cfg = legacy_capped_config()
    original = make_gpu_scenario(seed, cfg)
    constrained = make_capped_scenario(seed, cfg)
    grid = GridClearing(original.grid_spec)
    reference = grid.solve(original.baseline_load_mw, None)
    assert constrained.carbon_budget_t == pytest.approx(reference.emissions_t + CAP_TOLERANCE_T)
    np.testing.assert_array_equal(original.baseline_load_mw, constrained.baseline_load_mw)
    np.testing.assert_array_equal(original.background_mw, constrained.background_mw)
    np.testing.assert_array_equal(original.renewable_mw, constrained.renewable_mw)
    assert original.total_work_cu == constrained.total_work_cu
    assert [j.deadline_step for j in original.jobs] == [j.deadline_step for j in constrained.jobs]
    result = grid.solve(constrained.baseline_load_mw, constrained.carbon_budget_t)
    assert result.feasible
    assert result.emissions_t <= constrained.carbon_budget_t + 1e-7
    assert result.total_generation_cost == pytest.approx(reference.total_generation_cost, abs=1e-5)
    bg = grid.solve(constrained.background_mw, constrained.carbon_budget_t)
    assert constrained.calibration["background_only_generation_cost_yuan"] == pytest.approx(bg.total_generation_cost)
    assert constrained.calibration["background_only_carbon_budget_t"] == constrained.carbon_budget_t


def test_common_cap_does_not_depend_on_quote_eligibility_or_sharing_fields():
    base = FlexMarketConfig(conditional_rebates=False, sharing_enabled=False, carbon_eligibility=False)
    alternate = replace(base, conditional_rebates=True, sharing_enabled=True,
                        carbon_eligibility=True, resource_share=.3)
    one, two = make_capped_scenario(9, base), make_capped_scenario(9, alternate)
    assert one.carbon_budget_t == two.carbon_budget_t
    np.testing.assert_array_equal(one.baseline_load_mw, two.baseline_load_mw)


def test_factory_uses_old_settlement_and_independent_booking_state():
    supplied = FlexMarketConfig(conditional_rebates=True, sharing_enabled=True,
                                carbon_eligibility=True, internal_carbon_value_yuan_per_t=100)
    factory = capped_episode_factory(supplied, maxsize=2)
    one, two = factory(2), factory(2)
    assert not one.config.sharing_enabled
    assert not one.config.carbon_eligibility
    assert one.config.internal_carbon_value_yuan_per_t == 0
    assert one.config.conditional_rebates
    assert one.scenario is two.scenario and one is not two
    one.step_user(0)
    assert one.position == 1 and two.position == 0
    one.baseline_clearing.generation_mw[0, 0] = -12345
    assert two.baseline_clearing.generation_mw[0, 0] >= 0
    assert factory(2).baseline_clearing.generation_mw[0, 0] >= 0


def test_all_quotes_clear_under_the_same_cap_without_changing_allowance():
    factory = capped_episode_factory()
    cfg = factory.config
    catalogue = FixedGpuTariffCatalogue(cfg)
    baseline_cap = factory(3).scenario.carbon_budget_t
    for menu in (0, 42, 65):
        episode = factory(3)
        episode.set_platform_action(catalogue.decode(episode, menu))
        result = episode.play()
        assert result.metrics["completion_rate"] == pytest.approx(1)
        assert result.metrics["deadline_compliance"] == 1
        assert result.metrics["carbon_emissions_t"] <= baseline_cap + 1e-6
        assert result.metrics["carbon_budget_t"] == baseline_cap
        assert result.metrics["flexibility_settlement_yuan"] == 0


def test_scenario_and_legacy_factory_accept_a_new_frozen_config_subclass():
    @dataclass(frozen=True)
    class AmendmentLikeConfig(FlexMarketConfig):
        extra_parameter: float = .7
    cfg = AmendmentLikeConfig()
    scenario = make_capped_scenario(1, cfg)
    assert isinstance(scenario.config, AmendmentLikeConfig)
    episode = capped_episode_factory(cfg)(1)
    assert isinstance(episode.config, AmendmentLikeConfig)
    assert episode.config.extra_parameter == .7
    assert not episode.config.sharing_enabled


def test_legacy_default_is_published_price_without_internal_green_preference():
    cfg = legacy_capped_config()
    assert not cfg.conditional_rebates and not cfg.sharing_enabled
    assert not cfg.carbon_eligibility and cfg.internal_carbon_value_yuan_per_t == 0
    assert cfg.carbon_price_yuan_per_t == 100
    with pytest.raises(TypeError):
        legacy_capped_config(GpuMarketConfig())
    with pytest.raises(ValueError):
        capped_episode_factory(maxsize=0)
