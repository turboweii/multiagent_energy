"""Varied days preserve contracts and use independently sampled physical inputs."""
from dataclasses import replace

import numpy as np
import pytest

from energy_net.coordination.capped_market import legacy_capped_config
from energy_net.coordination.gpu_scenario import make_gpu_scenario
from energy_net.coordination.grid import GridClearing
from energy_net.coordination.varied_scenario import VariedMarketConfig, make_varied_scenario


@pytest.fixture(scope="module")
def days():
    return tuple(make_varied_scenario(seed) for seed in range(8))


def assert_same_physics(first, second):
    np.testing.assert_array_equal(first.background_mw, second.background_mw)
    np.testing.assert_array_equal(first.renewable_mw, second.renewable_mw)
    np.testing.assert_array_equal(first.grid_spec.generator_capacity_mw, second.grid_spec.generator_capacity_mw)
    np.testing.assert_array_equal(first.grid_spec.generator_cost_yuan_per_mwh, second.grid_spec.generator_cost_yuan_per_mwh)
    assert first.carbon_budget_t == second.carbon_budget_t
    for left, right in zip(first.jobs, second.jobs):
        assert (left.job_id, left.user_id, left.home_region, left.release_step,
                left.deadline_step, left.beta_yuan_per_cu_hour, left.migration_loss_yuan_per_cu) == (
                right.job_id, right.user_id, right.home_region, right.release_step,
                right.deadline_step, right.beta_yuan_per_cu_hour, right.migration_loss_yuan_per_cu)
        for a, b in zip(left.options, right.options):
            for name in ("allocation_cu", "extra_energy_mwh", "transfer_gb"):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_reproducible_with_independent_arrays_and_legacy_economic_normalization(days):
    config = VariedMarketConfig()
    first, second = days[0], make_varied_scenario(0, config)
    assert_same_physics(first, second)
    assert not np.shares_memory(first.background_mw, second.background_mw)
    normalized = legacy_capped_config(config)
    assert isinstance(normalized, VariedMarketConfig)
    assert_same_physics(first, make_varied_scenario(0, normalized))
    # Legacy config classes without the weather fields use the same defaults.
    assert_same_physics(first, make_varied_scenario(0, legacy_capped_config()))


def test_fixed_hardware_thermal_costs_network_and_regional_nameplates(days):
    config = VariedMarketConfig()
    for scenario in days:
        spec = scenario.grid_spec
        assert config.pue == (1.15, 1.25, 1.35)
        assert spec.n_generators == 6
        np.testing.assert_array_equal(spec.generator_bus, [0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(spec.generator_cost_yuan_per_mwh, [180, 450, 520, 15, 15, 15])
        np.testing.assert_array_equal(spec.generator_emissions_t_per_mwh, [.9, .35, .35, 0, 0, 0])
        np.testing.assert_array_equal(spec.generator_capacity_mw[:, :3], np.tile([55, 35, 15], (48, 1)))
        np.testing.assert_array_equal(spec.line_capacity_mw, [18, 7, 8])
        np.testing.assert_allclose(spec.generator_capacity_mw[:, 3:].sum(axis=1), scenario.renewable_mw)
        assert scenario.calibration["regional_solar_capacity_mw"] == [12, 42, 38]
        assert scenario.calibration["regional_wind_capacity_mw"] == [28, 12, 16]


def test_smooth_independent_weather_preserves_nameplates_and_daily_wind_mean(days):
    config = VariedMarketConfig()
    daily_wind = []
    for scenario in days:
        data = scenario.calibration
        solar, wind = np.asarray(data["solar_available_mw"]), np.asarray(data["wind_available_mw"])
        wind_cf = wind / np.asarray(config.wind_capacity_mw)
        assert np.all(solar >= 0) and np.all(wind >= 0)
        assert np.all(solar <= config.solar_capacity_mw)
        assert np.all(wind_cf <= 1)
        assert np.all(solar[:12] == 0) and np.all(solar[37:] == 0)
        np.testing.assert_allclose(wind_cf.mean(axis=0), data["wind_daily_capacity_factors"], atol=1e-14)
        assert all(.15 <= value <= .65 for value in data["wind_daily_capacity_factors"])
        assert all(value >= .25 for value in data["effective_cloud_factor_min"])
        assert all(value <= 1 for value in data["effective_cloud_factor_max"])
        assert np.max(np.abs(np.diff(wind_cf, axis=0))) < .08
        assert len(set(data["cloud_daily_factors"])) == 3
        daily_wind.append(data["wind_daily_capacity_factors"])
        np.testing.assert_allclose(solar + wind, scenario.grid_spec.generator_capacity_mw[:, 3:])
    assert np.max(np.std(daily_wind, axis=0)) > .05


def test_home_permutations_and_common_shift_preserve_every_task_option(days):
    seen = {user: set() for user in range(3)}
    for scenario in days:
        config = scenario.config
        originals = {job.job_id: job for job in make_gpu_scenario(scenario.seed, config).jobs}
        shift = scenario.calibration["common_release_jitter_steps"]
        assert shift in (-1, 0, 1)
        for release in sorted({job.release_step for job in scenario.jobs}):
            batch = [job for job in scenario.jobs if job.release_step == release]
            assert sorted(job.home_region for job in batch) == [0, 1, 2]
            assert sorted(job.user_id for job in batch) == [0, 1, 2]
        for job in scenario.jobs:
            original = originals[job.job_id]
            seen[job.user_id].add(job.home_region)
            assert job.work_cu == original.work_cu
            assert job.beta_yuan_per_cu_hour == original.beta_yuan_per_cu_hour
            assert job.migration_loss_yuan_per_cu == original.migration_loss_yuan_per_cu
            assert job.release_step == original.release_step + shift
            assert job.deadline_step == original.deadline_step + shift <= 48
            assert 0 <= job.release_step < job.deadline_step
            for option, old in zip(job.options, original.options):
                assert option.start_step == old.start_step + shift
                assert option.end_step == old.end_step + shift <= job.deadline_step
                assert option.allocation_cu.sum() == pytest.approx(job.work_cu)
                assert np.count_nonzero(option.allocation_cu.sum(axis=0)) == 1
                assert option.allocation_cu[:, option.region].sum() == pytest.approx(job.work_cu)
                assert option.delay_hours == old.delay_hours
                assert option.operating_cost_yuan == old.operating_cost_yuan
                assert option.migrated == (option.region != job.home_region)
                loss = job.work_cu * (job.beta_yuan_per_cu_hour * option.delay_hours
                                      + job.migration_loss_yuan_per_cu * option.migrated)
                assert option.user_loss_yuan == pytest.approx(loss)
                if option.migrated:
                    volume = job.work_cu * config.transfer_gb_per_cu
                    assert option.transfer_gb[option.start_step - 1, job.home_region, option.region] == pytest.approx(volume)
                    assert np.count_nonzero(option.transfer_gb) == 1
                    assert option.extra_energy_mwh[option.start_step - 1, option.region] == pytest.approx(volume * config.transfer_mwh_per_gb)
                else:
                    assert not np.any(option.transfer_gb) and not np.any(option.extra_energy_mwh)
    assert all(locations == {0, 1, 2} for locations in seen.values())


def test_common_cap_is_before_quotes_and_background_uses_that_exact_cap(days):
    for scenario in days:
        grid = GridClearing(scenario.grid_spec)
        unconstrained = grid.solve(scenario.baseline_load_mw, None)
        assert unconstrained.feasible
        assert scenario.carbon_budget_t == pytest.approx(unconstrained.emissions_t + 1e-7, abs=1e-9)
        constrained = grid.solve(scenario.baseline_load_mw, scenario.carbon_budget_t)
        background = grid.solve(scenario.background_mw, scenario.carbon_budget_t)
        assert constrained.feasible and background.feasible
        assert constrained.emissions_t <= scenario.carbon_budget_t + 1e-6
        assert background.total_generation_cost == pytest.approx(scenario.calibration["background_only_generation_cost_yuan"])
        assert scenario.calibration["background_only_carbon_budget_t"] == scenario.carbon_budget_t
        allocation = sum((job.options[0].allocation_cu for job in scenario.jobs),
                         np.zeros_like(scenario.background_mw))
        assert np.all(allocation <= np.asarray(scenario.config.capacity_cu_per_hour) * scenario.config.dt_hours)


def test_background_shapes_and_phases_are_independent_by_region(days):
    assert any(len(set(day.calibration["background_shape_indices"])) > 1 for day in days)
    assert any(len(set(day.calibration["background_peak_shifts_steps"])) > 1 for day in days)
    for day in days:
        assert all(.85 <= value <= 1.15 for value in day.calibration["background_daily_multipliers"])
        assert day.background_mw.shape == (48, 3)
        assert np.all(day.background_mw > 0)


def test_infeasible_declared_inputs_raise_instead_of_resampling():
    config = VariedMarketConfig(background_daily_multiplier_bounds=(10., 10.))
    with pytest.raises(ValueError, match="Varied physical day 0.*infeasible original"):
        make_varied_scenario(0, config)


@pytest.mark.parametrize("changes", [
    {"solar_capacity_mw": (-1, 2, 3)}, {"wind_capacity_mw": (1, 2)},
    {"cloud_factor_bounds": (-.1, 1)}, {"wind_daily_capacity_factor_bounds": (.7, .2)},
    {"background_daily_multiplier_bounds": (0., 1.)}, {"release_jitter_steps": (.5,)},
])
def test_invalid_physical_parameters_are_explicit(changes):
    with pytest.raises(ValueError):
        VariedMarketConfig(**changes)
