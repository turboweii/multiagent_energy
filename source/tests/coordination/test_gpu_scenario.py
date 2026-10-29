from dataclasses import replace

import numpy as np
import pytest

from energy_net.coordination.gpu_scenario import GpuMarketConfig, make_gpu_scenario
from energy_net.coordination.grid import GridClearing


def test_gpu_power_model_counts_idle_once_and_uses_gpu_hours():
    cfg = GpuMarketConfig()
    scenario = make_gpu_scenario(0, cfg)
    assert cfg.capacity_cu_per_hour == (6144.0,) * 3
    assert np.allclose(cfg.idle_power_mw, (0.70656, 0.768, 0.82944))
    assert np.allclose(cfg.energy_mwh_per_cu, (0.001035, 0.001125, 0.001215))
    for region, pue in enumerate(cfg.pue):
        full_dynamic = cfg.capacity_cu_per_hour[region] * cfg.energy_mwh_per_cu[region]
        full_total = full_dynamic + cfg.idle_power_mw[region]
        assert full_total == pytest.approx(cfg.installed_nodes_per_site * cfg.node_active_it_kw * pue / 1000)
    for job in scenario.jobs:
        original = job.options[0]
        allocated = original.allocation_cu[original.start_step:original.end_step].sum(axis=1)
        assert np.allclose(allocated / cfg.dt_hours, job.work_cu / cfg.batch_duration_hours)
        assert original.allocation_cu.sum() == pytest.approx(job.work_cu)


def test_gpu_jobs_keep_identical_work_and_deadlines_across_all_options():
    scenario = make_gpu_scenario(12)
    assert len(scenario.jobs) == 12
    for job in scenario.jobs:
        assert len(job.options) == 10
        assert job.options[0].is_baseline
        assert not job.options[1].is_baseline
        assert np.array_equal(job.options[0].allocation_cu, job.options[1].allocation_cu)
        for option in job.options:
            assert option.allocation_cu.sum() == pytest.approx(job.work_cu)
            assert option.end_step <= job.deadline_step <= 48
            assert option.end_step - option.start_step == 8
            assert option.user_loss_yuan >= 0


def test_gpu_scenario_reproducibility_and_carbon_price_does_not_change_jobs():
    one = make_gpu_scenario(7)
    two = make_gpu_scenario(7)
    higher_carbon = make_gpu_scenario(7, replace(GpuMarketConfig(), carbon_price_yuan_per_t=600))
    assert one.carbon_budget_t is None
    assert one.grid_spec.carbon_price_yuan_per_t == 100
    assert higher_carbon.grid_spec.carbon_price_yuan_per_t == 600
    for alternative in (two, higher_carbon):
        assert np.array_equal(one.background_mw, alternative.background_mw)
        assert np.array_equal(one.renewable_mw, alternative.renewable_mw)
        assert [job.work_cu for job in one.jobs] == [job.work_cu for job in alternative.jobs]
        for left, right in zip(one.jobs, alternative.jobs):
            assert left.beta_yuan_per_cu_hour == right.beta_yuan_per_cu_hour
            assert all(np.array_equal(a.allocation_cu, b.allocation_cu)
                       for a, b in zip(left.options, right.options))


def test_background_counterfactual_uses_same_spec_and_reports_resource_cost():
    scenario = make_gpu_scenario(7)
    background = GridClearing(scenario.grid_spec).solve(scenario.background_mw, None)
    assert background.feasible
    assert scenario.calibration["background_only_generation_cost_yuan"] == pytest.approx(background.total_generation_cost)
    assert scenario.calibration["background_only_generation_cost_yuan"] != pytest.approx(background.dispatch_objective_yuan)
    assert scenario.config.internal_carbon_value_yuan_per_t == 100
    assert "underutilized" in scenario.calibration["capacity_utilization_status"]
    assert "not a completed" in scenario.calibration["calibration_status"]


@pytest.mark.parametrize("seed", range(32))
def test_gpu_baseline_and_remote_contracts_have_physical_space(seed):
    scenario = make_gpu_scenario(seed)
    cfg = scenario.config
    assert GridClearing(scenario.grid_spec).solve(scenario.baseline_load_mw, None).feasible
    allocation = sum((job.options[0].allocation_cu for job in scenario.jobs), np.zeros((48, 3)))
    assert np.all(allocation <= np.asarray(cfg.capacity_cu_per_hour) * cfg.dt_hours + 1e-7)
    assert min(scenario.calibration["physically_feasible_remote_options_per_job"]) >= 1
    assert 0.08 <= scenario.calibration["dc_energy_share"] <= 0.15
    grid = GridClearing(scenario.grid_spec)
    original_indices = [0] * len(scenario.jobs)
    for position, job in enumerate(scenario.jobs):
        for option in job.options:
            if not option.migrated:
                continue
            proposed = original_indices.copy()
            proposed[position] = option.index
            allocation = sum((j.options[i].allocation_cu for j, i in zip(scenario.jobs, proposed)),
                             np.zeros((48, 3)))
            if np.any(allocation > np.asarray(cfg.capacity_cu_per_hour) * cfg.dt_hours + 1e-7):
                continue
            if grid.solve(scenario.load_for(proposed), None).feasible:
                break
        else:
            pytest.fail(f"seed {seed}: job {job.job_id} has no feasible remote contract")


def test_gpu_scale_changes_hardware_and_workload_together():
    small = make_gpu_scenario(0, GpuMarketConfig(site_scale=0.5))
    large = make_gpu_scenario(0, GpuMarketConfig(site_scale=2))
    assert small.config.capacity_cu_per_hour[0] * 4 == large.config.capacity_cu_per_hour[0]
    assert small.total_work_cu < large.total_work_cu
    assert small.calibration["dc_energy_share"] < large.calibration["dc_energy_share"]
    assert small.config.daily_fixed_compute_cost_yuan * 4 == large.config.daily_fixed_compute_cost_yuan


def test_heavy_checkpoint_is_an_explicit_sensitivity():
    light = make_gpu_scenario(0)
    heavy = make_gpu_scenario(0, replace(GpuMarketConfig(), transfer_gb_per_cu=2.0))
    for left, right in zip(light.jobs, heavy.jobs):
        remote_light = next(option for option in left.options if option.migrated)
        remote_heavy = next(option for option in right.options if option.migrated)
        assert remote_heavy.transfer_gb.sum() == pytest.approx(20 * remote_light.transfer_gb.sum())
        assert left.work_cu == right.work_cu


@pytest.mark.parametrize("settings", [
    {"site_scale": 0}, {"node_idle_it_kw": 9.0}, {"pue": (0.9, 1.2, 1.3)},
    {"batch_duration_hours": 4.1}, {"batch_release_steps": (1, 13, 25, 36)},
    {"internal_carbon_value_yuan_per_t": -1},
    {"internal_carbon_value_yuan_per_t": float("nan")},
    {"internal_carbon_value_yuan_per_t": float("inf")},
])
def test_invalid_physical_assumptions_are_rejected(settings):
    with pytest.raises(ValueError):
        GpuMarketConfig(**settings)
