"""Independent private draws preserve public physics and observation privacy."""
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from energy_net.coordination.amendment_market import amendment_episode_factory
from energy_net.coordination.capped_market import legacy_capped_config
from energy_net.coordination.dynamic_tariffs import ExpandedAmendmentCatalogue
from energy_net.coordination.flex_market import FlexMarketEpisode
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue
from energy_net.coordination.grid import GridClearing
from energy_net.coordination.learning import PolicyBundle
from energy_net.coordination.legacy_reference import FixedLegacyMenu
from energy_net.coordination.outside_bookings import make_outside_booking, outside_booking_factory
from energy_net.coordination.preference_scenario import (
    PREFERENCE_MULTIPLIER_BOUNDS, clone_with_preferences,
    decode_preference_episode_id, encode_preference_episode_id, make_preference_scenario,
)
from energy_net.coordination.roster_features import RosterAmendmentObserver
from energy_net.coordination.varied_scenario import VariedMarketConfig, make_varied_scenario


@pytest.fixture(scope='module')
def physical_days():
    return tuple(make_varied_scenario(seed) for seed in (0,1))


def coefficients(scenario):
    return {owner:next((job.beta_yuan_per_cu_hour,job.migration_loss_yuan_per_cu)
                       for job in scenario.jobs if job.user_id==owner) for owner in range(3)}


def assert_same_physical_inputs(first,second):
    assert first.config is second.config
    assert first.carbon_budget_t==second.carbon_budget_t
    assert first.grid_spec is second.grid_spec
    assert first.background_mw is second.background_mw
    assert first.renewable_mw is second.renewable_mw
    assert first.total_work_cu==second.total_work_cu
    np.testing.assert_array_equal(first.baseline_load_mw,second.baseline_load_mw)
    for left,right in zip(first.jobs,second.jobs):
        for name in ('job_id','user_id','work_cu','home_region','release_step','deadline_step'):
            assert getattr(left,name)==getattr(right,name)
        assert len(left.options)==len(right.options)
        for a,b in zip(left.options,right.options):
            for name in ('index','region','start_step','end_step','delay_hours','migrated',
                         'operating_cost_yuan','is_baseline'):
                assert getattr(a,name)==getattr(b,name)
            for name in ('allocation_cu','extra_energy_mwh','transfer_gb'):
                assert getattr(a,name) is getattr(b,name)
                np.testing.assert_array_equal(getattr(a,name),getattr(b,name))


@pytest.mark.parametrize('public,private',[(0,0),(255,1023),(70000,70001),
                                         (2**31-1,2**31-1),(2**64,2**64+1)])
def test_composite_id_is_negative_and_exactly_reversible(public,private):
    encoded=encode_preference_episode_id(public,private)
    assert encoded<0 and decode_preference_episode_id(encoded)==(public,private)


def test_composite_ids_do_not_alias_repeated_public_days_or_old_nonnegative_cases():
    ids={encode_preference_episode_id(public,private) for public in range(32) for private in range(32)}
    assert len(ids)==1024 and not ids.intersection(range(70032))
    assert decode_preference_episode_id(encode_preference_episode_id(np.int64(7),np.int64(9)))==(7,9)


@pytest.mark.parametrize('bad',[-1,.5,np.nan,np.inf,True])
def test_invalid_independent_seeds_do_not_silently_coalesce(bad):
    with pytest.raises(ValueError):encode_preference_episode_id(0,bad)
    with pytest.raises(ValueError):encode_preference_episode_id(bad,0)


@pytest.mark.parametrize('bad',[0,1,.5,True])
def test_only_composite_ids_can_be_decoded(bad):
    with pytest.raises(ValueError):decode_preference_episode_id(bad)


def test_all_physics_and_original_cap_are_unchanged_with_no_redispatch(physical_days,monkeypatch):
    original=physical_days[0]
    before=[(job.beta_yuan_per_cu_hour,job.migration_loss_yuan_per_cu,
             tuple(option.user_loss_yuan for option in job.options)) for job in original.jobs]
    def forbidden(*args,**kwargs):
        raise AssertionError('Preference resampling must not resolve OPF or recompute the original cap')
    monkeypatch.setattr(GridClearing,'solve',forbidden)
    draw=clone_with_preferences(original,1001)
    assert_same_physical_inputs(original,draw)
    after=[(job.beta_yuan_per_cu_hour,job.migration_loss_yuan_per_cu,
            tuple(option.user_loss_yuan for option in job.options)) for job in original.jobs]
    assert after==before
    assert draw.calibration['original_unconstrained_emissions_t']==original.calibration['original_unconstrained_emissions_t']
    assert draw.calibration['common_carbon_cap_t']==original.carbon_budget_t
    draw.calibration['cloud_daily_factors'][0]=-123
    assert original.calibration['cloud_daily_factors'][0]!=-123


def test_owner_classes_and_same_day_persistence_keep_the_original_distribution(physical_days):
    original=physical_days[0];draw=clone_with_preferences(original,101)
    assert PREFERENCE_MULTIPLIER_BOUNDS==(.8,1.2)
    assert coefficients(draw)!=coefficients(clone_with_preferences(original,102))
    for owner in range(3):
        jobs=[job for job in draw.jobs if job.user_id==owner]
        assert len(jobs)==4
        assert len({(job.beta_yuan_per_cu_hour,job.migration_loss_yuan_per_cu) for job in jobs})==1
        beta,kappa=coefficients(draw)[owner]
        assert .8<=beta/draw.config.beta_yuan_per_cu_hour[owner]<=1.2
        assert .8<=kappa/draw.config.migration_loss_yuan_per_cu[owner]<=1.2
        for job in jobs:
            for option in job.options:
                assert option.user_loss_yuan==pytest.approx(job.work_cu*(
                    beta*option.delay_hours+kappa*option.migrated))
    assert coefficients(draw)[0][0]<coefficients(draw)[1][0]<coefficients(draw)[2][0]
    assert coefficients(draw)[0][1]<coefficients(draw)[1][1]<coefficients(draw)[2][1]


def test_preference_rng_depends_only_on_its_own_seed_and_recloning_does_not_double_jitter(physical_days):
    first,second=(clone_with_preferences(day,1234) for day in physical_days)
    assert coefficients(first)==coefficients(second)
    assert first.calibration['public_seed']==0 and second.calibration['public_seed']==1
    again=clone_with_preferences(first,1235)
    direct=clone_with_preferences(physical_days[0],1235)
    assert coefficients(again)==coefficients(direct)
    assert again.seed==direct.seed==encode_preference_episode_id(0,1235)
    assert again.calibration['preference_seed']==1235
    assert_same_physical_inputs(first,again)


def test_convenience_factory_reproduces_original_generation_and_config():
    config=VariedMarketConfig(solar_capacity_mw=(10.,40.,36.))
    expected=make_varied_scenario(5,config)
    actual=make_preference_scenario(5,777,config)
    assert actual.seed==encode_preference_episode_id(5,777)
    assert actual.calibration['public_seed']==5 and actual.calibration['preference_seed']==777
    assert actual.carbon_budget_t==expected.carbon_budget_t
    np.testing.assert_array_equal(actual.background_mw,expected.background_mw)
    np.testing.assert_array_equal(actual.grid_spec.generator_capacity_mw,expected.grid_spec.generator_capacity_mw)
    np.testing.assert_array_equal(actual.baseline_load_mw,expected.baseline_load_mw)
    for a,b in zip(actual.jobs,expected.jobs):
        assert (a.job_id,a.home_region,a.release_step,a.deadline_step,a.work_cu)==(
            b.job_id,b.home_region,b.release_step,b.deadline_step,b.work_cu)


class OutsideOnly:
    def predict(self,observation,deterministic=True):
        return np.asarray(0,dtype=np.int64),None


@pytest.fixture
def draw_factories(physical_days):
    config=physical_days[0].config
    @lru_cache(maxsize=16)
    def scenario(case_id):
        public,private=decode_preference_episode_id(case_id)
        return clone_with_preferences(physical_days[public],private)
    oldconfig=legacy_capped_config(config)
    def old(case_id):
        return FlexMarketEpisode(replace(scenario(case_id),config=oldconfig))
    # Zero quotes and unchanged accepted tasks isolate direct preference
    # leakage from legitimate information revealed by old accepted behaviour.
    bundle=PolicyBundle(platform=FixedLegacyMenu(0),users={u:OutsideOnly() for u in range(3)},
                        platform_codec=FixedGpuTariffCatalogue(config))
    outside=outside_booking_factory(bundle,old)
    return scenario,old,outside,amendment_episode_factory(scenario,outside,config=config)


def test_outside_cache_keeps_different_private_draws_separate_and_repeatable(draw_factories):
    scenario,old,outside,factory=draw_factories
    a,b=encode_preference_episode_id(0,11),encode_preference_episode_id(0,12)
    one,two=outside(a),outside(b)
    assert one.scenario.seed==a and two.scenario.seed==b
    assert coefficients(one.scenario)!=coefficients(two.scenario)
    assert outside.cache_info().misses==2
    repeat=outside(a)
    assert coefficients(repeat.scenario)==coefficients(one.scenario)
    assert outside.cache_info().hits==1
    assert factory(a).scenario.seed==a and factory(b).scenario.seed==b
    assert one.scenario.carbon_budget_t==two.scenario.carbon_budget_t


def test_platform_observations_do_not_expose_preference_values_or_composite_ids(draw_factories):
    scenario,old,outside,factory=draw_factories
    a,b=encode_preference_episode_id(0,11),encode_preference_episode_id(0,12)
    first,second=factory(a),factory(b)
    assert first.outside_selected_options==second.outside_selected_options==(0,)*12
    np.testing.assert_array_equal(first.platform_observation(),second.platform_observation())
    # Separate observers force an actual public-feature computation for both
    # draws; an integer-cache alias cannot hide a direct seed/private leak.
    np.testing.assert_array_equal(RosterAmendmentObserver(ExpandedAmendmentCatalogue()).raw(first),
                                  RosterAmendmentObserver(ExpandedAmendmentCatalogue()).raw(second))


def test_neither_seed_is_a_user_state_field_and_only_own_private_parameters_are_exposed(draw_factories):
    scenario,old,outside,factory=draw_factories
    case_id=encode_preference_episode_id(0,11)
    source=scenario(case_id)
    first=old(case_id)
    owner=first.current_user_id
    jobs=tuple(replace(job,beta_yuan_per_cu_hour=999.,migration_loss_yuan_per_cu=888.,
        options=tuple(replace(option,user_loss_yuan=option.user_loss_yuan+1e12) for option in job.options))
        if job.user_id!=owner else job for job in source.jobs)
    changed_others=FlexMarketEpisode(replace(source,config=first.config,jobs=jobs))
    np.testing.assert_array_equal(first.current_user_observation(),changed_others.current_user_observation())
    # Change both identity/provenance numbers while preserving all preferences
    # and physical data: neither party's input may contain those identifiers.
    spoof=replace(source,seed=encode_preference_episode_id(90000,90001),
        config=first.config,calibration=dict(source.calibration,public_seed=90000,preference_seed=90001))
    changed_ids=FlexMarketEpisode(spoof)
    np.testing.assert_array_equal(first.current_user_observation(),changed_ids.current_user_observation())
    np.testing.assert_array_equal(first.platform_observation(),changed_ids.platform_observation())
    # Genuine resampling does reach the current user through its own losses.
    another=old(encode_preference_episode_id(0,12))
    assert not np.array_equal(first.current_user_observation(),another.current_user_observation())
