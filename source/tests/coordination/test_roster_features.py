"""Owner/location/time correlations remain public and cached."""
from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from energy_net.coordination import amendment_features, roster_features
from energy_net.coordination.amendment_features import AmendmentTariffCatalogue, AmendmentTariffObserver
from energy_net.coordination.amendment_market import AmendmentConfig, amendment_episode_factory
from energy_net.coordination.capped_market import capped_episode_factory
from energy_net.coordination.dynamic_tariffs import ExpandedAmendmentCatalogue
from energy_net.coordination.gpu_tariffs import FixedGpuTariffCatalogue
from energy_net.coordination.outside_bookings import OutsideBooking
from energy_net.coordination.roster_features import (
    RosterAmendmentObserver, feature_names, public_roster_context, roster_feature_names,
)


def public_job(name, owner, work, region, start, release, deadline):
    allocation = np.zeros((48,3))
    allocation[start:start+4,region] = work/4
    option = SimpleNamespace(allocation_cu=allocation,start_step=start,end_step=start+4)
    return SimpleNamespace(job_id=name,user_id=owner,work_cu=work,release_step=release,
                           deadline_step=deadline,options=(option,))


def test_roster_units_denominators_release_groups_and_remaining_booked_slack():
    config = AmendmentConfig()
    jobs = (public_job('a',0,100,1,2,0,12),public_job('b',0,300,2,16,12,28),
            public_job('c',1,200,0,26,24,32))
    values = public_roster_context(SimpleNamespace(config=config,jobs=jobs))
    allocations,releases,slacks = values[:36].reshape(3,3,4),values[36:48].reshape(3,4),values[48:].reshape(3,4)
    expected = np.zeros((3,3,4));expected[0,1,0]=.25;expected[0,2,1]=.75;expected[1,0,2]=1
    np.testing.assert_array_equal(allocations,expected)
    np.testing.assert_array_equal(releases,[[.25,.75,0,0],[0,0,1,0],[0,0,0,0]])
    np.testing.assert_array_equal(slacks,[[3,4,0,0],[0,0,1,0],[0,0,0,0]])
    np.testing.assert_array_equal(allocations.sum(axis=(1,2)),[1,1,0])
    np.testing.assert_array_equal(releases.sum(axis=1),[1,1,0])


def test_deadline_slack_is_work_weighted_within_a_release_cohort():
    jobs = (public_job('a',0,100,0,2,0,12),public_job('b',0,300,1,4,2,10))
    result = public_roster_context(SimpleNamespace(config=AmendmentConfig(),jobs=jobs))
    assert result[48] == pytest.approx((100*3+300*1)/400)


def test_owner_region_time_correlation_survives_equal_daily_owner_and_grid_totals():
    config=AmendmentConfig()
    first=(public_job('a',0,100,0,0,0,12),public_job('b',0,100,1,12,12,24),
           public_job('c',1,100,1,0,0,12),public_job('d',1,100,0,12,12,24))
    second=(public_job('a',0,100,1,0,0,12),public_job('b',0,100,0,12,12,24),
            public_job('c',1,100,0,0,0,12),public_job('d',1,100,1,12,12,24))
    np.testing.assert_array_equal(sum(j.options[0].allocation_cu for j in first),
                                  sum(j.options[0].allocation_cu for j in second))
    assert [np.mean([j.options[0].start_step for j in first if j.user_id==u]) for u in (0,1)] == [
        np.mean([j.options[0].start_step for j in second if j.user_id==u]) for u in (0,1)]
    a=public_roster_context(SimpleNamespace(config=config,jobs=first))
    b=public_roster_context(SimpleNamespace(config=config,jobs=second))
    assert not np.array_equal(a[:36],b[:36])
    np.testing.assert_array_equal(a[36:],b[36:])


@pytest.fixture(scope='module')
def factory():
    config=AmendmentConfig(jobs_per_user=1,batch_release_steps=(13,))
    old=capped_episode_factory(config)(4)
    old.set_platform_action(FixedGpuTariffCatalogue(config).decode(old,8))
    # A controlled accepted-contract fixture, not an evaluation-time oracle.
    while not old.done:
        old.step_user(old.best_response_action())
    booking=OutsideBooking.from_episode(old,'fixed_test_quote')
    return amendment_episode_factory(lambda seed:booking.scenario,lambda seed:booking.copy(),config=config)


@pytest.mark.parametrize('expanded',[False,True])
def test_existing_columns_are_unchanged_unique_mask_last_and_roster_shared(factory,expanded):
    episode=factory(4)
    catalogue=(ExpandedAmendmentCatalogue if expanded else AmendmentTariffCatalogue)(episode.config)
    base=AmendmentTariffObserver(catalogue).raw(episode)
    observer=RosterAmendmentObserver(catalogue)
    actual=observer(episode,training=False)
    assert actual.shape==(catalogue.n_actions,225)
    assert len(set(feature_names()))==len(feature_names())==225
    assert len(roster_feature_names())==60
    assert observer.feature_names==feature_names() and feature_names()[-1]=='unique_raw_action'
    np.testing.assert_array_equal(actual[:,:164],base[:,:164])
    np.testing.assert_array_equal(actual[:,-1],base[:,-1])
    np.testing.assert_array_equal(actual[:,164:224],
                                  np.broadcast_to(public_roster_context(episode),(catalogue.n_actions,60)))
    np.testing.assert_array_equal(observer(episode,training=True)[:,-1],np.ones(catalogue.n_actions))
    mean,scale=observer.fit_normalization(factory,[4])
    assert mean.shape==scale.shape==(224,) and np.isfinite(mean).all() and np.all(scale>=.01)


def test_roster_uses_actual_accepted_schedule_rather_than_original_no_shift_jobs(factory):
    episode=factory(4)
    assert any(job.options[0].start_step>job.release_step for job in episode.jobs)
    actual=public_roster_context(episode)
    original=public_roster_context(SimpleNamespace(config=episode.config,jobs=episode.original_scenario.jobs))
    assert np.any(actual[48:]<original[48:])
    for job in episode.jobs:
        block=job.release_step*4//48
        assert actual[48+job.user_id*4+block]==pytest.approx((job.deadline_step-job.options[0].end_step)*.5)


def private_changed(episode):
    other=copy(episode)
    other.jobs=tuple(replace(job,beta_yuan_per_cu_hour=99999.,migration_loss_yuan_per_cu=88888.,
        options=tuple(replace(option,user_loss_yuan=option.user_loss_yuan+1e20)
                      for option in job.options)) for job in episode.jobs)
    other.scenario=replace(episode.scenario,jobs=other.jobs)
    other.outside_losses=np.full(len(other.jobs),-1e30)
    other.booking=replace(episode.booking,metrics=dict(episode.booking.metrics,
        user_loss_yuan=1e30,user_gain_yuan=-1e30,system_cost_yuan=1e30))
    def forbidden(*args,**kwargs):
        raise AssertionError('Public observer must not run private decisions or final utilities')
    other.best_response_action=other.current_user_observation=other.outcome=forbidden
    return other


def test_private_preferences_losses_accounts_and_user_policies_do_not_enter_state(factory):
    episode=factory(4);other=private_changed(episode)
    catalogue=ExpandedAmendmentCatalogue(episode.config)
    np.testing.assert_array_equal(public_roster_context(episode),public_roster_context(other))
    # Separate observers avoid allowing a cache hit to conceal a privacy error.
    np.testing.assert_array_equal(RosterAmendmentObserver(catalogue).raw(episode),
                                  RosterAmendmentObserver(catalogue).raw(other))


def test_cache_reuses_roster_and_opfs_for_fresh_views_but_invalidates_public_deadlines(factory,monkeypatch):
    calls={'roster':0,'forecasts':0}
    roster=roster_features.public_roster_context
    forecasts=amendment_features.public_amendment_forecasts
    def count_roster(episode):
        calls['roster']+=1;return roster(episode)
    def count_forecasts(episode):
        calls['forecasts']+=1;return forecasts(episode)
    monkeypatch.setattr(roster_features,'public_roster_context',count_roster)
    monkeypatch.setattr(amendment_features,'public_amendment_forecasts',count_forecasts)
    observer=RosterAmendmentObserver(ExpandedAmendmentCatalogue())
    first,second=factory(4),factory(4)
    original=observer.raw(first)
    np.testing.assert_array_equal(observer.raw(second),original)
    np.testing.assert_array_equal(observer.raw(private_changed(second)),original)
    assert calls=={'roster':1,'forecasts':1}
    other=copy(second)
    other.jobs=(replace(second.jobs[0],deadline_step=second.jobs[0].deadline_step+2),*second.jobs[1:])
    other.scenario=replace(second.scenario,jobs=other.jobs)
    changed=observer.raw(other)
    assert calls=={'roster':2,'forecasts':2}
    owner=second.jobs[0].user_id;block=second.jobs[0].release_step*4//48
    assert changed[0,164+48+owner*4+block]==pytest.approx(original[0,164+48+owner*4+block]+1)
    changed[:]=0
    assert np.any(observer.raw(other)!=0)
