"""Public task-roster information added to amendment quote observations.

The original observer aggregates each owner's alternatives across the entire
day. These additional fields retain the relationship between owner, accepted
location and time, plus public release/deadline information. No inconvenience
parameter, private loss, user action or user-policy prediction is inspected.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .amendment_features import AmendmentTariffObserver, feature_names as base_feature_names


def roster_feature_names():
    names = [f"roster_owner_allocation_fraction_u{owner}_r{region}_b{block}"
             for owner in range(3) for region in range(3) for block in range(4)]
    names.extend(f"roster_owner_release_work_fraction_u{owner}_b{block}"
                 for owner in range(3) for block in range(4))
    names.extend(f"roster_owner_remaining_deadline_slack_hours_u{owner}_b{block}"
                 for owner in range(3) for block in range(4))
    return tuple(names)


def feature_names():
    base = base_feature_names()
    return (*base[:-1], *roster_feature_names(), base[-1])


def public_roster_context(episode):
    """Return 60 public fields describing accepted, not proposed, executions.

    Allocation and release-work fractions are divided by each owner's total
    daily GPU-hours, already exposed by the base observer. Slack is grouped
    by the task's release block and weighted by task GPU-hours. It is the
    remaining margin between the accepted completion and public deadline:
    ``(deadline_step - accepted_option.end_step) * dt_hours``. Empty cohorts
    have zero work and zero slack. Allocation uses ``episode.jobs[*].options[0]``
    after outside bookings have been placed at index zero.
    """
    config = episode.config
    blocks = np.minimum(np.arange(config.n_steps) * config.n_blocks // config.n_steps,
                        config.n_blocks - 1)
    quantities = np.zeros(3)
    allocation = np.zeros((3, 3, 4))
    released = np.zeros((3, 4))
    slack_sum = np.zeros((3, 4))
    for job in episode.jobs:
        owner, work, accepted = int(job.user_id), float(job.work_cu), job.options[0]
        quantities[owner] += work
        for block in range(4):
            allocation[owner, :, block] += accepted.allocation_cu[blocks == block].sum(axis=0)
        release_block = min(int(job.release_step) * config.n_blocks // config.n_steps,
                            config.n_blocks - 1)
        released[owner, release_block] += work
        slack_sum[owner, release_block] += work * (
            int(job.deadline_step) - int(accepted.end_step)) * config.dt_hours
    allocation_fraction = np.divide(allocation, quantities[:, None, None],
        out=np.zeros_like(allocation), where=quantities[:, None, None] > 0)
    release_fraction = np.divide(released, quantities[:, None],
        out=np.zeros_like(released), where=quantities[:, None] > 0)
    slack_hours = np.divide(slack_sum, released, out=np.zeros_like(slack_sum), where=released > 0)
    result = np.r_[allocation_fraction.ravel(), release_fraction.ravel(), slack_hours.ravel()]
    if result.shape != (60,) or not np.isfinite(result).all():
        raise ValueError("Expected 60 finite public task-roster fields")
    return result.astype(np.float32)


def _public_roster_signature(episode):
    """Cheap identity of immutable arrays and public scalar booking metadata.

    Option allocations are immutable scenario arrays. Their identity permits
    cache reuse across the factory's fresh episode views without summing the
    task roster at every platform observation. Private-only replacements keep
    this signature unchanged; new public deadlines or allocations invalidate it.
    """
    return tuple((job.job_id, int(job.user_id), float(job.work_cu), int(job.release_step),
                  int(job.deadline_step), int(job.options[0].start_step),
                  int(job.options[0].end_step), id(job.options[0].allocation_cu))
                 for job in episode.jobs)


class RosterAmendmentObserver(AmendmentTariffObserver):
    """The unchanged 165-field observer plus 60 shared public roster fields.

    The first 164 columns retain their old meaning, the next 60 are common to
    every candidate quote, and the final unique-action mask stays last. The
    inherited forecast/normalization/curriculum API is preserved. Balanced
    full-action training uses the default maximum discount .03 (all current
    204 quotes), or calls with ``training=False``.
    """

    def __init__(self, catalogue=None, maxsize=512):
        super().__init__(catalogue=catalogue, maxsize=maxsize)
        self.feature_names = feature_names()
        self._roster_cache = OrderedDict()

    @staticmethod
    def _identity(episode):
        reference, config, signature = AmendmentTariffObserver._identity(episode)
        return reference, config, (signature, _public_roster_signature(episode))

    def raw(self, episode):
        key, (reference, config, signature) = int(episode.scenario.seed), self._identity(episode)
        entry = self._roster_cache.get(key)
        if entry is None or entry[0] is not reference or entry[1] is not config or entry[2] != signature:
            base = super().raw(episode)
            roster = public_roster_context(episode)
            shared = np.broadcast_to(roster, (self.catalogue.n_actions, len(roster)))
            values = np.concatenate((base[:, :-1], shared, base[:, -1:]), axis=1)
            if values.shape != (self.catalogue.n_actions, 225) or not np.isfinite(values).all():
                raise ValueError("Expected one 225-field public roster row per quote")
            entry = (reference, config, signature, values)
            self._roster_cache[key] = entry
            if len(self._roster_cache) > self.maxsize:
                self._roster_cache.popitem(last=False)
        self._roster_cache.move_to_end(key)
        return entry[3].copy()
