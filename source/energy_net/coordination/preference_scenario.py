"""Independent private preference draws on an unchanged public physical day.

Public weather, workload, origins, releases and the original carbon allowance
come from the existing generator. This adapter changes only owner-level
inconvenience coefficients and the corresponding private option losses.
Episode IDs are internal cache addresses, never observation features.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from math import isqrt
from operator import index

import numpy as np

from .scenario import Scenario
from .varied_scenario import make_varied_scenario


PREFERENCE_MULTIPLIER_BOUNDS = (.8, 1.2)
PREFERENCE_STREAM_NAMESPACE = 9610429


def _integer(value, name):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        return int(index(value))
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _seed(value, name):
    value = _integer(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def encode_preference_episode_id(public_seed, preference_seed):
    """Injectively pair two nonnegative seeds in a negative integer namespace.

    All existing public generator seeds are nonnegative. Negative Cantor IDs
    cannot collide with those original cases and remain compatible with the
    existing integer-keyed episode, quote and response caches. They must be
    decoded before invoking a physical generator or seeding a NumPy RNG.
    """
    public = _seed(public_seed, "public_seed")
    private = _seed(preference_seed, "preference_seed")
    diagonal = public + private
    return -(diagonal * (diagonal + 1) // 2 + private + 1)


def decode_preference_episode_id(episode_id):
    """Recover ``(public_seed, preference_seed)`` using exact integer arithmetic."""
    episode_id = _integer(episode_id, "episode_id")
    if episode_id >= 0:
        raise ValueError("A preference episode ID must be negative")
    paired = -episode_id - 1
    diagonal = (isqrt(8 * paired + 1) - 1) // 2
    private = paired - diagonal * (diagonal + 1) // 2
    public = diagonal - private
    return int(public), int(private)


def clone_with_preferences(existing: Scenario, preference_seed):
    """Redraw private willingness while preserving every physical input.

    Each owner's beta and kappa are independent uniform multipliers of that
    owner's configured low/medium/high coefficients. All tasks of one owner
    share that pair for the complete day. The preference RNG depends only on
    the preference seed, not on public inputs or the generator's old private
    coefficients. Callers should allocate globally independent preference
    IDs to each public-day/draw pair rather than reuse four IDs for all days.

    Physical arrays, grid specification and allowance are reused unchanged;
    no dispatch is solved or allowance recomputed. Frozen Job/ExecutionOption
    objects are replaced, and calibration metadata is copied. Preference seed
    metadata is internal simulation provenance, excluded from platform and
    user observations. Accepted legacy contracts must subsequently be formed
    by the actual old policies on this new draw.
    """
    private_seed = _seed(preference_seed, "preference_seed")
    source_id = _integer(existing.seed, "existing scenario seed")
    public_seed = (decode_preference_episode_id(source_id)[0] if source_id < 0
                   else _seed(source_id, "public_seed"))
    config = existing.config
    rng = np.random.default_rng(np.random.SeedSequence(
        [private_seed, PREFERENCE_STREAM_NAMESPACE]))
    parameters = tuple((float(config.beta_yuan_per_cu_hour[owner]
                              * rng.uniform(*PREFERENCE_MULTIPLIER_BOUNDS)),
                        float(config.migration_loss_yuan_per_cu[owner]
                              * rng.uniform(*PREFERENCE_MULTIPLIER_BOUNDS)))
                       for owner in range(config.n_users))
    jobs = []
    for job in existing.jobs:
        beta, kappa = parameters[job.user_id]
        options = tuple(replace(option, user_loss_yuan=float(job.work_cu * (
            beta * option.delay_hours + kappa * bool(option.migrated))))
            for option in job.options)
        jobs.append(replace(job, beta_yuan_per_cu_hour=beta,
                            migration_loss_yuan_per_cu=kappa, options=options))
    episode_id = encode_preference_episode_id(public_seed, private_seed)
    metadata = deepcopy(existing.calibration)
    metadata.update(
        public_seed=public_seed,
        preference_seed=private_seed,
        preference_episode_id=episode_id,
        preference_episode_id_encoding="negative Cantor pairing of public and independent preference seeds",
        preference_multiplier_bounds=list(PREFERENCE_MULTIPLIER_BOUNDS),
        preference_sampling="independent owner beta/kappa draws; same owner shares its pair across all daily tasks; physical day unchanged",
        preference_metadata_visibility="internal simulation provenance; neither seed is an observation feature",
    )
    return replace(existing, seed=episode_id, jobs=tuple(jobs), calibration=metadata)


def make_preference_scenario(public_seed=0, preference_seed=0, config=None):
    """Generate the original varied physical day, then apply an independent draw.

    For repeated preference draws, cache ``make_varied_scenario(public_seed)``
    and use ``clone_with_preferences`` to avoid regenerating public physics.
    A factory receiving composite IDs can decode them with the helpers above.
    """
    public_seed = _seed(public_seed, "public_seed")
    private_seed = _seed(preference_seed, "preference_seed")
    return clone_with_preferences(make_varied_scenario(public_seed, config), private_seed)
