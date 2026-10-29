"""Fresh actors retain seeded Gym RNGs across PPO wrapper reconstruction."""
from __future__ import annotations

import torch

from energy_net.coordination.learning import (
    ActorEnv, PolicyBundle, alternating_train, cached_episode_factory,
)


class ScenarioTraceEnv(ActorEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scenario_trace = []

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.scenario_trace.append(info["scenario_seed"])
        return observation, info


def test_fresh_four_ppo_models_reproduce_days_and_parameters_after_rewrapping():
    """Exercise construction, set_env(force_reset=True), and real PPO learning.

    SB3 constructor seeds can remain pending on a temporary VecEnv. Replacing
    that wrapper must not leave the underlying scenario sampler OS-seeded.
    No initial_models are supplied: every actor follows the fresh-model path.
    """
    actors = ["platform", 0, 1, 2]
    factory = cached_episode_factory(maxsize=4)
    prototype = factory(0)
    experiment_seed = 37

    def train_fresh():
        bundle = PolicyBundle()
        envs = {
            actor: ScenarioTraceEnv(factory, bundle, actor, [0, 1, 2, 3], prototype)
            for actor in actors
        }
        try:
            models, records = alternating_train(
                bundle, envs, cycles=2, platform_days=8, user_steps=12,
                seed=experiment_seed,
                ppo_kwargs={"n_steps": 4, "batch_size": 4, "n_epochs": 1,
                            "policy_kwargs": {"net_arch": [8, 8]}},
            )
            traces = {actor: list(env.scenario_trace) for actor, env in envs.items()}
            sampler_seeds = {actor: env.np_random_seed for actor, env in envs.items()}
            parameters = {
                actor: {name: value.detach().cpu().clone()
                        for name, value in model.policy.state_dict().items()}
                for actor, model in models.items()
            }
            return traces, sampler_seeds, parameters, records
        finally:
            for env in envs.values():
                env.close()

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first = train_fresh()
        repeated = train_fresh()
    finally:
        torch.set_num_threads(old_threads)

    assert first[0] == repeated[0]
    expected = {actor: experiment_seed + index for index, actor in enumerate(actors)}
    assert first[1] == repeated[1] == expected
    # Multiple cases are actually sampled; a one-scenario factory would hide
    # the bug even if the policy weights happened to match.
    assert all(len(set(trace)) > 1 for trace in first[0].values())
    assert all(first[0][actor] for actor in actors)
    for actor in actors:
        assert first[2][actor].keys() == repeated[2][actor].keys()
        for name, value in first[2][actor].items():
            assert torch.equal(value, repeated[2][actor][name]), (actor, name)
    assert first[3] == repeated[3]
    assert all(record["actual_actor_steps"] == (8 if record["actor"] == "platform" else 12)
               for record in first[3])
