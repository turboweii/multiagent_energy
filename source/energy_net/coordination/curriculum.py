"""User PPO curriculum with varied posted prices and private utility rewards.

The reference current-task response is used only to measure held-out contract
selection quality. It supplies neither labels nor extra rewards during PPO.
"""
from __future__ import annotations

from collections.abc import Sequence
import time

import gymnasium as gym
import numpy as np

from .learning import ActorEnv, OptimizerStepCounter, PolicyBundle


class DiverseQuoteBundle(PolicyBundle):
    """Random daily posted prices, with frozen but sampled user opponents."""

    def __init__(self, catalogue, *, seed=0, users=None):
        super().__init__(users={} if users is None else dict(users))
        self.catalogue = catalogue
        self.quote_rng = np.random.default_rng(seed)

    def platform_action(self, episode, *args, **kwargs):
        return self.catalogue.sample(episode, self.quote_rng)

    def user_action(self, episode, *args, **kwargs):
        model = self.users.get(episode.current_user_id)
        if model is None:
            return 0
        action = model.predict(episode.current_user_observation(), deterministic=False)[0]
        return int(np.asarray(action).item())


def _selection_statistics(rows, episode_count):
    return {
        "tasks": len(rows),
        "regret_yuan_per_task": float(np.mean([r["current_task_regret_yuan"] for r in rows])),
        "minimizer_rate": float(np.mean([r["choice_is_cost_minimizer"] for r in rows])),
        "user_ir_rate": float(np.mean([r["individual_rational"] for r in rows])),
        "utility_gain_yuan_per_task": float(np.mean([r["utility_gain_yuan"] for r in rows])),
        "migrated_jobs_per_day": float(sum(r["migrated"] for r in rows) / episode_count),
        "delayed_jobs_per_day": float(sum(r["delay_hours"] > 0 for r in rows) / episode_count),
        "reservation_rejections_per_day": float(sum(r["requested"] != r["chosen"]
                                                     for r in rows) / episode_count),
    }


def evaluate_user_responses(bundle: PolicyBundle, episode_factory, seeds: Sequence[int],
                            *, quotes_per_day=4, seed=7919):
    """Evaluate deterministic user policies under reproducible diverse quotes.

    Returns ``summary``, ``per_user``, task-level ``tasks`` and market-level
    ``episodes``. Regret compares the chosen contract with the cheapest feasible
    contract in the *same current booking state*, not a joint task optimum.
    Each quote creates a fresh complete day, preserving all original contracts.
    """
    from .tariffs import TariffCatalogue

    seeds = tuple(int(value) for value in seeds)
    if not seeds or quotes_per_day < 1:
        raise ValueError("Provide scenarios and a positive quotes_per_day.")
    prototype = episode_factory(seeds[0])
    catalogue = TariffCatalogue(prototype.config)
    task_rows, episode_rows = [], []
    for scenario_seed in seeds:
        for quote_id in range(quotes_per_day):
            episode = episode_factory(scenario_seed)
            rng = np.random.default_rng(np.random.SeedSequence(
                [int(seed), scenario_seed, int(quote_id)]))
            episode.set_platform_action(catalogue.sample(episode, rng))
            published_action = episode.platform_action.tolist()
            while not episode.done:
                job = episode.jobs[episode.position]
                owner = episode.current_user_id
                observation = episode.current_user_observation()
                model = bundle.users.get(owner)
                requested = 0 if model is None else int(np.asarray(
                    model.predict(observation, deterministic=True)[0]).item())
                best_action = episode.best_response_action()
                best_option = job.options[best_action]
                best_cost = (episode.option_bill(job, best_option)
                             + best_option.user_loss_yuan)
                reward, _, info = episode.step_user(requested)
                actual_cost = info["bill_yuan"] + info["loss_yuan"]
                task_rows.append(dict(
                    scenario_seed=scenario_seed, quote_id=quote_id, **info,
                    current_task_best_action=best_action,
                    current_task_best_cost_yuan=float(best_cost),
                    generalized_cost_yuan=float(actual_cost),
                    current_task_regret_yuan=float(max(0.0, actual_cost - best_cost)),
                    choice_is_cost_minimizer=bool(actual_cost <= best_cost + 1e-6),
                    own_reward=float(reward),
                ))
            episode_rows.append(dict(
                episode.outcome().metrics, quote_id=quote_id,
                platform_action=published_action,
            ))
    count = len(episode_rows)
    summary = _selection_statistics(task_rows, count)
    summary.update(
        n_scenarios=len(seeds), quotes_per_day=int(quotes_per_day), episodes=count,
        completion_rate=float(np.mean([r["completion_rate"] for r in episode_rows])),
        deadline_compliance=float(np.mean([r["deadline_compliance"] for r in episode_rows])),
        mean_system_cost_yuan=float(np.mean([r["system_cost_yuan"] for r in episode_rows])),
        mean_profit_change_yuan=float(np.mean([r["delta_platform_profit_yuan"]
                                              for r in episode_rows])),
    )
    per_user = {
        owner: _selection_statistics([r for r in task_rows if r["user_id"] == owner], count)
        for owner in sorted({r["user_id"] for r in task_rows})
    }
    return {"summary": summary, "per_user": per_user,
            "tasks": task_rows, "episodes": episode_rows}


def pretrain_users(episode_factory, train_seeds: Sequence[int], *,
                   steps_per_user=32768, round_steps=2048, seed=0, users=None,
                   on_phase=None, validation_seeds=(), quotes_per_day=4):
    """Train three independent PPO users against diverse daily posted prices.

    Training keeps the market's original own-utility reward. Gamma zero makes
    this preparatory stage learn current-task comparison before joint learning.
    ``on_phase(record, models)`` receives actual transitions and optimizer steps;
    the final user phase of a round also includes optional validation summaries.
    The returned ordinary PolicyBundle contains the learned users and no random
    quote override, ready to attach to a learning platform.
    """
    from stable_baselines3 import PPO
    from .policies import ContractActorCriticPolicy
    from .tariffs import TariffCatalogue

    train_seeds = tuple(int(value) for value in train_seeds)
    validation_seeds = tuple(int(value) for value in validation_seeds)
    if not train_seeds or min(steps_per_user, round_steps) < 1:
        raise ValueError("Provide training scenarios and positive sample budgets.")
    prototype = episode_factory(train_seeds[0])
    catalogue = TariffCatalogue(prototype.config)
    bundle = DiverseQuoteBundle(catalogue, seed=seed + 301, users=users)
    actors = list(range(prototype.config.n_users))
    environments = {owner: ActorEnv(episode_factory, bundle, owner, train_seeds, prototype)
                    for owner in actors}
    # A reused PPO's set_env does not seed a fresh Gym environment. Initialise
    # only each environment's RNG here; do not reset global torch/NumPy streams
    # or advance bookings/opponent policies before the first actual rollout.
    for owner, env in environments.items():
        gym.Env.reset(env, seed=seed + owner + 1)
    models, counters, starts = {}, {}, {}
    for owner in actors:
        model = bundle.users.get(owner)
        if model is None:
            model = PPO(ContractActorCriticPolicy, environments[owner],
                        seed=seed + owner + 1, n_steps=128, batch_size=128,
                        n_epochs=6, learning_rate=3e-4, gamma=0.0, ent_coef=0.01,
                        device="cpu", verbose=0)
        else:
            model.gamma = 0.0
            model.rollout_buffer.gamma = 0.0
            model.set_env(environments[owner], force_reset=True)
        bundle.users[owner] = model
        models[owner] = model
        starts[owner] = model.num_timesteps
        counters[owner] = OptimizerStepCounter(model)
    records, round_id = [], 0
    started = time.monotonic()
    try:
        while any(models[owner].num_timesteps - starts[owner] < steps_per_user
                  for owner in actors):
            for owner in actors:
                model, env = models[owner], environments[owner]
                remaining = steps_per_user - (model.num_timesteps - starts[owner])
                if remaining <= 0:
                    continue
                requested = min(round_steps, remaining)
                previous_steps = model.num_timesteps
                previous_updates = counters[owner].count
                previous_days = env.completed_days
                model.set_env(env, force_reset=True)
                model.learn(total_timesteps=requested, reset_num_timesteps=False,
                            progress_bar=False)
                record = {
                    "stage": "diverse_quote_user_pretraining", "round": round_id,
                    "actor": str(owner), "seed": seed,
                    "requested_actor_steps": requested,
                    "actual_actor_steps": model.num_timesteps - previous_steps,
                    "stage_actor_steps": model.num_timesteps - starts[owner],
                    "cumulative_actor_steps": model.num_timesteps,
                    "optimizer_steps": counters[owner].count - previous_updates,
                    "stage_optimizer_steps": counters[owner].count,
                    "completed_actor_days": env.completed_days - previous_days,
                    "elapsed_seconds": time.monotonic() - started,
                }
                if owner == actors[-1] and validation_seeds:
                    result = evaluate_user_responses(
                        bundle, episode_factory, validation_seeds,
                        quotes_per_day=quotes_per_day, seed=seed + 7919)
                    record["validation"] = {
                        "summary": result["summary"], "per_user": result["per_user"]}
                records.append(record)
                if on_phase is not None:
                    on_phase(record, models)
            round_id += 1
    finally:
        for counter in counters.values():
            counter.close()
        for env in environments.values():
            env.close()
    return PolicyBundle(users=dict(models)), records
