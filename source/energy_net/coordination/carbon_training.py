"""Public, realized carbon signals alongside the unchanged posted-offer cash.

The expected-profit constraint belongs to the learning algorithm, not to the
market.  In particular, neither its dual weight nor a hypothetical carbon
payment enters a bill, the funding screen, the observation, or the Gym reward.
"""
from dataclasses import dataclass
import math

from .posted_offer_learning import PostedOfferEnv


@dataclass(frozen=True)
class CarbonConstraintConfig:
    """Training units and a constant, externally fitted profit reference.

    The reference is the TRAIN mean *incremental cash profit* of the reference
    policy.  The constraint is E[profit_gain_yuan] >= q * reference, rather than
    a per-day participation test.  The original contract's absolute profit
    floor and every market admission rule continue to be enforced by the
    unchanged posted-offer episode.

    ``lambda_init`` initializes a training-only nonnegative dual variable.
    ``profit_scale`` normalizes its profit target; the inherited Gym reward
    remains the existing cash increment in thousands of yuan.
    """

    q: float = .95
    profit_scale: float = 1000.
    emission_scale_t: float = 1.
    lambda_init: float = 1.
    profit_reference_yuan: float | None = None

    def __post_init__(self):
        for name in ("q", "profit_scale", "emission_scale_t", "lambda_init"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0. <= self.q <= 1.:
            raise ValueError("q must lie between zero and one")
        if self.profit_scale <= 0. or self.emission_scale_t <= 0.:
            raise ValueError("Training scales must be positive")
        if self.lambda_init < 0.:
            raise ValueError("The training dual weight must be nonnegative")
        if self.profit_reference_yuan is not None:
            if (not math.isfinite(self.profit_reference_yuan)
                    or self.profit_reference_yuan < 0.):
                raise ValueError("TRAIN incremental cash profit reference must be finite and nonnegative")

    @property
    def expected_profit_threshold_yuan(self):
        """Constant expected constraint, or None before TRAIN calibration."""
        if self.profit_reference_yuan is None:
            return None
        return self.q * self.profit_reference_yuan

    @property
    def normalized_profit_threshold(self):
        """The same expected threshold in the trainer's profit units."""
        threshold = self.expected_profit_threshold_yuan
        return None if threshold is None else threshold / self.profit_scale


class CarbonOfferEnv(PostedOfferEnv):
    """An additional realized-emission channel for the platform's two critics.

    ``step`` still returns the original cash reward, already divided by 1000
    in :class:`PostedOfferEnv`.  ``info['carbon_reduction_reward']`` is signed
    whole-grid OPF emission reduction in ``emission_scale_t`` units.  It is
    measured across the actual user response, never from an unaccepted offer
    or from a data centre's average consumption footprint.  With gamma=1 the
    carbon channel telescopes to outside-grid emissions minus final emissions.

    Platform info contains only realized public feedback, cash and emissions,
    including at termination.  Full outcome accounts remain in ``last_outcome``
    for separate evaluation.  User environments retain the original rewards,
    observations, stepping and info without any carbon channel.
    """

    def __init__(self, episode_factory, bundle, actor, scenario_seeds,
                 prototype=None, *, carbon_config=None):
        self.carbon_config = carbon_config or CarbonConstraintConfig()
        super().__init__(episode_factory, bundle, actor, scenario_seeds, prototype)
        self._carbon_initial_emissions_t = None
        self._carbon_daily_profit_gain_yuan = 0.

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        if self.actor == "platform":
            self._carbon_initial_emissions_t = float(self.episode.current_clearing.emissions_t)
            self._carbon_daily_profit_gain_yuan = 0.
        return observation, info

    def step(self, action):
        if self.actor != "platform":
            return super().step(action)
        if self.episode is None or self._finished:
            raise RuntimeError("Reset before stepping")

        before = float(self.episode.current_clearing.emissions_t)
        observation, cash_reward, terminated, truncated, base_info = super().step(action)
        after = float(self.episode.current_clearing.emissions_t)
        reduction_t = before - after
        profit_gain_yuan = float(cash_reward) * 1000.
        self._carbon_daily_profit_gain_yuan += profit_gain_yuan

        # Select public fields explicitly: the parent adds full evaluation
        # accounts at termination, and those can include private user losses.
        info = {
            "public_feedback": base_info["public_feedback"],
            "platform_step_gain": base_info["platform_step_gain"],
            "profit_gain_yuan": profit_gain_yuan,
            "grid_emissions_t_before": before,
            "grid_emissions_t_after": after,
            "grid_emissions_reduction_t": reduction_t,
            "carbon_reduction_reward": reduction_t / self.carbon_config.emission_scale_t,
        }
        if terminated:
            info["carbon_episode"] = {
                "profit_gain_yuan": self._carbon_daily_profit_gain_yuan,
                "grid_emissions_reduction_t": self._carbon_initial_emissions_t - after,
            }
        return observation, cash_reward, terminated, truncated, info
