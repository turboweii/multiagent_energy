"""Sequential PPO interfaces for a binding, privately accepted offer market."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .cached_grid import MemoizedGridClearing
from .flex_market import FlexMarketEpisode
from .learning import PolicyBundle
from .legacy_reference import load_legacy_bundle
from .outside_bookings import OutsideBooking
from .posted_offer_features import PostedOfferCatalogue, PostedOfferObserver
from .posted_offer_market import PostedOfferEpisode
from .preference_scenario import (clone_with_preferences, decode_preference_episode_id,
                                 encode_preference_episode_id)
from .varied_scenario import VariedMarketConfig, make_varied_scenario


def preference_cases(public_seeds, draws=8, namespace=100000):
    """Independent private IDs per public-day/draw, excluded from observations."""
    return tuple(encode_preference_episode_id(int(public), namespace+i*draws+draw)
                 for i, public in enumerate(public_seeds) for draw in range(draws))


def posted_episode_factory(legacy_path, config=None, *, public_cache_size=256,
                           case_cache_size=2048, grid_cache_size=2048):
    config=config or VariedMarketConfig()
    legacy=load_legacy_bundle(legacy_path)
    legacy_config=replace(config, conditional_rebates=legacy.legacy_config.conditional_rebates,
        sharing_enabled=False,carbon_eligibility=False,internal_carbon_value_yuan_per_t=0.)

    @lru_cache(maxsize=public_cache_size)
    def public_day(seed):return make_varied_scenario(int(seed),config)

    @lru_cache(maxsize=public_cache_size)
    def grid(seed):return MemoizedGridClearing(public_day(seed).grid_spec,maxsize=grid_cache_size)

    @lru_cache(maxsize=public_cache_size)
    def old_public_quote(seed):
        # Old platform pricing has no preference input; only its actual old
        # users respond to each independent private draw below.
        episode=FlexMarketEpisode(replace(public_day(seed),config=legacy_config),grid=grid(seed))
        return tuple(int(x) for x in legacy.platform_action(episode))

    @lru_cache(maxsize=case_cache_size)
    def scenario(case):
        public,private=decode_preference_episode_id(int(case))
        return clone_with_preferences(public_day(public),private)

    @lru_cache(maxsize=case_cache_size)
    def booking(case):
        public,_=decode_preference_episode_id(int(case))
        episode=FlexMarketEpisode(replace(scenario(case),config=legacy_config),grid=grid(public))
        episode.reset(old_public_quote(public))
        while not episode.done:episode.step_user(legacy.user_action(episode))
        return OutsideBooking.from_episode(episode,'frozen_legacy_on_independent_private_draw')

    def factory(case):
        public,_=decode_preference_episode_id(int(case))
        return PostedOfferEpisode(scenario(case),booking(case),grid=grid(public),config=config)
    factory.config=config
    factory.public_day=public_day
    factory.scenario=scenario
    factory.booking=booking
    factory.grid=grid
    return factory


class PostedPolicyBundle(PolicyBundle):
    """A platform controller or PPO and three live, independent user PPOs."""
    def platform_action(self, episode):
        if self.platform is None:
            return self.platform_codec.sample(episode,self.exploration_rng)
        if hasattr(self.platform,'choose'):
            return int(self.platform.choose(episode))
        return super().platform_action(episode)


class PostedOfferEnv(gym.Env):
    """Platform: one offered task/step. User: own next task/step.

    Both receive their actual cash/utility increment in thousands of yuan.
    With gamma=1, their episode returns are exactly their daily gains. Other
    agents remain frozen while the selected PPO is updated.
    """
    metadata={'render_modes':[]}

    def __init__(self,episode_factory,bundle,actor,scenario_seeds,prototype=None):
        super().__init__()
        self.episode_factory=episode_factory;self.bundle=bundle;self.actor=actor
        self.scenario_seeds=tuple(int(x) for x in scenario_seeds)
        if not self.scenario_seeds:raise ValueError('Provide public/private training cases')
        prototype=prototype or episode_factory(self.scenario_seeds[0])
        if actor=='platform':
            self.action_space=spaces.Discrete(61)
            sample=bundle.platform_observation(prototype)
        else:
            if actor not in (0,1,2):raise ValueError('Three private owners or the platform')
            self.action_space=spaces.Discrete(2);sample=prototype.current_user_observation()
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=sample.shape,dtype=np.float32)
        self.completed_days=0;self.actor_steps=0;self.last_outcome=None
        self.episode=None;self._finished=True;self.feedback_records=[]

    def _respond(self,action):
        _,done,info=self.episode.step_accept(action)
        self.feedback_records.append(dict(info['public_feedback']))
        return done,info

    def _advance(self):
        while not self.episode.done:
            self.episode.set_offer(self.bundle.platform_action(self.episode))
            if self.episode.current_user_id==self.actor:return
            self._respond(self.bundle.user_action(self.episode))

    def _observation(self):
        if self.episode.done:return np.zeros(self.observation_space.shape,dtype=np.float32)
        if self.actor=='platform':return self.bundle.platform_observation(self.episode)
        return self.episode.current_user_observation()

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        case=int(self.np_random.choice(self.scenario_seeds))
        self.episode=self.episode_factory(case);self.episode.reset();self._finished=False
        if self.actor!='platform':self._advance()
        return self._observation(),{'case_id':case,'actor':str(self.actor)}

    def step(self,action):
        if self.episode is None or self._finished:raise RuntimeError('Reset before stepping')
        if self.actor=='platform':
            self.episode.set_offer(action)
            _,info=self._respond(self.bundle.user_action(self.episode))
            reward=float(info['platform_step_gain'])
            # The selected platform receives the public response and its own
            # cash, never a user's private loss or reservation reason.
            info={'public_feedback':info['public_feedback'],'platform_step_gain':reward}
        else:
            if self.episode.current_user_id!=self.actor:raise RuntimeError('Wrong responding owner')
            _,info=self._respond(action)
            reward=float(info['utility_gain_yuan']/1000.)
            self._advance()
        self.actor_steps+=1;terminated=self.episode.done
        if terminated:
            self.completed_days+=1;self._finished=True
            self.last_outcome=self.episode.outcome().to_dict()
            # Full accounts are for logging/evaluation and are never policy
            # observations or actor rewards.
            info=dict(info,market_metrics=self.last_outcome)
        return self._observation(),reward,bool(terminated),False,info
