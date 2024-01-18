import copy
import operator
from collections import deque
from typing import Callable

import gymnasium as gym

from sample_factory.algo.utils.misc import EPS
from sample_factory.envs.env_utils import RewardShapingInterface
from sample_factory.utils.utils import log

REWARD_SHAPING = dict(
    delta=dict(
        MONSTER_SCORE=(1, -1),
        HEALTH=(),
        WAND_IDENTIFY=(0.5, -0.5),
        POTION_IDENTIFY=(0.5, -0.5),
        SCROLL_IDENTIFY=(0.5, -0.5),
        DLVL_REACHED=(0.5, -0.5),
        DEEP_DLVL_REACHED=(0.5, -0.5),
        GOLD_SCORE=(0.01, -0.01),
        EATING_SCORE=(0.01, -0.01),
        SCOUT_SCORE=(0.01, -0.01),
    )
)


def true_objective_winning_the_game(info):
    # TODO: for now define true objective as game score
    # but we also could test reaching as deep into the dungeon as possible
    # eventually solving the game
    pass


class NetHackRewardShapingWrapper(gym.Wrapper, RewardShapingInterface):
    """Convert game info variables into scalar reward using a reward shaping scheme."""

    def __init__(self, env, reward_shaping_scheme=None, true_objective_func=None):
        gym.Wrapper.__init__(self, env)
        RewardShapingInterface.__init__(self)

        self.reward_shaping_scheme = reward_shaping_scheme
        self.true_objective_func: Callable = true_objective_func

        # TODO: do we need this?
        self.reward_delta_limits = dict()

        self.prev_vars = dict()
        self.prev_dead = True

        self.orig_env_reward = self.total_shaping_reward = 0.0

        self.selected_weapon = deque([], maxlen=5)

        self.reward_structure = {}

        self.verbose = False
        self.print_once = False

        # save a reference to this wrapper in the actual env class, for other wrappers
        self.env.unwrapped.reward_shaping_interface = self

    def get_default_reward_shaping(self):
        return self.reward_shaping_scheme

    def set_reward_shaping(self, reward_shaping: dict, agent_idx: int):
        self.reward_shaping_scheme = reward_shaping

    def _delta_rewards(self, info):
        # TODO:
        pass

    def _parse_info(self, info, done):
        # TODO:
        pass

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.prev_vars = dict()
        self.prev_dead = True
        self.reward_structure = dict()

        self.orig_env_reward = self.total_shaping_reward = 0.0

        self.print_once = False
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if obs is None:
            return obs, rew, terminated, truncated, info

        done = terminated | truncated

        self.orig_env_reward += rew

        shaping_rew = self._parse_info(info, done)

    def close(self):
        self.env.unwrapped.reward_shaping_interface = None
        return self.env.close()
