import copy
import operator
from collections import deque
from typing import Callable

import gymnasium as gym

from sample_factory.algo.utils.misc import EPS
from sample_factory.envs.env_utils import RewardShapingInterface
from sample_factory.utils.utils import log
from sf_examples.nethack.utils.blstats import BLStats

GAME_REWARD = dict(
    delta=dict(
        KILL_MONSTER=(1, -1),
        IDENTIFY=(1, -1),
        EAT_TRIPE_RATION=(1, -1),
        QUAFFLE_FROM_SINK=(1, -1),
        READ_NOVEL=(1, -1),
        REACH_ORACLE=(1, -1),
        LEVEL_REACHED=(1, -1),
        DEEP_LEVEL_REACHED=(1, -1),
        OTHER=(1, -1),
    )
)

REWARD_SHAPING = dict(
    delta=dict(
        STRENGTH=(100, -100),
        DEXTERITY=(100, -100),
        CONSTITUTION=(100, -100),
        INTELLIGENCE=(100, -100),
        WISDOM=(100, -100),
        CHARISMA=(100, -100),
        HITPOINTS=(1, -1),
        MAX_HITPOINTS=(10, -10),
        GOLD=(1, -0.5),
        ENERGY=(0.01, -0.01),
        MAX_ENERGY=(10, -10),
        ARMOR_CLASS=(100, -100),
        EXPERIENCE_LEVEL=(100, -100),
        EXPERIENCE_POINTS=(0.01, -0.01),
        KILL_MONSTER=(1, -1),
        IDENTIFY=(1, -1),
        EAT_TRIPE_RATION=(1, -1),
        QUAFFLE_FROM_SINK=(1, -1),
        READ_NOVEL=(1, -1),
        REACH_ORACLE=(1, -1),
        LEVEL_REACHED=(1, -1),
        DEEP_LEVEL_REACHED=(1, -1),
    )
)


def true_objective_winning_the_game(info):
    # TODO: for now define true objective as game score
    # but we also could test reaching as deep into the dungeon as possible
    # eventually solving the game
    return info["true_objective"]


class NetHackRewardShapingWrapper(gym.Wrapper, RewardShapingInterface):
    """Convert game info variables into scalar reward using a reward shaping scheme."""

    def __init__(self, env, reward_shaping_scheme=None, true_objective_func=None):
        gym.Wrapper.__init__(self, env)
        RewardShapingInterface.__init__(self)

        self.reward_shaping_scheme = reward_shaping_scheme
        self.true_objective_func: Callable = true_objective_func

        # TODO: do we need this? we could use this to cap the monster scores
        # at some point to encourage getting score for dungeon depth
        self.reward_delta_limits = dict()

        self.prev_vars = dict()
        self.sokoban_level = None

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
        reward = 0.0
        deltas = []

        for var_name, rewards in self.reward_shaping_scheme["delta"].items():
            if var_name not in self.prev_vars:
                continue

            # generate reward based on how the env variable values changed
            new_value = info.get(var_name, 0.0)
            prev_value = self.prev_vars[var_name]
            delta = new_value - prev_value

            if var_name in self.reward_delta_limits:
                delta = min(delta, self.reward_delta_limits[var_name])

            if abs(delta) > EPS:
                if delta > EPS:
                    reward_delta = delta * rewards[0]
                else:
                    reward_delta = -delta * rewards[1]

                reward += reward_delta
                deltas.append((var_name, reward_delta, delta))
                self.reward_structure[var_name] = self.reward_structure.get(var_name, 0.0) + reward_delta

        return reward, deltas

    def _monster_kill_reward(self, blstats):
        dungeon_level = blstats.depth

        # check if we are in sokoban
        # we have to calculate dlvl differently in sokoban https://nethackwiki.com/wiki/Dungeon_level
        if blstats.dungeon_number == 4:
            if self.sokoban_level is None:
                # save the level when sokoban first appeared
                self.sokoban_level = dungeon_level
            # set the difficulty to custom sokoban
            dungeon_level = 2 * self.sokoban_level - dungeon_level + 2

        return 4 * dungeon_level**2

    def _parse_info(self, info, done):
        if self.reward_shaping_scheme is None:
            # skip reward calculation
            return 0.0

        shaping_reward = 0.0
        if not done:
            shaping_reward, deltas = self._delta_rewards(info)

            # change the reward for killing a monster
            for var_name, reward_delta, delta in deltas:
                if var_name == "KILL_MONSTER":
                    shaping_reward -= reward_delta

                    blstats = BLStats(*self.env.unwrapped.last_observation[self.env.unwrapped._blstats_index])
                    shaping_reward += self._monster_kill_reward(blstats)

        return shaping_reward

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.prev_vars = dict()
        self.sokoban_level = None
        self.reward_structure = dict()

        self.orig_env_reward = self.total_shaping_reward = 0.0

        self.print_once = False
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        done = terminated | truncated

        self.orig_env_reward += rew

        shaping_rew = self._parse_info(info, done)
        # IMPORTANT: we use shaping reward as a substitue for reward
        rew = shaping_rew
        self.total_shaping_reward += shaping_rew

        if self.verbose:
            log.info("Original env reward before shaping: %.3f", self.orig_env_reward)

            log.info(
                "Total shaping reward is %.3f (done %d)",
                self.total_shaping_reward,
                done,
            )

        # remember new variable values
        for var_name in self.reward_shaping_scheme["delta"].keys():
            self.prev_vars[var_name] = info.get(var_name, 0.0)

        if done:
            if self.true_objective_func is None:
                true_objective = self.orig_env_reward
            else:
                true_objective = self.true_objective_func(info)

            info["true_objective"] = true_objective

        return obs, rew, terminated, truncated, info

    def close(self):
        self.env.unwrapped.reward_shaping_interface = None
        return self.env.close()
