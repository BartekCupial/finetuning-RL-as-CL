from collections import namedtuple

import gymnasium as gym

from sf_examples.nethack.utils.blstats import BLStats


class TtyrecInfoWrapper(gym.Wrapper):
    def step(self, action):
        ttyrec = self.env.unwrapped.gym_env.nethack._ttyrec
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated | truncated:
            info["episode_extra_stats"]["ttyrec"] = ttyrec

        return obs, reward, terminated, truncated, info
