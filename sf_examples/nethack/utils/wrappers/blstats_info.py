from collections import namedtuple

import gymnasium as gym

from sf_examples.nethack.utils.blstats import BLStats


class BlstatsInfoWrapper(gym.Wrapper):
    def step(self, action):
        # because we will see done=True at the first timestep of the new episode
        # to properly calculate blstats at the end of the episode we need to keep the last_observation around
        last_observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated | truncated:
            info["episode_extra_stats"] = self.add_more_stats(info, last_observation)

        return obs, reward, terminated, truncated, info

    def add_more_stats(self, info, last_observation):
        extra_stats = info.get("episode_extra_stats", {})
        blstats = BLStats(*last_observation[self.env.unwrapped._blstats_index])
        new_extra_stats = {
            "score": blstats.score,
            "turns": blstats.time,
            "dlvl": blstats.depth,
            "max_hitpoints": blstats.max_hitpoints,
            "max_energy": blstats.max_energy,
            "armor_class": blstats.armor_class,
            "experience_level": blstats.experience_level,
            "experience_points": blstats.experience_points,
        }
        return {**extra_stats, **new_extra_stats}
