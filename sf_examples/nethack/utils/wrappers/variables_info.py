import gymnasium as gym

from sf_examples.nethack.utils.blstats import BLStats
from sf_examples.nethack.utils.variables import DeepDungeonLevel, DungeonLevel, Monster, Score


class VariablesInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.variables = [
            Monster(),
            DungeonLevel(),
            DeepDungeonLevel(),
            Score(),
        ]

    def _parse_variables(self, last_observation, observation, end_status):
        values = {}
        for variable in self.variables:
            name = variable.name.upper()
            values[name] = variable.get_value(self.env.unwrapped, last_observation, observation, end_status)

        return values

    def _parse_blstats(self, observation):
        blstats = BLStats(*observation[self.env.unwrapped._blstats_index])
        blstats = blstats._asdict()

        include = [
            "strength",
            "dexterity",
            "constitution",
            "intelligence",
            "wisdom",
            "charisma",
            "hitpoints",
            "max_hitpoints",
            "gold",
            "energy",
            "max_energy",
            "armor_class",
            "experience_level",
            "experience_points",
        ]
        blstats = dict(filter(lambda item: item[0] in include, blstats.items()))
        blstats_info = dict(zip(map(lambda s: s.upper(), blstats.keys()), blstats.values()))

        return blstats_info

    def reset(self, **kwargs):
        for task in self.variables:
            task.reset_value()

        # use tuple and copy to avoid shallow copy (`last_observation` would be the same as `observation`)
        last_observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        obs, info = super().reset(**kwargs)
        observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        # we have to set StepStatus here since in NLE there is no info TODO: move it elsewhere
        end_status = self.env.unwrapped.StepStatus.RUNNING
        info["end_status"] = end_status

        new_info = self._parse_variables(last_observation, observation, end_status)
        blstats_info = self._parse_blstats(observation)
        info = {**info, **blstats_info, **new_info}

        return obs, info

    def step(self, action):
        # use tuple and copy to avoid shallow copy (`last_observation` would be the same as `observation`)
        last_observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        obs, reward, terminated, truncated, info = super().step(action)
        observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        end_status = info["end_status"]

        new_info = self._parse_variables(last_observation, observation, end_status)
        blstats_info = self._parse_blstats(observation)
        info = {**info, **blstats_info, **new_info}

        return obs, reward, terminated, truncated, info
