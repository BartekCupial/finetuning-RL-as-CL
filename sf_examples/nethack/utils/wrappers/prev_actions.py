import gymnasium as gym
import numpy as np


class PrevActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_action = 0

        obs_spaces = {"prev_actions": self.env.action_space}
        obs_spaces.update([(k, self.env.observation_space[k]) for k in self.env.observation_space])
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def reset(self, **kwargs):
        self.prev_action = 0
        obs, info = super().reset(**kwargs)
        obs["prev_actions"] = np.array([self.prev_action])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.prev_action = action
        obs["prev_actions"] = np.array([self.prev_action])
        return obs, reward, terminated, truncated, info
