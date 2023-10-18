from __future__ import annotations

from dataclasses import dataclass
from typing import List

import gymnasium as gym

from sample_factory.algo.utils.action_distributions import calc_num_actions


@dataclass
class DatasetInfo:
    obs_space: gym.Space
    action_space: gym.Space
    num_agents: int
    action_splits: List[int]  # in the case of tuple actions, the splits for the actions
    all_discrete: bool  # in the case of tuple actions, whether the actions are all discrete
    gpu_actions: bool  # whether actions provided by the agent should be on GPU or not
    gpu_observations: bool  # whether environment provides data (obs, etc.) on GPU or not


def extract_dataset_info(dataset, cfg) -> DatasetInfo:
    obs_space = dataset.observation_space
    action_space = dataset.action_space

    gpu_actions = cfg.env_gpu_actions
    gpu_observations = cfg.env_gpu_observations

    action_splits = None
    all_discrete = None
    if isinstance(action_space, gym.spaces.Tuple):
        action_splits = [calc_num_actions(space) for space in action_space]
        all_discrete = all(isinstance(space, gym.spaces.Discrete) for space in action_space)

    env_info = DatasetInfo(
        obs_space=obs_space,
        action_space=action_space,
        num_agents=1,  # actually 0
        action_splits=action_splits,
        all_discrete=all_discrete,
        gpu_actions=gpu_actions,
        gpu_observations=gpu_observations,
    )
    return env_info
