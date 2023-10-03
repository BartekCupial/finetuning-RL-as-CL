from __future__ import annotations

import multiprocessing
import os
import pickle
from dataclasses import dataclass
from os.path import join
from typing import List, Optional, Tuple


from sample_factory.algo.utils.action_distributions import calc_num_actions
from sample_factory.algo.utils.context import set_global_context, sf_global_context
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log, project_tmp_dir

from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.utils.typing import Config
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.action_distributions import calc_num_action_parameters, calc_num_actions


ACTOR_CRITIC_INFO_PROTOCOL_VERSION = 1

@dataclass
class ActorCriticInfo:
    policy_output_shapes: List[Tuple[str, List]]
    
    # version of the protocol, used to detect changes in the ActorCriticInfo class and invalidate the cache if needed
    # bump this version if you make any changes to the ActorCriticInfo class
    actor_critic_info_protocol_version: Optional[int] = None


def action_info(env_info: EnvInfo) -> Tuple[int, int]:
    action_space = env_info.action_space
    num_actions = calc_num_actions(action_space)
    num_action_distribution_parameters = calc_num_action_parameters(action_space)
    return num_actions, num_action_distribution_parameters


def extract_actor_critic_info(actor_critic: ActorCritic, env_info: EnvInfo, cfg: Config) -> ActorCriticInfo:
    num_actions, num_action_distribution_parameters = action_info(env_info)
    
    actor_critic_info = ActorCriticInfo(
        policy_output_shapes = actor_critic.policy_output_shapes(num_actions, num_action_distribution_parameters)
    )
    
    return actor_critic_info


def spawn_tmp_ac_and_get_info(sf_context, res_queue, cfg, env_info):
    set_global_context(sf_context)

    tmp_actor_critic = create_actor_critic(cfg, env_info.obs_space, env_info.action_space)
    actor_critic_info = extract_actor_critic_info(tmp_actor_critic, env_info, cfg)
    del tmp_actor_critic

    log.debug("Env info: %r", actor_critic_info)
    res_queue.put(actor_critic_info)


def actor_critic_info_cache_filename() -> str:
    return join(project_tmp_dir(), f"actor_critic_info")


def obtain_ac_info_in_a_separate_process(cfg: Config, env_info: EnvInfo) -> ActorCriticInfo:
    cache_filename = actor_critic_info_cache_filename()
    if cfg.use_actor_critic_info_cache and os.path.isfile(cache_filename):
        log.debug(f"Loading actor critic info from cache: {cache_filename}")
        with open(cache_filename, "rb") as fobj:
            actor_critic_info = pickle.load(fobj)
            if actor_critic_info.actor_critic_info_protocol_version == ACTOR_CRITIC_INFO_PROTOCOL_VERSION:
                return actor_critic_info

    sf_context = sf_global_context()

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=spawn_tmp_ac_and_get_info, args=(sf_context, q, cfg, env_info))
    p.start()

    actor_critic_info = q.get()
    p.join()

    if cfg.use_actor_critic_info_cache:
        with open(cache_filename, "wb") as fobj:
            pickle.dump(actor_critic_info, fobj)

    return actor_critic_info
