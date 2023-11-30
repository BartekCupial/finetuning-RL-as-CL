import json
import os
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torch import multiprocessing as mp

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint, parse_full_cfg, parse_sf_args
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import log
from sf_examples.nethack.nethack_params import (
    add_extra_params_eval,
    add_extra_params_general,
    add_extra_params_learner,
    add_extra_params_model,
    add_extra_params_nethack_env,
    nethack_override_defaults,
)
from sf_examples.nethack.train_nethack import register_nethack_components

# Necessary for multithreading.
os.environ["OMP_NUM_THREADS"] = "1"


mp.set_sharing_strategy("file_system")  # see https://github.com/pytorch/pytorch/issues/11201


class Rollout:
    """
    A class used to rollout trained models on the NLE environment.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Load trained model config
        cfg = load_from_checkpoint(cfg)
        cfg.num_envs = 1

        # initialize the Torch modules
        if self.cfg.seed is None:
            log.info("Starting seed is not provided")
        else:
            log.info("Setting fixed seed %d", self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        env = make_env_func_batched(self.cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))
        self.env_info = extract_env_info(env, self.cfg)

    def _agent_setup(self):
        """
        Construct agent and load in weights.
        """
        actor_critic = create_actor_critic(self.cfg, self.env_info.obs_space, self.env_info.action_space)
        actor_critic.eval()

        policy_id = self.cfg.policy_index
        name_prefix = dict(latest="checkpoint", best="best")[self.cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(self.cfg, policy_id), f"{name_prefix}_*")
        checkpoint_dict = Learner.load_checkpoint(checkpoints, "cpu")

        # check if student teacher architecture
        student_params = dict(filter(lambda x: x[0].startswith("student"), checkpoint_dict["model"].items()))
        if len(student_params) > 0:
            # modify keys
            student_params = dict(map(lambda x: (x[0].removeprefix("student."), x[1]), student_params.items()))
            actor_critic.load_state_dict(student_params)
        else:
            actor_critic.load_state_dict(checkpoint_dict["model"])

        return actor_critic

    def _submit_actor(self, ctx, seed: int, idx: int):
        """
        Submit and return actor idx with given seed.
        """
        actor = ctx.Process(
            target=self._single_rollout,
            args=(seed, idx),
            name="Actor-%i" % idx,
        )
        actor.start()

        return actor

    def _get_seeds(self):
        """
        Generate num_eval_rollouts number of seeds.
        """
        return range(self.cfg.num_eval_rollouts)

    def _spawn_rollouts(self):
        """
        Spawn cfg.num_eval_workers number of parallel actors to perform rollouts.
        """
        # Make sure we don't copy agent memory
        self.agent.share_memory()

        # Spawn actors
        actor_processes = []
        seeds = self._get_seeds()

        # Get context
        ctx = mp.get_context("fork")

        # Spawn first set of actors
        for i in range(self.cfg.num_eval_workers):
            actor = self._submit_actor(ctx, seeds[i], i)
            actor_processes.append(actor)
        i += 1

        # Keep spawning new processes as old ones finish
        while len(actor_processes) < self.cfg.num_eval_rollouts:
            if not self.done_q.empty():
                print(self.done_q.get())
                actor = self._submit_actor(ctx, seeds[i], i)
                actor_processes.append(actor)
                i += 1

        # Wait for all actors to finish
        for actor in actor_processes:
            actor.join()

    def _setup_env(
        self,
        seed: int,
        actor_num: int,
    ):
        """
        All logic related to setting up the appropriate NLE environment.
        """
        render_mode = "human"
        if self.cfg.no_render:
            render_mode = None

        env = make_env_func_batched(
            self.cfg,
            env_config=AttrDict(worker_index=actor_num, vector_index=0, env_id=actor_num),
            render_mode=render_mode,
        )
        env.seed(seed)

        return env

    @torch.no_grad()
    def _single_rollout(self, seed: int, actor_num: int, device: torch.device = torch.device("cpu")):
        """
        Rollout and log relevant objects (observations, actions, returns).
        """
        env = self._setup_env(seed, actor_num)
        cfg = self.cfg
        env_info = self.env_info
        actor_critic = self.agent

        episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
        num_frames = 0

        def max_frames_reached(frames):
            return cfg.max_num_frames is not None and frames > cfg.max_num_frames

        reward_list = []

        obs, infos = env.reset()
        rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
        episode_reward = None
        finished_episode = [False for _ in range(env.num_agents)]

        num_episodes = 0

        while not max_frames_reached(num_frames):
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

            policy_outputs = actor_critic(normalized_obs, rnn_states)

            # sample actions from the distribution by default
            actions = policy_outputs["actions"]

            if cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]

            obs, rew, terminated, truncated, infos = env.step(actions)
            dones = make_dones(terminated, truncated)
            infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

            if episode_reward is None:
                episode_reward = rew.float().clone()
            else:
                episode_reward += rew.float()

            num_frames += 1

            dones = dones.cpu().numpy()
            for agent_i, done_flag in enumerate(dones):
                if done_flag:
                    finished_episode[agent_i] = True
                    rew = episode_reward[agent_i].item()
                    episode_rewards[agent_i].append(rew)

                    rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                    episode_reward[agent_i] = 0

                    if cfg.use_record_episode_statistics:
                        # we want the scores from the full episode not a single agent death (due to EpisodicLifeEnv wrapper)
                        if "episode" in infos[agent_i].keys():
                            num_episodes += 1
                            reward_list.append(infos[agent_i]["episode"]["r"])
                    else:
                        num_episodes += 1

            if all(finished_episode):
                log.info("Reached done signal.")
                self._wrap_up_rollout(num_frames, episode_rewards, infos)
                break
        else:
            log.info("Cutting episode short ...")
            self._wrap_up_rollout(num_frames, episode_rewards, infos)

        env.close()

    def _wrap_up_rollout(self, num_frames, episode_rewards, infos):
        """
        Do any final logging/saving/etc. that needs to happen
        when the game ends.
        """
        for agent_i in range(self.env_info.num_agents):
            metrics = {
                "episode_return": episode_rewards[agent_i][-1],
                "episode_step": num_frames,
                **infos[agent_i]["episode_extra_stats"],
            }

            log.info(
                "Episode finished for agent %d at %d frames. Reward: %.3f",
                agent_i,
                num_frames,
                episode_rewards[agent_i][-1],
            )

            self.metrics_q.put(metrics)

            if self.done_q:
                self.done_q.put("done!")

    def rollout_cpu(self):
        """
        Rollout trained model ~cfg.num_eval_rollouts number of times on CPU.
        """
        self.agent = self._agent_setup()

        self.metrics_q = mp.Manager().Queue()
        self.done_q = mp.Manager().Queue()

        start_time = time.time()
        self._spawn_rollouts()
        wall_time = time.time() - start_time

        self._post_process(wall_time)

    def rollout_gpu(self):
        """
        Rollout trained model ~flags.num_rollouts number of times on GPU.
        """
        self.agent = self._agent_setup()

        device = torch.device("cuda")
        self.agent.model_to_device(device)

        seeds = self._get_seeds()

        self.metrics_q = mp.Manager().Queue()
        self.done_q = mp.Manager().Queue()

        start_time = time.time()
        for idx, seed in enumerate(seeds):
            self._single_rollout(seed, idx, device)
        wall_time = time.time() - start_time

        self._post_process(wall_time)

    def _post_process(self, wall_time):
        """
        Compute and save final metrics.
        """
        # TODO:

        returns = []
        episode_lens = []
        data = []
        while not self.metrics_q.empty():
            metrics = self.metrics_q.get()

            # returns
            returns.append(metrics["episode_return"])

            # episode lens
            episode_lens.append(metrics["episode_step"])

            # metrics
            data.append(metrics)

        log.info(f"Avg. return: {np.mean(returns)}")
        log.info(
            f"95% CI: {str(stats.t.interval(0.95, len(returns)-1, loc=np.mean(returns), scale=stats.sem(returns)))}"
        )

        data = pd.DataFrame(data)
        data.to_csv("eval.csv")

        results = {f"eval/{k}": np.mean(v) for k, v in data.items()}
        results["eval/count"] = self.cfg.num_eval_rollouts
        results["eval/wall_time"] = wall_time
        log.info(json.dumps(results, indent=4))


def eval(cfg: Config):
    rollout = Rollout(cfg)
    if cfg.device == "cpu":
        rollout.rollout_cpu()
    else:
        rollout.rollout_gpu()
    return ExperimentStatus.SUCCESS


def main():  # pragma: no cover
    """Script entry point."""
    register_nethack_components()

    parser, cfg = parse_sf_args(evaluation=True)
    add_extra_params_nethack_env(parser)
    add_extra_params_model(parser)
    add_extra_params_eval(parser)
    add_extra_params_learner(parser)
    add_extra_params_general(parser)
    nethack_override_defaults(cfg.env, parser)
    cfg = parse_full_cfg(parser)

    status = eval(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
