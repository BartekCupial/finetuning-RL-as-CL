import json
import sys

import torch
from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.env_info import EnvInfo, obtain_env_info_in_a_separate_process
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.cfg.arguments import cfg_dict, parse_full_cfg, parse_sf_args
from sample_factory.utils.gpu_utils import set_global_cuda_envvars
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import cfg_file, experiment_dir, init_file_logger, log, save_git_diff
from sf_examples.nethack.nethack_params import (
    add_extra_params_general,
    add_extra_params_learner,
    add_extra_params_model,
    add_extra_params_model_scaled,
    add_extra_params_nethack_env,
    nethack_override_defaults,
)
from sf_examples.nethack.train_nethack import register_nethack_components


def _load_model(learner, model_source_path):
    assert learner.is_initialized

    actor_critic = learner.actor_critic
    checkpoint_dict = learner._get_checkpoint_dict()
    pretrained_checkpoint = torch.load(model_source_path)

    def load_weight(d, key, value):
        assert d[key] is not None
        assert d[key].shape == value.shape
        d[key] = value

    for key, value in pretrained_checkpoint["learner_state"]["model"].items():
        loaded = False
        # encoder
        if (
            key.startswith("topline")
            or key.startswith("screen")
            or key.startswith("bottom")
            or key.startswith("crop")
            or key.startswith("extract_crop")
            or key.startswith("fc")
        ):
            load_weight(checkpoint_dict["model"], "encoder." + key, value)
            loaded = True

        # core lstm
        if key.startswith("core"):
            load_weight(checkpoint_dict["model"], "core." + key, value)
            loaded = True

        # critic linear
        if key.startswith("baseline.weight"):
            load_weight(checkpoint_dict["model"], "critic_linear.weight", value)
            loaded = True
        if key.startswith("baseline.bias"):
            load_weight(checkpoint_dict["model"], "critic_linear.bias", value)
            loaded = True

        # policy linear
        if key.startswith("policy.weight"):
            load_weight(checkpoint_dict["model"], "action_parameterization.distribution_linear.weight", value)
            loaded = True
        if key.startswith("policy.bias"):
            load_weight(checkpoint_dict["model"], "action_parameterization.distribution_linear.bias", value)
            loaded = True

        if not loaded:
            log.debug(f"Key `{key}` was not found in pretrained checkpoint, default value was kept.")

    actor_critic.load_state_dict(checkpoint_dict["model"], strict=True)


def convert_model(cfg: Config, env_info: EnvInfo):
    set_global_cuda_envvars(cfg)

    buffer_mgr = BufferMgr(cfg, env_info)
    policy_versions_tensor: Tensor = buffer_mgr.policy_versions

    param_servers = {}
    init_model_data = {}
    learners = {}
    for policy_id in range(cfg.num_policies):
        param_servers[policy_id] = ParameterServer(policy_id, policy_versions_tensor, cfg.serial_mode)
        learners[policy_id] = Learner(cfg, env_info, policy_versions_tensor, policy_id, param_servers[policy_id])
        init_model_data[policy_id] = learners[policy_id].init()

    # 1) save data as new experiment
    init_file_logger(cfg)

    fname = cfg_file(cfg)
    with open(fname, "w") as json_file:
        log.debug(f"Saving configuration to {fname}...")
        json.dump(cfg_dict(cfg), json_file, indent=2)

    save_git_diff(experiment_dir(cfg))

    # 2) load model for each policy
    for policy_id in range(cfg.num_policies):
        _load_model(learners[policy_id], cfg.model_source_path)

    # 3) save each policy with checkpoint
    for policy_id in range(cfg.num_policies):
        learners[policy_id].save()


def add_extra_params_convert_model(parser):
    """
    Specify any additional command line arguments for NetHack model conversion..
    """
    p = parser
    p.add_argument("--model_source_path", type=str, default=None)


def main():  # pragma: no cover
    """Script entry point."""
    register_nethack_components()

    parser, cfg = parse_sf_args()
    add_extra_params_nethack_env(parser)
    add_extra_params_model(parser)
    add_extra_params_model_scaled(parser)
    add_extra_params_learner(parser)
    add_extra_params_general(parser)
    add_extra_params_convert_model(parser)
    nethack_override_defaults(cfg.env, parser)
    cfg = parse_full_cfg(parser)

    env_info = obtain_env_info_in_a_separate_process(cfg)
    status = convert_model(cfg, env_info)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
