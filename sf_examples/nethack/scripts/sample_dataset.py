import sys

import numpy as np
import torch

from nle import nethack
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.utils.utils import log
from sf_examples.nethack.algo.learning.learner import DatasetLearner
from sf_examples.nethack.train_nethack import parse_nethack_args, register_nethack_components


def main():
    register_nethack_components()
    cfg = parse_nethack_args(evaluation=True)
    tmp_env = make_env_func_batched(cfg, env_config=None)
    env_info = extract_env_info(tmp_env, cfg)

    policy_id = 0
    policy_versions = torch.zeros([cfg.num_policies], dtype=torch.int32)
    param_server = ParameterServer(policy_id, policy_versions, cfg.serial_mode)

    learner = DatasetLearner(cfg, env_info, policy_versions, policy_id, param_server)
    learner.init()

    for i in range(10):
        done = False
        while not done:
            batch = learner._get_dataset_minibatch()
            done = batch["done"].any()

            if not cfg.no_render:
                tty_chars = batch["tty_chars"].cpu().numpy()[0][0].astype(np.uint8)
                tty_colors = batch["tty_colors"].cpu().numpy()[0][0].astype(np.uint8)
                tty_cursor = batch["tty_cursor"].cpu().numpy()[0][0].astype(np.uint8)
                print(nethack.tty_render(tty_chars, tty_colors, tty_cursor))

    log.info("Done!")


if __name__ == "__main__":
    sys.exit(main())

# Example usage
# python -m sf_examples.nethack.scripts.sample_dataset --env=challenge --dataset_batch_size=1 --rollout=1 --dataset_rollout=1 --dataset_num_splits=1 --use_dataset=True --supervised_loss_coeff=1.0 --device=cpu --db_path=/home/bartek/Workspace/data/nethack/AA-taster/ttyrecs.db
