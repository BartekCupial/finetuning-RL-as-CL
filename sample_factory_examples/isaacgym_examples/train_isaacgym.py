import sys
from typing import List

# this is here just to guarantee that isaacgym_examples is imported before PyTorch
import isaacgym
import torch
from torch import nn, Tensor

from sample_factory.algo.utils.context import global_env_registry
from sample_factory.algorithms.appo.model_utils import register_custom_encoder, EncoderBase, get_obs_shape
from sample_factory.algorithms.utils.arguments import parse_args
from sample_factory.envs.isaacgym.make_env import make_isaacgym_env
from sample_factory.train import run_rl
from sample_factory.utils.utils import str2bool


class _IsaacGymMlpEncoderImlp(nn.Module):
    def __init__(self, obs_space, mlp_layers: List[int]):
        super().__init__()

        obs_shape = get_obs_shape(obs_space)
        assert len(obs_shape.obs) == 1

        layer_input_width = obs_shape.obs[0]
        encoder_layers = []
        for layer_width in mlp_layers:
            encoder_layers.append(nn.Linear(layer_input_width, layer_width))
            layer_input_width = layer_width
            encoder_layers.append(nn.ELU(inplace=True))

        self.mlp_head = nn.Sequential(*encoder_layers)

    def forward(self, obs: Tensor):
        x = self.mlp_head(obs)
        return x


class IsaacGymMlpEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        self._impl = _IsaacGymMlpEncoderImlp(obs_space, cfg.mlp_layers)
        self._impl = torch.jit.script(self._impl)
        self.encoder_out_size = cfg.mlp_layers[-1]  # TODO: we should make this an abstract method

    def forward(self, obs_dict):
        x = self._impl(obs_dict['obs'])
        return x


def add_extra_params_func(env, parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument('--env_agents', default=4096, type=int, help='Num agents in each env')
    p.add_argument('--env_headless', default=True, type=str2bool, help='Headless == no rendering')
    p.add_argument('--mlp_layers', default=[256, 128, 64], type=int, nargs='*', help='MLP layers to use with isaacgym_examples envs')


def override_default_params_func(env, parser):
    """
    Override default argument values for this family of environments.
    All experiments for environments from my_custom_env_ family will have these parameters unless
    different values are passed from command line.

    """
    parser.set_defaults(
        # we're using a single very vectorized env, no need to parallelize it further
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        actor_worker_gpus=[0],  # obviously need a GPU

        train_for_env_steps=10000000,

        encoder_custom='isaac_gym_mlp_encoder',
        use_rnn=False,
        adaptive_stddev=False,
        policy_initialization='torch_default',
        env_gpu_actions=True,
        reward_scale=0.01,
        rollout=16,
        max_grad_norm=0.0,
        batch_size=32768,
        num_batches_per_iteration=2,
        ppo_epochs=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=2.0,
        exploration_loss_coeff=0.0,
        nonlinearity='elu',
        learning_rate=3e-4,
        lr_schedule='kl_adaptive_epoch',
        lr_schedule_kl_threshold=0.008,
        shuffle_minibatches=False,
        gamma=0.99,
        gae_lambda=0.95,
        with_vtrace=False,
        recurrence=1,
        value_bootstrap=True,  # assuming reward from the last step in the episode can generally be ignored
        normalize_input=True,
        experiment_summaries_interval=3,  # experiments are short so we should save summaries often
        save_every_sec=15,

        serial_mode=True,  # it makes sense to run isaacgym envs in serial mode since most of the parallelism comes from the env itself (although async mode works!)
        async_rl=False,
    )

    # environment specific overrides
    env_name = '_'.join(env.split('_')[1:]).lower()
    if env_name == 'ant':
        parser.set_defaults(mlp_layers=[256, 128, 64])
    elif env_name == 'humanoid':
        parser.set_defaults(
            train_for_env_steps=1310000000,  # to match how much it is trained in rl-games
            mlp_layers=[400, 200, 100],
            rollout=32,
            ppo_epochs=5,
            value_loss_coeff=4.0,
            max_grad_norm=1.0,
            num_batches_per_iteration=4,
        )
    elif env_name == 'allegrohandlstm':
        parser.set_defaults(
            env_agents=16384,

            train_for_env_steps=10_000_000_000,
            mlp_layers=[512, 256, 128],
            gamma=0.99,
            rollout=16,
            recurrence=16,
            use_rnn=True,
            learning_rate=1e-4,
            lr_schedule_kl_threshold=0.016,
            reward_scale=1.0,
            ppo_epochs=4,
            max_grad_norm=1.0,
            num_batches_per_iteration=8,
        )


def register_isaacgym_custom_components():
    global_env_registry().register_env(
        env_name_prefix='isaacgym_',
        make_env_func=make_isaacgym_env,
        add_extra_params_func=add_extra_params_func,
        override_default_params_func=override_default_params_func,
    )

    register_custom_encoder('isaac_gym_mlp_encoder', IsaacGymMlpEncoder)


def main():
    """Script entry point."""
    register_isaacgym_custom_components()
    cfg = parse_args()
    status = run_rl(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())