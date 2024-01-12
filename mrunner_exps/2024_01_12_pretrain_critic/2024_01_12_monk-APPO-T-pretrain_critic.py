from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-APPO-T",
    "train_for_env_steps": 500_000_000,
    "group": "monk-APPO-T",
    "character": "mon-hum-neu-mal",
    "num_workers": 16,
    "num_envs_per_worker": 32,
    "worker_num_splits": 2,
    "rollout": 32,
    "batch_size": 4096,  # this equals bs = 128, 128 * 32 = 4096
    "async_rl": True,
    "serial_mode": False,
    "wandb_user": "bartekcupial",
    "wandb_project": "sf2_nethack",
    "wandb_group": "gmum",
    "with_wandb": True,
    "use_prev_action": True,
    "model": "ScaledNet",
    "use_resnet": True,
    "rnn_size": 1738,
    "h_dim": 1738,
}

# params different between exps
params_grid = [
    {
        "seed": list(range(1)),
        "learning_rate": [0.0001, 0.00001],
        "use_pretrained_checkpoint": [True],
        "model_path": ["/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_checkpoints/amzn-AA-BC_pretrained"],
        "freeze": [{"encoder": 0, "core": 0, "decoder": 0, "action_parameterization": 0, "critic_linear": 0}],
        "unfreeze": [
            {
                "critic_linear": 50_000_000,
            }
        ],
        "warmup": [50_000_000],
        "train_for_env_steps": [500_000_000, 1_000_000_000],
    },
    {
        "seed": list(range(1)),
        "learning_rate": [0.0001, 0.00001],
        "use_pretrained_checkpoint": [True],
        "model_path": ["/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_checkpoints/amzn-AA-BC_pretrained"],
        "freeze": [
            {
                "actor_encoder": 0,
                "actor_core": 0,
                "actor_decoder": 0,
                "action_parameterization": 0,
                "critic_encoder": 0,
                "critic_core": 0,
                "critic_decoder": 0,
                "critic_linear": 0,
            }
        ],
        "unfreeze": [
            {
                "critic_encoder": 50_000_000,
                "critic_core": 50_000_000,
                "critic_decoder": 50_000_000,
                "critic_linear": 50_000_000,
            }
        ],
        "actor_critic_share_weights": [False],
        "warmup": [50_000_000],
        "train_for_env_steps": [500_000_000, 1_000_000_000],
    },
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="sf2_nethack",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)
