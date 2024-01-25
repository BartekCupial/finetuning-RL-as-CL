from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-APPO-BC-T",
    "train_for_env_steps": 2_000_000_000,
    "group": "monk-APPO-BC-T",
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
    "use_dataset": True,
    "dataset_rollout": 32,
    "dataset_batch_size": 4096,  # this equals bs = 256, 256 * 32 = 8192
    "use_pretrained_checkpoint": True,
    "model_path": "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_checkpoints/amzn-AA-BC_pretrained",
    "distillation_loss_coeff": 0.5,
    "teacher_path": "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_checkpoints/amzn-AA-BC_pretrained",
    "run_teacher_hs": False,
    "use_prev_action": True,
    "model": "ScaledNet",
    "use_resnet": True,
    "learning_rate": 0.0001,
    "rnn_size": 1738,
    "h_dim": 1738,
    "exploration_loss_coeff": 0.0,
    "gamma": 1.0,
    "skip_train": 25_000_000,
    "lr_schedule": "linear_decay",
    "save_milestones_ith": 25_000_000,
}

# params different between exps
params_grid = [
    {
        "seed": list(range(5)),
        "freeze": [{"encoder": 0}],
        "distillation_loss_coeff": [2.0],
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
