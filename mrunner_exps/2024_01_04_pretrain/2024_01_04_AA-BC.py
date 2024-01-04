from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-AA-BC",
    "train_for_env_steps": 2_000_000_000,
    "group": "monk-AA-BC",
    "character": "mon-hum-neu-mal",
    "num_workers": 16,
    "num_envs_per_worker": 32,
    "worker_num_splits": 2,
    "rollout": 32,
    "batch_size": 4096,  # this equals bs = 64, 64 * 32 = 2048
    "async_rl": True,
    "serial_mode": False,
    "wandb_user": "bartekcupial",
    "wandb_project": "sf2_nethack",
    "wandb_group": "gmum",
    "with_wandb": True,
    "use_dataset": True,
    "dataset_rollout": 32,
    "dataset_num_workers": 8,
    "dataset_batch_size": 16384,  # this equals bs = 512, 512 * 32 = 16384
    "supervised_loss_coeff": 1.0,
    "behavioral_clone": True,
}

# params different between exps
params_grid = [
    {
        "seed": list(range(3)),
        "character": ["@", "mon-hum-neu-mal"],
        "learning_rate": [0.001],
        "use_prev_action": [False, True],
        # "with_wandb": [False],
        # "batch_size": [1024],
        # "dataset_batch_size": [1024],
        # "num_workers": [8],
        # "dataset_num_workers": [8],
        # "restart_behavior": ["overwrite"],
        # "db_path": ["/home/bartek/Workspace/data/nethack/AA-taster/ttyrecs.db"],
        # "model_path": [
        #     "train_dir/monk-AA-BC_pretrained_use_prev_action"
        # ],
        # "teacher_path": [
        #     "train_dir/monk-AA-BC_pretrained_use_prev_action"
        # ],
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
