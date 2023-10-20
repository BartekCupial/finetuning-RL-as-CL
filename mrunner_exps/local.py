from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-APPO",
    "train_for_env_steps": 100_000_000,
    "group": "monk-APPO",
    "character": "mon-hum-neu-mal",
    "num_workers": 32,
    "num_envs_per_worker": 30,
    "worker_num_splits": 2,
    "rollout": 32,
    "batch_size": 4096,  # this equals bs = 128, 128 * 32 = 4096
    "async_rl": True,
    "serial_mode": False,
    "save_milestones_ith": 10_000_000,
    "wandb_user": "bartekcupial",
    "wandb_project": "sf2_nethack",
    "wandb_group": "gmum",
    "with_wandb": True,
}

# params different between exps
params_grid = [
    {
        "seed": list(range(1)),
        "num_workers": [8],
        "batch_size": [1024],
        "dataset_batch_size": [2048],
        "teacher_path": ["/home/bartek/Workspace/data/sf_checkpoints/monk-AA-BC/pretrained"],
        "model_path": ["/home/bartek/Workspace/data/sf_checkpoints/monk-AA-BC/pretrained"],
        "db_path": ["/home/bartek/Workspace/data/nethack/AA-taster/ttyrecs.db"],
        # "serial_mode": [True],
        "with_wandb": [False],
        "restart_behavior": ["overwrite"],
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
