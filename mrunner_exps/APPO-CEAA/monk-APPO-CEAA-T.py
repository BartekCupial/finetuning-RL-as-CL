from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-APPO-CEAA-T",
    "train_for_env_steps": 2_000_000_000,
    "group": "monk-APPO-CEAA-T",
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
    "use_dataset": True,
    "db_path": "/ttyrecs/ttyrecs.db",
    "data_path": "/nle/nld-aa/nle_data",
    "dataset_name": "autoascend",
    "dataset_rollout": 32,
    "dataset_batch_size": 8192,  # this equals bs = 256, 256 * 32 = 8192
    "supervised_loss_coeff": 0.5,
}

# params different between exps
params_grid = [
    {
        "seed": list(range(5)),
        "use_pretrained_checkpoint": [True],
        "model_path": ["/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_checkpoints/@-AA-BC/pretrained"],
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
