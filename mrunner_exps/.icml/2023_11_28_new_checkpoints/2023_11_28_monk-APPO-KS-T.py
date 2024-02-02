from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-APPO-KS-T",
    "train_for_env_steps": 2_000_000_000,
    "group": "monk-APPO-KS-T",
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
    "kickstarting_loss_coeff": 0.05,
    "teacher_path": "None",
    "run_teacher_hs": False,
    "load_checkpoint_kind": "best",
}

checkpoints = [
    "/raid/NFS_SHARE/results/bartlomiej.cupial/sf_checkpoints/@-AA-BC-use_prev_action/2023-10-11-@-aa-bc-use-prev-action_kxvg_0/train_dir/default_experiment",
    "/raid/NFS_SHARE/results/bartlomiej.cupial/sf_checkpoints/@-AA-BC-use_prev_action/2023-10-11-@-aa-bc-use-prev-action_kxvg_1/train_dir/default_experiment",
    "/raid/NFS_SHARE/results/bartlomiej.cupial/sf_checkpoints/monk-AA-BC-use_prev_action/2023-10-11-monk-aa-bc-use-prev-action_tqri_0/train_dir/default_experiment",
    "/raid/NFS_SHARE/results/bartlomiej.cupial/sf_checkpoints/monk-AA-BC-use_prev_action/2023-10-11-monk-aa-bc-use-prev-action_tqri_1/train_dir/default_experiment",
]
use_prev_action = [True, False, True, False]

# params different between exps
params_grid = [
    {
        "seed": list(range(1)),
        "kickstarting_loss_coeff": [0.05, 0.1],
        "use_pretrained_checkpoint": [True],
        "model_path": [model_path],
        "teacher_path": [model_path],
        "use_prev_action": [use_prev_action],
    }
    for model_path, use_prev_action in zip(checkpoints, use_prev_action)
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
