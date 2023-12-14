from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "run_script": "sf_examples.nethack.eval_nethack",
    "env": "challenge",
    "exp_tags": [name],
    "character": "mon-hum-neu-mal",
    # "character": "@",
    "decorrelate_envs_on_one_worker": False,
    "use_pretrained_checkpoint": False,
    "load_checkpoint_kind": "latest",
    "train_dir": "/home/bartek/Workspace/ideas/sample-factory/train_dir",
    "experiment": "amzn-AA-BC_pretrained",
    "sample_env_episodes": 1024,
    "num_workers": 16,
    "num_envs_per_worker": 32,
    "worker_num_splits": 2,
}

csv_folder_name = f"{config['character']}_episodes{config['sample_env_episodes']}"

# params different between exps
params_grid = [
    {
        "seed": list(range(1)),
        "csv_folder_name": [csv_folder_name],
        "train_dir": ["/raid/NFS_SHARE/results/bartlomiej.cupial/sf_checkpoints/train_dir"],
        "experiment": [
            "amzn-AA-BC_pretrained",
            "@-AA-BC_pretrained",
            "monk-AA-BC_pretrained",
            "monk-AA-BC_pretrained_use_prev_action",
        ],
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
