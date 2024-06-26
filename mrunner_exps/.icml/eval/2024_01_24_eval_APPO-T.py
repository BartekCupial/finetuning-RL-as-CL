from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "run_script": "sf_examples.nethack.eval_nethack",
    "env": "challenge",
    "exp_tags": [name],
    "character": "mon-hum-neu-mal",
    # "character": "@",
    "with_wandb": True,
    "use_pretrained_checkpoint": False,
    "load_checkpoint_kind": "latest",
    "train_dir": "/home/bartek/Workspace/ideas/sample-factory/train_dir",
    "experiment": "amzn-AA-BC_pretrained",
    "sample_env_episodes": 20,
    "num_workers": 16,
    "num_envs_per_worker": 1,
    "worker_num_splits": 1,
    "restart_behavior": "overwrite",
}

csv_folder_name = f"{config['character']}_episodes{config['sample_env_episodes']}"

# params different between exps
params_grid = [
    {
        "seed": list(range(1)),
        "csv_folder_name": [csv_folder_name],
        "train_dir": [
            "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_exp/23_01-10_46-trusting_pare/2024-01-23-monk-appo-t-baseline_p4m5_0/train_dir",
            "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_exp/23_01-10_46-trusting_pare/2024-01-23-monk-appo-t-baseline_p4m5_1/train_dir",
            "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_exp/23_01-10_46-trusting_pare/2024-01-23-monk-appo-t-baseline_p4m5_2/train_dir",
            "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_exp/23_01-10_46-trusting_pare/2024-01-23-monk-appo-t-baseline_p4m5_3/train_dir",
            "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_exp/23_01-10_46-trusting_pare/2024-01-23-monk-appo-t-baseline_p4m5_4/train_dir",
        ],
        "experiment": ["default_experiment"],
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
