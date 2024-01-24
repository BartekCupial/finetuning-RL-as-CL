from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper

from sf_examples.nethack.utils.paramiko import get_save_paths

name = globals()["script"][:-3]

# params for all exps
config = {
    "run_script": "sf_examples.nethack.eval_nethack",
    "env": "challenge",
    "exp_tags": [name],
    "character": "mon-hum-neu-mal",
    # "character": "@",
    "use_pretrained_checkpoint": False,
    "load_checkpoint_kind": "latest",
    "train_dir": "/home/bartek/Workspace/ideas/sample-factory/train_dir",
    "experiment": "amzn-AA-BC_pretrained",
    "sample_env_episodes": 128,
    "num_workers": 16,
    "num_envs_per_worker": 8,
    "worker_num_splits": 2,
    "restart_behavior": "overwrite",
}

csv_folder_name = f"{config['character']}_episodes{config['sample_env_episodes']}"

save_root_path = Path("/net/pr2/projects/plgrid/plgggmum_crl/bcupial/gamesavedir")

expected_saves = 128
folder = "sokoban"
saves = get_save_paths(save_root_path / folder)
if len(saves) < expected_saves:
    saves = saves * ((expected_saves // len(saves)) + 1)
saves = [save_root_path / folder / s for e, s in enumerate(saves) if e < expected_saves]

# params different between exps
params_grid = [
    {
        "seed": list(range(1)),
        "csv_folder_name": [csv_folder_name],
        "train_dir": [
            "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_exp/23_01-10_47-pedantic_johnson/2024-01-23-monk-appo-bc-t-baseline_plmv_0/train_dir",
            "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_exp/23_01-10_47-pedantic_johnson/2024-01-23-monk-appo-bc-t-baseline_plmv_1/train_dir",
            "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_exp/23_01-10_47-pedantic_johnson/2024-01-23-monk-appo-bc-t-baseline_plmv_2/train_dir",
            "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_exp/23_01-10_47-pedantic_johnson/2024-01-23-monk-appo-bc-t-baseline_plmv_3/train_dir",
            "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_exp/23_01-10_47-pedantic_johnson/2024-01-23-monk-appo-bc-t-baseline_plmv_4/train_dir",
        ],
        "experiment": ["default_experiment"],
        "gameloaddir": [saves],
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
