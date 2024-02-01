from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper

from sf_examples.nethack.utils.paramiko import get_checkpoint_paths

name = globals()["script"][:-3]

# params for all exps
config = {
    "run_script": "sf_examples.nethack.eval_nethack",
    "env": "challenge",
    "exp_tags": [name],
    "character": "mon-hum-neu-mal",
    "with_wandb": True,
    "use_pretrained_checkpoint": False,
    "load_checkpoint_kind": "latest",
    "train_dir": "/home/bartek/Workspace/ideas/sample-factory/train_dir",
    "experiment": "amzn-AA-BC_pretrained",
    "sample_env_episodes": 200,
    "num_workers": 16,
    "num_envs_per_worker": 1,
    "worker_num_splits": 1,
    "restart_behavior": "overwrite",
}

csv_folder_name = f"{config['character']}_episodes{config['sample_env_episodes']}"

checkpoints = get_checkpoint_paths("/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_exp/appo-baseline/")
checkpoints = list(map(Path, checkpoints))
checkpoints = list(filter(lambda p: p.parent.name in ["default_experiment"], checkpoints))

# params different between exps
params_grid = [
    {
        "csv_folder_name": [csv_folder_name],
        "train_dir": [str(checkpoint.parent.parent)],
        "experiment": [checkpoint.parent.name],
    }
    for checkpoint in checkpoints
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
