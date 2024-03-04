from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "challenge",
    "exp_tags": [name],
    "exp_point": "monk-APPO-KS-T",
    "train_for_env_steps": 500_000_000,
    "group": "monk-APPO-KS-T",
    "character": "@",
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
    "use_pretrained_checkpoint": True,
    "model_path": "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_checkpoints/amzn-AA-BC_pretrained",
    "kickstarting_loss_coeff": 0.5,
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
    "kickstarting_loss_decay": 0.99998,
}

params_grid = []
expected_batch_size = 4096

for rollout in [32, 64, 128, 256]:
    for target_batch_size in [128, 256, 512, 1024, 2048, 4096]:
        batch_size = min(expected_batch_size, min(target_batch_size * rollout, expected_batch_size * 8))
        batches_to_accumulate = max(1, (rollout * target_batch_size) // expected_batch_size)
        optim_step_every_ith = max(1, batches_to_accumulate // 8)
        params_grid.append(
            {
                "seed": list(range(1)),
                "learning_rate": [0.0001],
                "freeze": [{"encoder": 0}],
                "rollout": [rollout],
                "batch_size": [batch_size],  # 32 * 512, 64 * 256, 128 * 128
                "num_batches_per_epoch": [min(8, batches_to_accumulate)],
                "optim_step_every_ith": [optim_step_every_ith],
                "target_batch_size": [target_batch_size],
                "kickstarting_loss_decay": [0.9999, 0.99996, 0.99998],
            }
        )


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
