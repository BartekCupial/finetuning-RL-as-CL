import sys
from pathlib import Path

import pandas as pd

import wandb
from sample_factory.cfg.arguments import checkpoint_override_defaults, parse_full_cfg, parse_sf_args
from sample_factory.eval import do_eval
from sample_factory.utils.utils import experiment_dir
from sf_examples.nethack.nethack_params import (
    add_extra_params_general,
    add_extra_params_learner,
    add_extra_params_model,
    add_extra_params_nethack_env,
    nethack_override_defaults,
)
from sf_examples.nethack.scripts.analyze_results import get_xlogdata
from sf_examples.nethack.train_nethack import register_nethack_components


def main():  # pragma: no cover
    """Script entry point."""
    register_nethack_components()

    parser, cfg = parse_sf_args(evaluation=True)
    add_extra_params_nethack_env(parser)
    add_extra_params_model(parser)
    add_extra_params_learner(parser)
    add_extra_params_general(parser)
    nethack_override_defaults(cfg.env, parser)

    # important, instead of `load_from_checkpoint` as in enjoy we want
    # to override it here to be able to use argv arguments
    checkpoint_override_defaults(cfg, parser)

    cfg = parse_full_cfg(parser)

    status = do_eval(cfg)

    # if we saved ttyrecs use xlogfiles and add their information to csv
    if cfg.save_ttyrec_every != 0 and cfg.add_stats_to_info:
        for policy_id in range(cfg.num_policies):
            csv_output_dir = Path(experiment_dir(cfg=cfg))
            if cfg.csv_folder_name is not None:
                csv_output_dir = csv_output_dir / cfg.csv_folder_name
            csv_output_dir.mkdir(exist_ok=True, parents=True)
            csv_output_path = csv_output_dir / f"eval_p{policy_id}.csv"

            data = pd.read_csv(csv_output_path)

            xlogdata = get_xlogdata(cfg.savedir)
            merged_df = pd.merge(data, xlogdata, on="ttyrecname", how="inner")
            merged_df.to_csv(csv_output_dir / f"merged_p{policy_id}.csv")

            if cfg.with_wandb:
                # reupload the table with information about death etc
                table = wandb.Table(dataframe=merged_df)
                wandb.log({"table_results": table})

    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
