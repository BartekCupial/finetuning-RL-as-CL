import collections
from pathlib import Path

import pandas as pd
from nle.dataset.populate_db import XLOGFILE_COLUMNS

XLOGFILE_COLUMNS.append(("ttyrecname", str))


def game_data_generator(xlogfile, filter=lambda x: x, separator="\t"):
    with open(xlogfile, "rb") as f:
        for line in filter(f.readlines()):
            game_data = collections.defaultdict(lambda: -1)
            for words in line.decode("latin-1").strip().split(separator):
                key, *var = words.split("=")
                game_data[key] = "=".join(var)

            if "while" in game_data:
                game_data["death"] += " while " + game_data["while"]

            yield tuple(ctype(game_data[key]) for key, ctype in XLOGFILE_COLUMNS)


eval_csv_path = Path("train_dir/monk-AA-BC_pretrained_use_prev_action/mon-hum-neu-mal_episodes20/eval_p0.csv")
eval_df = pd.read_csv(eval_csv_path)
eval_df["ttyrecname"] = eval_df["ttyrec"].apply(lambda x: Path(x).name)

# right now assume that all ttyrecs are in the same folder
xlogdata = []
for xlogfile in reversed(
    sorted(filter(lambda p: p.suffix == ".xlogfile", Path(eval_df["ttyrec"][0]).parent.iterdir()))
):
    sep = ":" if str(xlogfile).endswith(".txt") else "\t"
    game_gen = game_data_generator(xlogfile, separator=sep)
    for values in game_gen:
        xlogdata.append(values)

xlogdf = pd.DataFrame(xlogdata, columns=list(map(lambda x: x[0], XLOGFILE_COLUMNS)))
xlogdf = xlogdf[["maxlvl", "role", "race", "gender", "align", "death", "ttyrecname"]]

merged_df = pd.merge(eval_df, xlogdf, on="ttyrecname", how="inner")
merged_df.to_csv(eval_csv_path.parent / "merged_p0.csv")
