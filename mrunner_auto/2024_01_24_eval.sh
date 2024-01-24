#!/bin/bash

ssh-add

# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/eval/2024_01_24_eval_APPO.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/eval/2024_01_24_eval_APPO-T.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/eval/2024_01_24_eval_APPO-KS-T.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/eval/2024_01_24_eval_APPO-BC-T.py

mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/eval/2024_01_24_eval_dlvl4_APPO.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/eval/2024_01_24_eval_dlvl4_APPO-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/eval/2024_01_24_eval_dlvl4_APPO-KS-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/eval/2024_01_24_eval_dlvl4_APPO-BC-T.py

mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/eval/2024_01_24_eval_sokoban_APPO.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/eval/2024_01_24_eval_sokoban_APPO-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/eval/2024_01_24_eval_sokoban_APPO-KS-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/eval/2024_01_24_eval_sokoban_APPO-BC-T.py