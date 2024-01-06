#!/bin/bash

ssh-add

# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_05_finetune_amzn/2024_01_05_monk-APPO-KS-T.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_05_finetune_amzn/2024_01_05_monk-APPO-T.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_05_finetune_amzn/2024_01_05_monk-APPO.py

mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_05_finetune_amzn/2024_01_05_monk-APPO-KS-T_long.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_05_finetune_amzn/2024_01_05_monk-APPO-T_long.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_05_finetune_amzn/2024_01_05_monk-APPO_long.py
