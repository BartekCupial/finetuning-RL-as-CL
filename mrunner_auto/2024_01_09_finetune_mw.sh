#!/bin/bash

ssh-add

# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_09_finetune_mw/2024_01_08_monk-APPO-T_sweep2.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_09_finetune_mw/2024_01_08_monk-APPO-T_sweep1.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_09_finetune_mw/2024_01_08_monk-APPO-T_sweep3.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_09_finetune_mw/2024_01_08_monk-APPO-KS-T_sweep3.py
