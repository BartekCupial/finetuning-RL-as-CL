#!/bin/bash

ssh-add

mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_18_finetune_better/2024_01_18_monk-APPO-KS-T-pretrain_critic.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_18_finetune_better/2024_01_18_monk-APPO-KS-T-baseline_freeze_encoder-2B.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_18_finetune_better/2024_01_18_monk-APPO-BC-T-baseline.py