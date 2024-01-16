#!/bin/bash

ssh-add

# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_15_finetune_improve/2024_01_15_monk-APPO-KS-T-init_critic_head.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_15_finetune_improve/2024_01_15_monk-APPO-KS-T-ppo_epochs.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_15_finetune_improve/2024_01_15_monk-APPO-KS-T-reward.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_15_finetune_improve/2024_01_15_monk-APPO-KS-T-separate.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_15_finetune_improve/2024_01_15_monk-APPO-KS-T-unfreeze_everything.py

# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_15_finetune_improve/2024_01_16_monk-APPO-KS-T-min_kickstarting_coeff.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_15_finetune_improve/2024_01_16_monk-APPO-KS-T-no_warmup.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_15_finetune_improve/2024_01_16_monk-APPO-KS-T-return_ema.py