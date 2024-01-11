#!/bin/bash

ssh-add

# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_09_finetune_mw/2024_01_10_monk-APPO-T_normalize_returns.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_09_finetune_mw/2024_01_10_monk-APPO-KS-T_normalize_returns.py

# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_11_finetune_summary/2024_01_11_monk-APPO-T_unfreeze_critic.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_11_finetune_summary/2024_01_11_monk-APPO-KS-T_unfreeze_critic.py

# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_11_finetune_summary/2024_01_11_monk-APPO-KS-T_sweep2.py

# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_11_finetune_summary/2024_01_11_monk-APPO-T_separate.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_11_finetune_summary/2024_01_11_monk-APPO-KS-T_separate.py

# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_11_finetune_summary/2024_01_11_monk-APPO-T_ppo_epochs_reward_scale.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_11_finetune_summary/2024_01_11_monk-APPO-KS-T_ppo_epochs_reward_scale.py