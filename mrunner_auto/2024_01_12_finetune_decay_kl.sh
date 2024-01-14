#!/bin/bash

ssh-add

# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_12_finetune_decay_kl/2024_01_12_monk-APPO-KS-T-freeze-decay.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_12_finetune_decay_kl/2024_01_12_monk-APPO-KS-T-freeze-decay-returns.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_12_finetune_decay_kl/2024_01_12_monk-APPO-KS-T-freeze-decay-gamma.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_12_finetune_decay_kl/2024_01_12_monk-APPO-KS-T-freeze-decay-reward_scale.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_12_finetune_decay_kl/2024_01_12_monk-APPO-KS-T-freeze_separate-decay.py
# mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_12_finetune_decay_kl/2024_01_12_monk-APPO-KS-T-freeze_encoder_only-decay.py