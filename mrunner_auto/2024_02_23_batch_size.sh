#!/bin/bash

ssh-add

mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_02_23_batch_size/2024_03_04_monk-APPO-KS-T-baseline-batch_size-no_freeze.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_02_23_batch_size/2024_03_04_monk-APPO-KS-T-baseline-batch_size.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_02_23_batch_size/2024_03_04_monk-APPO-T-baseline-batch_size.py
