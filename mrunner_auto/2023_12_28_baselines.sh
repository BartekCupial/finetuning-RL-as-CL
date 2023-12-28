#!/bin/bash

ssh-add

mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO-CEAA.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO-KLAA.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO-KS.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO.py
