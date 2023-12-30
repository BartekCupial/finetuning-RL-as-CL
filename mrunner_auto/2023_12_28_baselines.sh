#!/bin/bash

ssh-add

mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO-CEAA-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO-KLAA-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO-KS-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO-T.py

mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO-CEAA.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO-KLAA.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO-KS.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_12_28_baselines/2023_12_28_monk-APPO.py
