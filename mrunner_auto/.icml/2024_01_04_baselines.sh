#!/bin/bash

ssh-add

mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_04_baselines/2024_01_04_monk-APPO-CEAA-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_04_baselines/2024_01_04_monk-APPO-KLAA-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_04_baselines/2024_01_04_monk-APPO-KS-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_04_baselines/2024_01_04_monk-APPO-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2024_01_04_baselines/2024_01_04_monk-APPO.py
