#!/bin/bash

ssh-add

mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_20_10_baselines/2023_20_10_monk-AA-BC.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_20_10_baselines/2023_20_10_monk-APPO-CEAA-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_20_10_baselines/2023_20_10_monk-APPO-CEAA.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_20_10_baselines/2023_20_10_monk-APPO-KLAA-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_20_10_baselines/2023_20_10_monk-APPO-KLAA.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_20_10_baselines/2023_20_10_monk-APPO-KS-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_20_10_baselines/2023_20_10_monk-APPO-KS.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_20_10_baselines/2023_20_10_monk-APPO-T.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/2023_20_10_baselines/2023_20_10_monk-APPO.py
