#!/bin/bash

ssh-add

mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/.icml/2024_01_23_seeds_monkey/2024_01_25_monk-APPO-baseline.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/.icml/2024_01_23_seeds_monkey/2024_01_25_monk-APPO-T-baseline.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/.icml/2024_01_23_seeds_monkey/2024_01_25_monk-APPO-KS-T-baseline.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/.icml/2024_01_23_seeds_monkey/2024_01_25_monk-APPO-BC-T-baseline.py
