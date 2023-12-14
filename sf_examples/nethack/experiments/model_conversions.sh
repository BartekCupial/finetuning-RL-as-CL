#!bin/bash

python -m sf_examples.nethack.scripts.convert_model \
    --env=challenge \
    --experiment=amzn-AA-BC_pretrained \
    --model_source_path=/home/bartek/Workspace/data/sf_checkpoints/dungeons/amzn-AA-BC/checkpoint_v100000000 \
    --use_prev_action=True \
    --model=ScaledNet \
    --use_resnet=True \
    --rnn_size=1738

python -m sf_examples.nethack.scripts.convert_model \
    --env=challenge \
    --experiment=monk-AA-BC_pretrained_use_prev_action \
    --model_source_path=/home/bartek/Workspace/data/sf_checkpoints/dungeons/monk-AA-BC/checkpoint.tar \
    --use_prev_action=True


python -m sf_examples.nethack.scripts.convert_model \
    --env=challenge \
    --experiment=monk-AA-BC_pretrained \
    --model_source_path=/home/bartek/Workspace/data/sf_checkpoints/dungeons/monk-AA-BC/checkpoint_v2000000000

python -m sf_examples.nethack.scripts.convert_model \
    --env=challenge \
    --experiment=@-AA-BC_pretrained \
    --model_source_path=/home/bartek/Workspace/data/sf_checkpoints/dungeons/@-AA-BC/checkpoint_v2000000000
