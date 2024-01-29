#!/bin/bash

train_dir="train_dir/default_experiment"
checkpoint_p0_dir="checkpoint_p0/milestones"

# Iterate through the checkpoint files
for file_path in $train_dir/$checkpoint_p0_dir/checkpoint_*.pth; do
    # Extract the checkpoint name from the file path
    checkpoint_name=$(basename "$file_path" .pth)

    # Create the experiment directory
    experiment_dir="train_dir/$checkpoint_name"
    mkdir -p "$experiment_dir/checkpoint_p0"

    # Copy files to the experiment directory excluding checkpoint_p0
    cp $train_dir/config.json "$experiment_dir/"
    cp $train_dir/sf_log.txt "$experiment_dir/"

    # Create the checkpoint_p0 directory and copy the checkpoint file
    mkdir -p "$experiment_dir/checkpoint_p0"
    cp "$file_path" "$experiment_dir/checkpoint_p0"
done

echo "Script executed successfully!"