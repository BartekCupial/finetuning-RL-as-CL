# Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem

See also our ICML paper https://arxiv.org/abs/2402.02868

## Installation
Works in Python 3.10. Higher versions have problems with building NLE.

To install NetHack, you need nle and its dependencies.

```bash
# nle dependencies
apt-get install build-essential python3-dev python3-pip python3-numpy autoconf libtool pkg-config libbz2-dev
conda install cmake flex bison lit

# install nle locally and modify it to enable seeding and handle rendering with gymnasium
git clone https://github.com/facebookresearch/nle.git nle && cd nle \
&& git checkout v0.9.0 && git submodule init && git submodule update --recursive \
&& sed '/#define NLE_ALLOW_SEEDING 1/i#define NLE_ALLOW_SEEDING 1' include/nleobs.h -i \
&& sed '/self\.nethack\.set_initial_seeds = f/d' nle/env/tasks.py -i \
&& sed '/self\.nethack\.set_current_seeds = f/d' nle/env/tasks.py -i \
&& sed '/self\.nethack\.get_current_seeds = f/d' nle/env/tasks.py -i \
&& sed '/def seed(self, core=None, disp=None, reseed=True):/d' nle/env/tasks.py -i \
&& sed '/raise RuntimeError("NetHackChallenge doesn.t allow seed changes")/d' nle/env/tasks.py -i \
&& python setup.py install && cd .. 

# install sample factory with nethack extras
pip install -e .[nethack]
conda install -c conda-forge pybind11
pip install -e sf_examples/nethack/nethack_render_utils
```

## Running Experiments

Run NetHack experiments with the scripts in `sf_examples.nethack`.
The default parameters have been chosen to match [dungeons & data](https://github.com/dungeonsdatasubmission/dungeonsdata-neurips2022) which is based on [nle sample factory baseline](https://github.com/Miffyli/nle-sample-factory-baseline). By moving from D&D to sample factory we've managed to increase the APPO score from 2k to 2.8k.

To train a model in the `nethack_challenge` environment:

```
python -m sf_examples.nethack.train_nethack \
    --env=nethack_challenge \
    --batch_size=4096 \
    --num_workers=16 \
    --num_envs_per_worker=32 \
    --worker_num_splits=2 \
    --rollout=32 \
    --character=mon-hum-neu-mal \
    --model=ChaoticDwarvenGPT5 \
    --rnn_size=512 \
    --experiment=nethack_monk
```

To visualize the training results, use the `enjoy_nethack` script:

```
python -m sf_examples.nethack.enjoy_nethack --env=nethack_challenge --character=mon-hum-neu-mal --experiment=nethack_monk
```

Additionally it's possible to use an alternative `fast_eval_nethack` script which is much faster

```
python -m sf_examples.nethack.fast_eval_nethack --env=nethack_challenge --sample_env_episodes=128 --num_workers=16 --num_envs_per_worker=2 --character=mon-hum-neu-mal --experiment=nethack_monk 
```

### List of Supported Environments

- nethack_staircase
- nethack_score
- nethack_pet
- nethack_oracle
- nethack_gold
- nethack_eat
- nethack_scout
- nethack_challenge

### Reproducing Paper Results
Parameters for final experiments can be found in `mrunner_exps/.icml/2024_01_23_seeds`, in particular our best agent `mrunner_exps/.icml/2024_01_23_seeds/2024_01_25_monk-APPO-KS-T-baseline.py`

```bash
python -m sf_examples.nethack.train_nethack \
  --env=challenge \
  --train_for_env_steps=500000000 \
  --group=monk-APPO-KS-T \
  --character=mon-hum-neu-mal \
  --num_workers=16 \
  --num_envs_per_worker=32 \
  --worker_num_splits=2 \
  --rollout=32 \
  --batch_size=4096 \
  --async_rl=True \
  --serial_mode=False \
  --wandb_user=bartekcupial \
  --wandb_project=sf2_nethack \
  --wandb_group=gmum \
  --with_wandb=True \
  --use_pretrained_checkpoint=True \
  --model_path=/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_checkpoints/amzn-AA-BC_pretrained \
  --teacher_path=/net/pr2/projects/plgrid/plgggmum_crl/bcupial/sf_checkpoints/amzn-AA-BC_pretrained \
  --run_teacher_hs=False \
  --use_prev_action=True \
  --model=ScaledNet \
  --use_resnet=True \
  --learning_rate=0.0001 \
  --rnn_size=1738 \
  --h_dim=1738 \
  --exploration_loss_coeff=0.0 \
  --gamma=1.0 \
  --skip_train=25000000 \
  --lr_schedule=linear_decay \
  --save_milestones_ith=25000000 \
  --freeze="{'encoder': 0}" \
  --kickstarting_loss_decay=0.99998 \
  --min_kickstarting_loss_coeff=0.33 \
  --kickstarting_loss_coeff=0.75 
```

Wandb Report https://api.wandb.ai/links/bartekcupial/zo8agr4p

## Citation

If you use this repository in your work or otherwise wish to cite it, please make reference to our ICML2024 paper.

```
@article{wolczyk2024fine,
  title={Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem},
  author={Wo{\l}czyk, Maciej and Cupia{\l}, Bart{\l}omiej and Ostaszewski, Mateusz and Bortkiewicz, Micha{\l} and Zaj{\k{a}}c, Micha{\l} and Pascanu, Razvan and Kuci{\'n}ski, {\L}ukasz and Mi{\l}o{\'s}, Piotr},
  journal={arXiv preprint arXiv:2402.02868},
  year={2024}
}
```

