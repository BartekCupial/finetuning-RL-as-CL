import os

import gymnasium as gym
import nle.dataset as nld
import numpy as np
import render_utils
import torch

from sample_factory.cfg.configurable import Configurable
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log


class NetHackDataset(Configurable):
    def __init__(self, cfg, threadpool):
        super().__init__(cfg)

        self.threadpool = threadpool

        self.dataset_batch_size = cfg.dataset_batch_size // cfg.dataset_rollout
        self.dataset = load_nld_dataset(
            threadpool=self.threadpool,
            data_path=cfg.data_path,
            db_path=cfg.db_path,
            dataset_name=cfg.dataset_name,
            seq_len=cfg.dataset_rollout,
            batch_size=self.dataset_batch_size,
            character=cfg.character,
            dataset_demigod=cfg.dataset_demigod,
            dataset_highscore=cfg.dataset_highscore,
            dataset_midscore=cfg.dataset_midscore,
            dataset_deep=cfg.dataset_deep,
        )

        self.env = create_env(cfg.env, cfg=cfg)
        obs = self.env.reset()

        embed_actions = torch.zeros((256, 1))
        for i, a in enumerate(self.env.actions):
            embed_actions[a.value][0] = i
        self.embed_actions = torch.nn.Embedding.from_pretrained(embed_actions)
        self.char_array = np.ascontiguousarray(self.env.char_array)
        self.crop_dim = cfg.crop_dim

        self.dataset_warmup = cfg.dataset_warmup
        self.prev_action_shape = (self.dataset_batch_size, cfg.dataset_rollout)
        self.screen_shape = self.prev_action_shape + obs[0]["screen_image"].shape

        obs_spaces = {
            "prev_action": self.env.action_space,
            "actions_converted": self.env.action_space,
            "dones": gym.spaces.Discrete(2),
            "mask": gym.spaces.Discrete(2),
        }
        obs_spaces.update(
            [
                (k, self.env.observation_space[k])
                for k in self.env.observation_space
                if k not in ["message", "blstats"]  # our dataset doesn't contain those keys
            ]
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.action_space = self.env.action_space

    def make_single_iter(self, dataset):
        def _iter():
            prev_action = np.zeros((self.dataset_batch_size, 1), dtype=np.uint8)
            prev_mb = {
                key: np.zeros((self.dataset_batch_size, 1, *obs_space.shape), dtype=np.uint8)
                for key, obs_space in self.observation_space.items()
            }
            while True:
                for i, mb in enumerate(dataset):
                    if i < self.dataset_warmup:
                        continue

                    # flake8: noqa
                    screen_image = np.zeros(self.screen_shape, dtype=np.uint8)
                    cursor_uint8 = mb["tty_cursor"].astype(np.uint8)
                    convert = lambda i: render_utils.render_crop(
                        mb["tty_chars"][i],
                        mb["tty_colors"][i],
                        cursor_uint8[i],
                        self.char_array,
                        screen_image[i],
                        self.crop_dim,
                    )
                    list(self.threadpool.map(convert, range(self.dataset_batch_size)))

                    final_mb = {
                        "tty_chars": mb["tty_chars"],
                        "tty_colors": mb["tty_colors"],
                        "tty_cursor": cursor_uint8,
                        "screen_image": screen_image,
                        "dones": mb["done"].astype(bool),
                        "mask": np.ones_like(mb["done"]).astype(bool),
                    }

                    if "keypresses" in mb:
                        actions = mb["keypresses"].astype(int)
                        actions_converted = (
                            self.embed_actions(torch.from_numpy(actions)).squeeze(-1).numpy().astype(np.uint8)
                        )
                        final_mb["actions_converted"] = actions_converted
                        final_mb["prev_action"] = np.concatenate([prev_action, actions_converted[:, :-1]], axis=1)
                        prev_action = actions_converted[:, -1:]

                    # we need to allocate an extra rollout step here to calculate the value estimates for the last step
                    for key in final_mb.keys():
                        prev_obs = prev_mb[key]
                        final_mb[key] = np.concatenate([prev_obs, final_mb[key]], axis=1)
                        prev_mb[key] = final_mb[key][:, -1:]

                    yield final_mb

        return iter(_iter())


def load_nld_dataset(
    threadpool,
    data_path: str,
    db_path: str,
    dataset_name: str,
    seq_len: int,
    batch_size: int,
    character: str = "@",
    dataset_demigod: bool = False,
    dataset_highscore: bool = False,
    dataset_midscore: bool = False,
    dataset_deep: bool = False,
) -> nld.TtyrecDataset:
    if not os.path.isfile(db_path):
        nld.db.create(db_path)
        nld.add_nledata_directory(data_path, dataset_name, db_path)

    subselect = []
    if character == "mon-hum-neu-mal":
        subselect.append(" role='Mon' AND race='Hum' ")
    if dataset_demigod:
        subselect.append(" death='ascended' ")
    if dataset_highscore:
        subselect.append(" points>10000")
    if dataset_midscore:
        subselect.append(" points>1000 AND points<10000")
    if dataset_deep:
        subselect.append(" maxlvl>1")

    subselect_sql = None
    if subselect:
        subselect_sql = "SELECT gameid FROM games WHERE " + "AND".join(subselect)

    dataset = nld.TtyrecDataset(
        dataset_name=dataset_name,
        dbfilename=db_path,
        batch_size=batch_size,
        seq_length=seq_len,
        shuffle=True,
        loop_forever=True,
        subselect_sql=subselect_sql,
        threadpool=threadpool,
    )
    log.info(f"Total games in the filtered dataset: {len(dataset._gameids)}")

    return dataset
