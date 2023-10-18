import os
from functools import partial

import gym
import nle.dataset as nld
import numpy as np
from nle import _pyconverter as converter
from nle import nethack
from nle.dataset.dataset import TtyrecDataset, convert_frames
from nle.env.tasks import NetHackChallenge

from sample_factory.cfg.configurable import Configurable


class NetHackTtyrec(Configurable, NetHackChallenge, TtyrecDataset):
    def __init__(self, cfg, **kwargs):
        Configurable.__init__(self, cfg)
        NetHackChallenge.__init__(self, **kwargs)

        if not os.path.isfile(cfg.db_path):
            nld.db.create(cfg.db_path)
            nld.add_nledata_directory(cfg.data_path, cfg.dataset_name, cfg.db_path)

        subselect = []
        if cfg.character == "mon-hum-neu-mal":
            subselect.append(" role='Mon' AND race='Hum' ")
        if cfg.dataset_demigod:
            subselect.append(" death='ascended' ")
        if cfg.dataset_highscore:
            subselect.append(" points>10000")
        if cfg.dataset_midscore:
            subselect.append(" points>1000 AND points<10000")
        if cfg.dataset_deep:
            subselect.append(" maxlvl>1")

        subselect_sql = None
        if subselect:
            subselect_sql = "SELECT gameid FROM games WHERE " + "AND".join(subselect)

        nld.TtyrecDataset.__init__(
            self,
            dataset_name=cfg.dataset_name,
            dbfilename=cfg.db_path,
            batch_size=1,
            seq_length=1,
            shuffle=True,
            loop_forever=True,
            subselect_sql=subselect_sql,
            threadpool=None,
        )

        self.converter = converter.Converter(self.rows, self.cols, self._ttyrec_version)
        self._convert_frames = None

        self.prev_action = None
        self.prev_score = None
        self.last_observation = None

        self.embed_actions = np.zeros(256, dtype=np.uint8)
        for i, a in enumerate(self.actions):
            self.embed_actions[a.value] = i

        obs_spaces = {
            "prev_action": gym.spaces.Discrete(len(self.actions)),
            "action_converted": gym.spaces.Discrete(len(self.actions)),
        }
        obs_spaces.update([(k, self.observation_space[k]) for k in self.observation_space])
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _next_step(self):
        seq_length = self.seq_length
        rows = self.rows
        cols = self.cols

        chars = np.zeros((seq_length, rows, cols), dtype=np.uint8)
        colors = np.zeros((seq_length, rows, cols), dtype=np.int8)
        cursors = np.zeros((seq_length, 2), dtype=np.int16)
        timestamps = np.zeros((seq_length), dtype=np.int64)
        actions = np.zeros((seq_length), dtype=np.uint8)
        resets = np.zeros((seq_length), dtype=np.uint8)
        gameids = np.zeros((seq_length), dtype=np.int32)
        scores = np.zeros((seq_length), dtype=np.int32)

        key_vals = [
            ("tty_chars", chars),
            ("tty_colors", colors),
            ("tty_cursor", cursors),
            ("timestamps", timestamps),
            ("done", resets),
            ("gameids", gameids),
        ]

        ttyrec_version = self._ttyrec_version
        if ttyrec_version >= 2:
            key_vals.append(("keypresses", actions))
        if ttyrec_version >= 3:
            key_vals.append(("scores", scores))

        self._convert_frames(
            chars=chars,
            colors=colors,
            curs=cursors,
            timestamps=timestamps,
            actions=actions,
            scores=scores,
            resets=resets,
            gameids=gameids,
        )

        key_vals = dict(key_vals)

        key_vals["tty_cursor"] = key_vals["tty_cursor"].astype(np.uint8)
        key_vals["done"] = key_vals["done"].astype(bool)
        key_vals["keypresses"] = key_vals["keypresses"].astype(int)
        for key, value in key_vals.items():
            if value.size == 1:
                # key_vals[key] = value.item()
                pass
            else:
                key_vals[key] = value.squeeze(0)

        # get action
        key_vals["prev_action"] = self.prev_action
        action_converted = self.embed_actions[key_vals["keypresses"]]
        key_vals["action_converted"] = action_converted
        self.prev_action = action_converted

        # get reward
        reward = key_vals["scores"].item() - self.prev_score
        self.prev_score = key_vals["scores"].item()

        # get done
        done = key_vals["done"].item()

        # TODO: get info
        info = {}

        # get observation
        observation = {
            key: value
            for key, value in key_vals.items()
            if key in ["tty_chars", "tty_colors", "tty_cursor", "prev_action", "action_converted"]
        }
        self.last_observation = observation

        return observation, reward, done, info

    def step(self, action):
        # Ignore the action, sample the next step from the dataset
        return self._next_step()

    def reset(self):
        gameids = list(self._gameids)
        if self.shuffle:
            np.random.shuffle(gameids)

        load_fn = self._make_load_fn(gameids)
        self.gameids = gameids
        assert load_fn(self.converter)
        self._convert_frames = partial(convert_frames, converter=self.converter, load_fn=load_fn)

        self.prev_action = np.zeros(1, dtype=np.uint8)
        self.prev_score = 0
        self.last_observation = None
        key_vals, reward, done, info = self._next_step()

        return key_vals

    def render(self, mode="human"):
        # if mode == "human":
        obs = self.last_observation
        tty_chars = obs["tty_chars"]
        tty_colors = obs["tty_colors"]
        tty_cursor = obs["tty_cursor"]
        print(nethack.tty_render(tty_chars, tty_colors, tty_cursor))
        return
