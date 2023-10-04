import os
from concurrent.futures import ThreadPoolExecutor

import nle.dataset as nld
import numpy as np
import render_utils
import torch

from sample_factory.cfg.configurable import Configurable
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log


class TtyrecDatasetWorker(Configurable):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.idx = 0
        self.dataset_num_splits = cfg.dataset_num_splits
        self.device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
        self.threadpool = ThreadPoolExecutor(max_workers=self.cfg.dataset_num_workers)

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

        env = create_env(cfg.env, cfg=cfg)
        obs = env.reset()

        embed_actions = torch.zeros((256, 1))
        for i, a in enumerate(env.actions):
            embed_actions[a.value][0] = i
        self.embed_actions = torch.nn.Embedding.from_pretrained(embed_actions)
        self.embed_actions = self.embed_actions.to(self.device)
        self.char_array = np.ascontiguousarray(env.char_array)
        self.crop_dim = cfg.crop_dim

        self.dataset_warmup = cfg.dataset_warmup
        self.prev_action_shape = (self.dataset_batch_size, cfg.dataset_rollout)
        self.screen_shape = self.prev_action_shape + obs[0]["screen_image"].shape

        self._iterators = []
        self._results = []
        for _ in range(self.dataset_num_splits):
            it = self.make_single_iter(self.dataset)
            self._iterators.append(it)
            self._results.append(self.threadpool.submit(next, it))

    def result(self):
        return self._results[self.idx].result()

    def step(self):
        fut = self.threadpool.submit(next, self._iterators[self.idx])
        self._results[self.idx] = fut
        self.idx = (self.idx + 1) % self.dataset_num_splits

    def make_single_iter(self, dataset):
        def _iter():
            mb_tensors = {
                "screen_image": torch.zeros(self.screen_shape, dtype=torch.uint8),
                "prev_action": torch.zeros(self.prev_action_shape, dtype=torch.uint8),
            }

            prev_action = torch.zeros((self.dataset_batch_size, 1), dtype=torch.uint8).to(self.device)
            while True:
                for i, mb in enumerate(dataset):
                    if i == 0:
                        # create torch tensors from first minibatch
                        screen_image = mb_tensors["screen_image"].numpy()
                        for k, array in mb.items():
                            mb_tensors[k] = torch.from_numpy(array)
                        [v.pin_memory() for v in mb_tensors.values()]

                    if i < self.dataset_warmup:
                        continue

                    cursor_uint8 = mb["tty_cursor"].astype(np.uint8)
                    # flake8: noqa
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
                        "tty_chars": mb_tensors["tty_chars"],
                        "tty_colors": mb_tensors["tty_colors"],
                        "tty_cursor": torch.from_numpy(cursor_uint8),
                        "screen_image": mb_tensors["screen_image"],
                        "dones": mb_tensors["done"].bool(),
                        "mask": torch.ones_like(mb_tensors["timestamps"]).bool(),
                    }

                    if "keypresses" in mb_tensors:
                        actions = mb_tensors["keypresses"].long().to(self.device)
                        actions_converted = self.embed_actions(actions).squeeze(-1).long()
                        final_mb["actions_converted"] = actions_converted
                        final_mb["prev_action"] = torch.cat([prev_action, actions_converted[:, :-1]], dim=1)
                        prev_action = actions_converted[:, -1:]

                    # DATASET is: [B T ...] but MODEL expects [T B ...]
                    data = {k: t.transpose(0, 1).to(self.device) for k, t in final_mb.items()}

                    yield data

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
