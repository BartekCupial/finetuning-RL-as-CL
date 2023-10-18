from concurrent.futures import ThreadPoolExecutor

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.cfg.configurable import Configurable
from sf_examples.nethack.datasets.dataset import NetHackDataset


class DatasetRolloutWorker(Configurable):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.threadpool = ThreadPoolExecutor(max_workers=self.cfg.dataset_num_workers)

        self.dataset = NetHackDataset(cfg, self.threadpool)

        self._iterators = []
        self._results = []
        for _ in range(self.cfg.dataset_num_splits):
            it = self.dataset.make_single_iter(self.dataset.dataset)
            self._iterators.append(it)
            self._results.append(self.threadpool.submit(next, it))

    def sample_batch(self, batch_idx):
        batch = self._results[batch_idx].result()

        fut = self.threadpool.submit(next, self._iterators[batch_idx])
        self._results[batch_idx] = fut

        return TensorDict(batch)
