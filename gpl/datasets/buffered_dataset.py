from typing import List, Dict, Any, Optional
from tqdm import tqdm
import random
import torch
from torch.utils.data import IterableDataset


class BufferedDataset(IterableDataset):
    def __init__(self,
                 trace_paths: List[str],
                 trace_config: Dict[str, Any],
                 car_config: Dict[str, Any],
                 train: Optional[bool] = False,
                 buffer_size: Optional[int] = 1,
                 snippet_size: Optional[int] = 100,
                 shuffle: Optional[bool] = False):
        self._trace_paths = trace_paths
        self._trace_config = trace_config
        self._car_config = car_config

        self._train = train
        self._buffer_size = max(1, buffer_size)
        self._snippet_size = max(1, snippet_size)
        self._shuffle = shuffle

        self._rng = None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        self._rng = random.Random(worker_id)
        buf = []

        pbar = tqdm(total=self.buffer_size)
        for x in self._simulate():
            if len(buf) == self.buffer_size:
                if self.shuffle:
                    idx = self._rng.randint(0, self.buffer_size - 1)
                else:
                    idx = 0  # FIFO
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
                pbar.update(1)

        if self.shuffle:
            self._rng.shuffle(buf)
        while buf:
            yield buf.pop()

    def _simulate(self):
        raise NotImplementedError('Please implement custom simulate function')

    @property
    def trace_paths(self) -> List[str]:
        return self._trace_paths

    @property
    def trace_config(self) -> Dict[str, Any]:
        return self._trace_config

    @property
    def car_config(self) -> Dict[str, Any]:
        return self._car_config

    @property
    def train(self) -> bool:
        return self._train

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def snippet_size(self) -> int:
        return self._snippet_size

    @property
    def shuffle(self) -> int:
        return self._shuffle
