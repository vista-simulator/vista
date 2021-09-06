from typing import List, Dict, Any, Optional
import numpy as np
import torch

import vista
from .buffered_dataset import BufferedDataset
from .utils import transform_lidar

__all__ = ['VistaDataset', 'worker_init_fn']


class VistaDataset(BufferedDataset):
    def __init__(self,
                 trace_paths: List[str],
                 trace_config: Dict[str, Any],
                 car_config: Dict[str, Any],
                 lidar_config: Dict[str, Any],
                 train: Optional[bool] = False,
                 buffer_size: Optional[int] = 1,
                 snippet_size: Optional[int] = 100,
                 shuffle: Optional[bool] = False,
                 **kwargs):
        super(VistaDataset,
              self).__init__(trace_paths, trace_config, car_config, train,
                             buffer_size, snippet_size, shuffle)

        self._lidar_config = lidar_config

    def _simulate(self):
        # Initialization for singl-processing dataloader
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self._world = vista.World(self.trace_paths, self.trace_config)
            self._agent = self._world.spawn_agent(self.car_config)
            self._lidar = self._agent.spawn_lidar(self.lidar_config)
            self._world.reset()

        # Data generator from simulation
        self._snippet_i = 0
        while True:
            # reset simulator
            if self._agent.done or self._snippet_i >= self.snippet_size:
                if worker_info is not None:
                    self._world.set_seed(worker_info.id)
                self._world.reset()
                self._snippet_i = 0

            # step simulator
            sensor_name = self._lidar.name
            # associate action t with observation t-1
            pcd = self._agent.observations[sensor_name]
            self._agent.step_dataset(step_dynamics=False)

            # preprocess and produce data-label pairs
            data = transform_lidar(pcd, self._lidar, self.train)
            label = np.array([self._agent.human_curvature]).astype(np.float32)

            self._snippet_i += 1

            yield {'lidar': data, 'target': label}

    @property
    def lidar_config(self) -> Dict[str, Any]:
        return self._lidar_config


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset._world = vista.World(dataset.trace_paths, dataset.trace_config)
    dataset._agent = dataset._world.spawn_agent(dataset.car_config)
    dataset._lidar = dataset._agent.spawn_lidar(dataset.lidar_config)
    dataset._world.set_seed(worker_id)
    dataset._world.reset()
