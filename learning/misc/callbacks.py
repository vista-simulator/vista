# Ref: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
from typing import Dict
import numpy as np

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class BasicCallbacks(DefaultCallbacks):
    def __init__(self, agent_ids, legacy_callbacks_dict: Dict[str, callable] = None):
        super(BasicCallbacks, self).__init__(legacy_callbacks_dict)
        self.agent_ids = agent_ids

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        pass

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        pass

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        for agent_id in self.agent_ids:
            info = episode.last_info_for(agent_id)
            if info:
                for k in ['distance', 'model_velocity']:
                    episode.custom_metrics['{}/{}'.format(agent_id, k)] = info[k]

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        pass

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass