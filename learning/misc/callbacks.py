# Ref: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
from typing import Dict
import numpy as np
from collections import deque

import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class BasicCallbacks(DefaultCallbacks):
    def __init__(self, agent_ids, n_recent_ckpt=0, main_policy_id=None, ckpt_policy_id=None, 
                 ckpt_freq=1, legacy_callbacks_dict: Dict[str, callable] = None):
        super(BasicCallbacks, self).__init__(legacy_callbacks_dict)
        self.agent_ids = agent_ids
        self.recent_ckpt_weights = deque(maxlen=n_recent_ckpt)
        self.n_recent_ckpt = n_recent_ckpt
        self.main_policy_id = main_policy_id
        self.ckpt_policy_id = ckpt_policy_id
        self.ckpt_freq = ckpt_freq
        self.cnt = 0

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
                for k in info.keys():
                    if k in ['distance', 'model_velocity', 'success', 'passed_cars', 
                             'cum_collide', 'done_out_of_lane_or_max_rot', 'has_collided',
                             'off_lane', 'max_rot']:
                        episode.custom_metrics['{}/{}'.format(agent_id, k)] = info[k]

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        if self.recent_ckpt_weights.maxlen > 0:
            # keep most recent-N copies of checkpoint
            if ((result['training_iteration']-1) % self.ckpt_freq) == 0:
                local_main_policy_weights = trainer.workers.local_worker().get_policy(self.main_policy_id).get_weights()
                local_ckpt_policy_weights = trainer.workers.local_worker().get_policy(self.ckpt_policy_id).get_weights()
                weights_with_new_key = dict() # update keys for policy weights
                for (k1, v1), (k2, v2) in zip(local_main_policy_weights.items(), local_ckpt_policy_weights.items()):
                    weights_with_new_key[k2] = v1
                self.recent_ckpt_weights.append(weights_with_new_key)

            if False and self.cnt > 0: # NOTE: for sanity check
                # check if there are matches between current opponent policies' weights in all remote workers 
                # and any one of the recent checkpoint weights.
                opponent_weights = trainer.workers.foreach_policy(lambda p, pid: p.get_weights() if pid == 'default_policy_1' else None)
                for v in opponent_weights: # loop thru opponent policy from all workers
                    if v is None:
                        continue
                    allclose_list = []
                    for rckpt in self.recent_ckpt_weights: # loop thru all recent checkpoint weights
                        allclose = True
                        for kk, vv in rckpt.items():
                            allclose = allclose and np.allclose(vv, v[kk])
                        allclose_list.append(allclose)
                    assert np.any(allclose_list)
            
            # set weights for opponent policy of remote workers
            n_ckpt = len(self.recent_ckpt_weights)
            for r_w in trainer.workers.remote_workers():
                ckpt_idx = np.random.choice(np.arange(n_ckpt))
                weights = ray.put({self.ckpt_policy_id: self.recent_ckpt_weights[ckpt_idx]})
                r_w.set_weights.remote(weights)

            # set weights for opponent policy of the local worker
            ckpt_idx = np.random.choice(np.arange(n_ckpt))
            weights = {self.ckpt_policy_id: self.recent_ckpt_weights[ckpt_idx]}
            trainer.workers.local_worker().set_weights(weights)

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass
