import os
import sys
import argparse
import pickle5 as pickle
import numpy as np
import torch
import gym

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import misc
import models


class ExportedPolicy(object):
    def __init__(self, ckpt_path, is_exported=True, temp_dir='~/tmp', run='PPO'):
        # Load config
        config_dir = os.path.dirname(ckpt_path)
        config_path = os.path.join(config_dir, 'params.pkl')
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, '../params.pkl')
        
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        self.config = config
    
        # Export checkpoint
        if not is_exported:
            ckpt_path = self._export_model(ckpt_path, self.config, temp_dir, run)
            print('Export model to {}'.format(ckpt_path))

        # Load exported model
        model_cls = getattr(models, self.config['model']['custom_model'])
        model_config = dict()
        for k, v in self.config['model'].items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    model_config[kk] = vv
            else:
                model_config[k] = v

        ckpt = torch.load(ckpt_path)
        self.model = model_cls(**ckpt['env'], **model_config, model_config=self.config['model'])
        self.model.load_state_dict(ckpt['model'])

        if False: # TODO: make independent of rllib object
            from ray.rllib import utils as rllib_utils
            from ray.rllib.models.preprocessors import get_preprocessor
            self.preprocessor = get_preprocessor(ckpt['env']['obs_space'])(ckpt['env']['obs_space'])
            self.obs_filter = getattr(rllib_utils.filter, self.config['observation_filter'])(self.preprocessor.shape)
        else:
            self.preprocessor = ckpt['preprocessors']['default_policy']
            self.obs_filter = ckpt['filters']['default_policy']

        self.act_dist_cls = getattr(models, self.config['model']['custom_action_dist'])
        self.act_dist_config = self.config['model']['custom_model_config']['custom_action_dist_config']

    def compute_action(self, obs, state, seq_lens, stochastic=False, input_mode=0):
        # prepare input
        if input_mode == 0:
            pp_obs = self.preprocessor.transform(obs)
            filtered_obs = self.obs_filter(pp_obs, update=False)
            filtered_obs = torch.Tensor(filtered_obs).cuda()
            input_dict = {'obs_flat': filtered_obs[None,:]}
            if isinstance(self.model.obs_space, gym.spaces.Tuple):
                obs_list = torch.split(filtered_obs, [np.prod(osp.shape) for osp in self.model.obs_space])
                obs_list = list(obs_list)
                for i in range(len(obs_list)):
                    obs_list[i] = obs_list[i].reshape(self.model.obs_space[i].shape)[None,...]
                input_dict['obs'] = obs_list
            else:
                input_dict['obs'] = filtered_obs.reshape(self.model.obs_space.shape)[None,...]
        elif input_mode == 1: # already preprocessed and filtered; also augmented with batch dimension
            input_dict = {'obs_flat': obs}
            if isinstance(self.model.obs_space, gym.spaces.Tuple):
                obs_list = torch.split(obs, [np.prod(osp.shape) for osp in self.model.obs_space], dim=-1)
                obs_list = list(obs_list)
                for i, o in enumerate(obs_list):
                    obs_list[i] = o.reshape(o.shape[0], *self.model.obs_space[i].shape)
                input_dict['obs'] = obs_list
            else:
                input_dict['obs'] = obs
        else:
            raise ValueError('Invalid input mode of ExportedPolicy')
        
        # network inference
        with torch.no_grad():
            act_dist_inputs, new_state = self.model.forward(input_dict, state, seq_lens)

        # action distribution
        act_dist = self.act_dist_cls(act_dist_inputs, self.model, **self.act_dist_config)
        act = act_dist.sample() if stochastic else act_dist.deterministic_sample()
        act_info = {
            'act_dist': act_dist,
            'action_dist_inputs': act_dist_inputs.cpu().numpy(),
            'action_logp': act_dist.logp(act).cpu().numpy(),
            'rllib_action_logp': act_dist.logp(act).cpu().numpy(),
        }
        act = act.cpu().numpy()[0]
        return act, new_state, act_info

    def _export_model(self, ckpt_path, config, temp_dir, run):
        import ray
        from ray.rllib.evaluation.worker_set import WorkerSet
        from ray.tune.registry import get_trainable_cls

        # Overwrite some config with arguments
        config['num_workers'] = 0
        config['num_gpus'] = 0

        # Register custom model
        misc.register_custom_env(config['env'])
        misc.register_custom_model(config['model'])

        # Start ray
        temp_dir = os.path.abspath(os.path.expanduser(temp_dir))
        ray.init(
            local_mode=True,
            _temp_dir=temp_dir,
            include_dashboard=False)

        # Get agent
        cls = get_trainable_cls(run)
        agent = cls(env=config['env'], config=config)
        agent.restore(ckpt_path)

        # Export model
        config_dir = os.path.dirname(ckpt_path)
        export_path = os.path.join(config_dir, 'exported_model.pkl')
        state_dict = agent.get_policy().model.state_dict()

        env = agent.workers.local_worker().env
        obs_space = env.observation_space
        act_space = env.action_space
        model_name = agent.get_policy().model.name
        model_num_outputs = agent.get_policy().model.num_outputs

        saved_data = {
            'model': state_dict, 
            'env': {
                'obs_space': obs_space,
                'action_space': act_space,
                'name': model_name,
                'num_outputs': model_num_outputs
            },
            'preprocessors': agent.workers.local_worker().preprocessors,
            'filters': agent.workers.local_worker().filters
        }
        torch.save(saved_data, export_path)

        # Close agent and ray
        ray.shutdown()

        return export_path


def main():
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'ckpt_path',
        type=str,
        help='Checkpoint from which to roll out.')
    parser.add_argument(
        '--export',
        action='store_true',
        help='Whether to export the model or test on the exported model.')
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Use monitor wrapper.')
    args = parser.parse_args()

    # create policy
    policy = ExportedPolicy(args.ckpt_path, is_exported=not args.export)
    policy.model.cuda()

    # create env
    env_creator = misc.register_custom_env(policy.config['env'])
    env = env_creator(env_config=policy.config['env_config'])
    if args.monitor:
        from envs.wrappers import MultiAgentMonitor
        env = MultiAgentMonitor(env, os.path.expanduser('~/tmp/monitor'), video_callable=lambda x: True, force=True)

    # step environments
    for ep in range(10):
        done = False
        ep_steps = 0
        ep_rew = 0
        obs = env.reset()
        while not done:
            act = dict()
            for _i, k in enumerate(env.controllable_agents.keys()):
                if True: # use exported policy
                    a_obs = obs[k]
                    a_act, _, a_act_info = policy.compute_action(a_obs, None, None)
                    act[k] = a_act
                    if False: # for debugging; need to set agent while exporting
                        a_act_rllib, _, a_act_info_rllib = policy.agent.compute_action(a_obs,
                            None, policy_id='default_policy', full_fetch=True)
                elif False: # follow human trajectory
                    ts = env.world.agents[_i].get_current_timestamp()
                    act[k] = env.world.agents[_i].trace.f_curvature(ts)
                else: # random action
                    act[k] = env.action_space.sample()
            obs, rew, done, info = env.step(act)
            ep_rew += np.mean(list(rew.values()))
            done = np.any(list(done.values()))
            ep_steps += 1
        print('[{}th episodes] {} steps {} reward'.format(ep, ep_steps, ep_rew))


if __name__ == '__main__':
    main()