import copy
from functools import partialmethod


class PolicyManager(object):
    def __init__(self, env_creator, config):
        self.env_creator = env_creator
        self.env = self.env_creator(config['env_config'])
        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space
        self.config = config
        self.policy_ids = []

    def get_policy(self, policy_id):
        single_config = copy.deepcopy(self.config) # otherwise will cause "ValueError: Circular reference detected"
        if 'default_policy' in policy_id:
            policy = (None, self.obs_space, self.act_space, single_config)
        else:
            raise ValueError('Invalid policy ID {}'.format(policy_id))
        self.policy_ids.append(policy_id)

        return policy

    def get_policy_mapping_fn(self, policy_mapping_fn):
        if policy_mapping_fn == 'all_to_default':
            fn = lambda agent_id: 'default_policy'
        elif policy_mapping_fn == 'one_to_one':
            fn = lambda agent_id: 'default_policy_{}'.format(agent_id.split('_')[-1])
        else:
            raise ValueError('Invalid policy_mapping_fn {}'.format(policy_mapping_fn))

        return fn


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return NewCls