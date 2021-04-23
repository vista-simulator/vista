import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.framework import get_activation_fn
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.rnn_sequencing import add_time_dimension

import kerasncp as kncp
from kerasncp.torch import LTCCell

from .base import Base

torch, nn = try_import_torch()


class ConvNet(Base):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 value_fcnet_hiddens,
                 value_fcnet_activation,
                 vec_obs_dim=0,
                 vec_branch_hiddens=None,
                 vec_branch_activation=None,
                 use_cnn=True,
                 use_recurrent=False,
                 with_bn=False,
                 policy_hiddens=None,
                 policy_activation=None,
                 **kwargs):
        assert use_cnn, 'ConvNet must have use_cnn = True'
        assert not use_recurrent, 'ConvNet doesn\'t use recurrent network'
        nn.Module.__init__(self)
        super(ConvNet, self).__init__(obs_space, action_space, num_outputs,
                                      model_config, name, value_fcnet_hiddens,
                                      value_fcnet_activation, vec_obs_dim, vec_branch_hiddens, 
                                      vec_branch_activation, use_cnn, use_recurrent, with_bn, **kwargs)

        # define policy
        assert policy_hiddens[0] == self.feat_channel
        assert policy_hiddens[-1] == self.num_outputs
        self.policy = self._build_fcnet(policy_hiddens, policy_activation, with_bn=with_bn, no_last_act=True)

    def policy_inference(self, inputs, state, seq_lens):
        out = self.policy(inputs)

        return out, state