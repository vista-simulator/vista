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


class SegPretrainedNet(Base):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 value_fcnet_hiddens,
                 value_fcnet_activation,
                 value_fcnet_dropout=0.,
                 vec_obs_dim=0,
                 vec_branch_hiddens=None,
                 vec_branch_activation=None,
                 use_recurrent=False,
                 seg_model_name=None,
                 seg_out_channel=16,
                 feat_roi_crop=None,
                 policy_hiddens=None,
                 policy_activation=None,
                 policy_dropout=0.,
                 **kwargs):
        assert not use_recurrent, 'SegPretrainedNet doesn\'t use recurrent network'
        use_cnn = True
        use_seg_pretrained = True
        with_bn = False
        nn.Module.__init__(self)
        super(SegPretrainedNet, self).__init__(obs_space, action_space, num_outputs, model_config, name, 
                                               value_fcnet_hiddens=value_fcnet_hiddens, 
                                               value_fcnet_activation=value_fcnet_activation, 
                                               value_fcnet_dropout=value_fcnet_dropout, 
                                               vec_obs_dim=vec_obs_dim, 
                                               vec_branch_hiddens=vec_branch_hiddens, 
                                               vec_branch_activation=vec_branch_activation, 
                                               use_cnn=use_cnn, 
                                               use_recurrent=use_recurrent,
                                               with_bn=with_bn,
                                               use_seg_pretrained=use_seg_pretrained, 
                                               seg_model_name=seg_model_name, 
                                               seg_out_channel=seg_out_channel, 
                                               feat_roi_crop=feat_roi_crop, 
                                               **kwargs)

        # define policy
        assert policy_hiddens[0] == self.feat_channel
        assert policy_hiddens[-1] == self.num_outputs
        self.policy = self._build_fcnet(policy_hiddens, policy_activation, with_bn=with_bn,
                                        dropout=policy_dropout, no_last_act=True)

    def policy_inference(self, inputs, state, seq_lens):
        out = self.policy(inputs)

        return out, state