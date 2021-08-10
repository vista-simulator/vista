import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

from .base import Base

torch, nn = try_import_torch()


class ConvNet(Base):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 value_fcnet_hiddens=None,
                 value_fcnet_activation=None,
                 value_fcnet_dropout=0.,
                 vec_obs_dim=0,
                 vec_branch_hiddens=None,
                 vec_branch_activation=None,
                 use_cnn=True,
                 use_recurrent=False,
                 with_bn=False,
                 policy_hiddens=None,
                 policy_activation=None,
                 policy_dropout=0.,
                 auto_append_policy_hiddens=False,
                 post_lstm_dropout=False,
                 **kwargs):
        assert use_cnn, 'ConvNet must have use_cnn = True'
        nn.Module.__init__(self)
        super(ConvNet, self).__init__(obs_space, action_space, num_outputs, model_config, name, 
                                      value_fcnet_hiddens=value_fcnet_hiddens, 
                                      value_fcnet_activation=value_fcnet_activation, 
                                      value_fcnet_dropout=value_fcnet_dropout, 
                                      vec_obs_dim=vec_obs_dim, 
                                      vec_branch_hiddens=vec_branch_hiddens, 
                                      vec_branch_activation=vec_branch_activation, 
                                      use_cnn=use_cnn, 
                                      use_recurrent=use_recurrent,
                                      with_bn=with_bn, 
                                      auto_append_policy_hiddens=auto_append_policy_hiddens,
                                      **kwargs)

        # define policy
        if self.use_recurrent:
            assert not auto_append_policy_hiddens, 'Do not use auto_append_policy_hiddens with recurrent network.'
        if policy_hiddens[0] != self.feat_channel and auto_append_policy_hiddens:
            print('The channel of the first layer in policy does not equal to that of feature extraction output. Append!!')
            policy_hiddens = [self.feat_channel] + policy_hiddens
        assert policy_hiddens[-1] == self.num_outputs
        if self.use_recurrent:
            self.cell_size = model_config['lstm_cell_size']
            assert self.rnn_num_layers == 1, 'Multi-layer LSTM is not ready yet.'
            self.pre_lstm_dropout = nn.Dropout(policy_dropout)
            self.lstm = nn.LSTM(self.feat_channel, self.cell_size, batch_first=not self.time_major, 
                                num_layers=self.rnn_num_layers)
            if post_lstm_dropout:
                self.post_lstm_dropout = nn.Dropout(policy_dropout)
        self.policy = self._build_fcnet(policy_hiddens, policy_activation, with_bn=with_bn,
                                        dropout=policy_dropout, no_last_act=True)

    def policy_inference(self, inputs, state, seq_lens):
        if self.use_recurrent:
            self.lstm.train() # HACKY
            # inference lstm
            inputs = self.pre_lstm_dropout(inputs)
            if self.rnn_num_layers > 1:
                lstm_feat, [h, c] = self.lstm(inputs, [state[0].permute(1,0,2).contiguous(), state[1].permute(1,0,2).contiguous()])
            else:
                lstm_feat, [h, c] = self.lstm(inputs, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
            state = [torch.squeeze(h, 0), torch.squeeze(c, 0)]
            if hasattr(self, 'post_lstm_dropout'):
                lstm_feat = self.post_lstm_dropout(lstm_feat)
            out = self.policy(lstm_feat)
            # update value function input
            self.value_inp = lstm_feat
        else:
            out = self.policy(inputs)

        return out, state

    @override(ModelV2)
    def get_initial_state(self):
        device = next(self.parameters()).device
        if self.use_recurrent:
            if self.rnn_num_layers > 1:
                return [
                    torch.zeros((self.rnn_num_layers, self.cell_size), dtype=torch.float32).to(device),
                    torch.zeros((self.rnn_num_layers, self.cell_size), dtype=torch.float32).to(device)
                ]
            else:
                return [
                    torch.zeros((self.cell_size), dtype=torch.float32).to(device),
                    torch.zeros((self.cell_size), dtype=torch.float32).to(device)
                ]
        else:
            return torch.zeros((1,))