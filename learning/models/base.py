import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.framework import get_activation_fn
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.rnn_sequencing import add_time_dimension

torch, nn = try_import_torch()


class Base(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, 
                 value_fcnet_hiddens,
                 value_fcnet_activation,
                 vec_obs_dim=0,
                 vec_branch_hiddens=None,
                 vec_branch_activation=None,
                 use_cnn=False,
                 use_recurrent=False,
                 with_bn=False,
                 **kwargs):
        nn.Module.__init__(self)
        super(Base, self).__init__(obs_space, action_space, num_outputs,
                                   model_config, name)
        self.use_recurrent = use_recurrent
        
        # define feature extractor
        self.use_cnn = use_cnn
        if self.use_cnn:
            # NOTE: cannot check obs shape for obs_space is originally a tuple obs space
            extractor_filters = model_config['conv_filters']
            extractor_activation = model_config['conv_activation']
            self.extractor = self._build_convnet(extractor_filters, extractor_activation, with_bn=with_bn)
            feat_channel = extractor_filters[-1][1]

            if vec_branch_hiddens is not None:
                self.vec_extractor = self._build_fcnet(vec_branch_hiddens, vec_branch_activation, with_bn=with_bn)
                feat_channel += vec_branch_hiddens[-1]
            else:
                feat_channel += vec_obs_dim
        else:
            assert obs_space.shape[0] == model_config['fcnet_hiddens'][0], \
                'Feature extractor input channel should be consistent with observation space'
            extractor_filters = model_config['fcnet_hiddens']
            extractor_activation = model_config['fcnet_activation']
            self.extractor = self._build_fcnet(extractor_filters, extractor_activation, with_bn=with_bn)
            feat_channel = extractor_filters[-1]
        self.feat_channel = feat_channel

        # define value function
        if model_config['vf_share_layers']:
            assert value_fcnet_hiddens[0] == self.feat_channel, \
                'Input channel of value function FCs should be consistent with feature extractor.'
        else:
            if self.use_cnn:
                self.vf_extractor = self._build_convnet(extractor_filters, 
                    extractor_activation, with_bn=with_bn)
                if vec_branch_hiddens is not None:
                    self.vf_vec_extractor = self._build_fcnet(vec_branch_hiddens, vec_branch_activation, with_bn=with_bn)
            else:
                self.vf_extractor = self._build_fcnet(extractor_filters, 
                    extractor_activation, with_bn=with_bn)
        assert value_fcnet_hiddens[-1] == 1, 'Last channel should be 1 in value function'
        self.vf_fcs = self._build_fcnet(value_fcnet_hiddens, value_fcnet_activation, 
            with_bn=with_bn, no_last_act=True) # NOTE: last layer is linear; not using NCP for value function

    def policy_inference(self):
        raise NotImplementedError

    @override(ModelV2)
    def get_initial_state(self):
        return None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # set train/eval mode for BN
        self.train(mode=input_dict.get('is_training', False))

        # feature extraction
        if self.use_cnn:
            if isinstance(input_dict['obs'], list):
                img_obs = input_dict['obs'][0].permute(0, 3, 1, 2).float()
                vec_obs = input_dict['obs'][1]
                feat = self.extractor(img_obs)
                if hasattr(self, 'vec_extractor'):
                    vec_feat = self.vec_extractor(vec_obs)
                else:
                    vec_feat = vec_obs
                feat = torch.cat([feat, vec_feat], -1)
                obs = [img_obs, vec_obs]
            else:
                obs = input_dict['obs'].permute(0, 3, 1, 2).float()
                feat = self.extractor(obs)
        else:
            obs = input_dict['obs_flat']
            feat = self.extractor(obs)
        self.value_inp = feat if self.model_config['vf_share_layers'] else obs

        # policy inference
        if self.use_recurrent:
            # convert to format for RNN inference
            if isinstance(seq_lens, np.ndarray):
                seq_lens = torch.Tensor(seq_lens).int()
            max_seq_len = feat.shape[0] // seq_lens.shape[0]
            self.time_major = self.model_config.get('_time_major', False)
            rnn_inputs = add_time_dimension(
                feat,
                max_seq_len=max_seq_len,
                framework='torch',
                time_major=self.time_major,
            )

            # recurrent inference
            output, new_state = self.policy_inference(rnn_inputs, state, seq_lens)
            output = torch.reshape(output, [-1, self.num_outputs])
        else:
            new_state = state
            output = self.policy_inference(feat, state, seq_lens)

        return output, new_state

    @override(ModelV2)
    def value_function(self):
        if self.model_config['vf_share_layers']:
            feat = self.value_inp
        else:
            if isinstance(self.value_inp, list):
                img_obs, vec_obs = self.value_inp
                feat = self.vf_extractor(img_obs)
                if hasattr(self, 'vf_vec_extractor'):
                    vec_feat = self.vf_vec_extractor(vec_obs)
                else:
                    vec_feat = vec_obs
                feat = torch.cat([feat, vec_feat], -1)
            else:
                feat = self.vf_extractor(self.value_inp)
        values = self.vf_fcs(feat)
        return torch.reshape(values, [-1])

    def _build_convnet(self, filters, activation, with_bn=False, no_last_act=False, to_nn_seq=True):
        activation = get_activation_fn(activation, 'torch')
        modules = nn.ModuleList()
        for filt in filters:
            modules.append(nn.Conv2d(*filt))
            if not no_last_act:
                if with_bn:
                    modules.append(nn.BatchNorm2d(filt[1]))
                modules.append(activation())
        modules.append(nn.AdaptiveAvgPool2d((1, 1)))
        modules.append(nn.Flatten())
        if to_nn_seq:
            modules = nn.Sequential(*modules)
        return modules

    def _build_fcnet(self, filters, activation, with_bn=False, no_last_act=False, to_nn_seq=True):
        activation = get_activation_fn(activation, 'torch')
        modules = nn.ModuleList()
        for i in range(len(filters)-1):
            modules.append(nn.Linear(filters[i], filters[i+1]))
            if not no_last_act:
                if with_bn:
                    modules.append(nn.BatchNorm1d(filters[i+1]))
                modules.append(activation())
        if to_nn_seq:
            modules = nn.Sequential(*modules)
        return modules
