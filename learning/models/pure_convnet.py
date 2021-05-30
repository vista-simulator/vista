import os
import sys
import numpy as np
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class PureConvNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super(PureConvNet, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        conv_filters = model_config['custom_model_config'].get('conv_filters', None)
        conv_activation = model_config['custom_model_config'].get('conv_activation', 'relu')
        with_bn = model_config['custom_model_config'].get('with_bn', False)
        fc_hiddens = model_config['custom_model_config'].get('fc_hiddens', None)
        fc_activation = model_config['custom_model_config'].get('fc_activation', 'relu')
        fc_dropout = model_config['custom_model_config'].get('fc_dropout', 0.)

        self.convnet = self._build_convnet(conv_filters, conv_activation, with_bn=with_bn)
        self.fcnet = self._build_fcnet(fc_hiddens, fc_activation, no_last_act=True, dropout=fc_dropout)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # set train/eval mode for BN
        self.train(mode=input_dict.get('is_training', False))
        if isinstance(input_dict['obs'], list):
            obs = input_dict['obs'][0].permute(0, 3, 1, 2).float()
            vec_obs = input_dict['obs'][1]
        else:
            obs = input_dict['obs'].permute(0, 3, 1, 2).float()
        # import cv2; cv2.imwrite('/home/gridsan/tsunw/workspace/vista-integrate-new-api/learning/test.png', obs[0].permute(1,2,0).cpu().numpy().astype(np.uint8))
        feat = self.convnet(obs)
        if isinstance(input_dict['obs'], list):
            feat = torch.cat([feat,vec_obs], 1)
        output = self.fcnet(feat)
        return output, state

    def _build_convnet(self, filters, activation, with_bn=False, no_last_act=False, dropout=0., to_nn_seq=True):
        activation = get_activation_fn(activation, 'torch')
        modules = nn.ModuleList()
        for i, filt in enumerate(filters):
            modules.append(nn.Conv2d(*filt))
            if not (no_last_act and i == len(filters)-1):
                if with_bn:
                    modules.append(nn.BatchNorm2d(filt[1]))
                modules.append(activation())
                if dropout > 0.:
                    modules.append(nn.Dropout2d(p=dropout))
        modules.append(nn.AdaptiveAvgPool2d((1, 1)))
        modules.append(nn.Flatten())
        if to_nn_seq:
            modules = nn.Sequential(*modules)
        return modules

    def _build_fcnet(self, filters, activation, with_bn=False, no_last_act=False, dropout=0., to_nn_seq=True):
        activation = get_activation_fn(activation, 'torch')
        modules = nn.ModuleList()
        for i in range(len(filters)-1):
            modules.append(nn.Linear(filters[i], filters[i+1]))
            if not (no_last_act and i == len(filters)-2):
                if with_bn:
                    modules.append(nn.BatchNorm1d(filters[i+1]))
                modules.append(activation())
                if dropout > 0.:
                    modules.append(nn.Dropout(p=dropout))
        if to_nn_seq:
            modules = nn.Sequential(*modules)
        return modules