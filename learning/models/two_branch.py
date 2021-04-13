import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import get_activation_fn


class TwoBranch(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 merge_fcnet_hiddens, merge_fcnet_activation, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.convnet = self._build_convnet(model_config)
        self.fcnet = self._build_fcnet(model_config)
        self.merge_fcnet = self._build_fcnet(model_config, filters=merge_fcnet_hiddens)
        self.action_fc = nn.Linear(merge_fcnet_hiddens[-1], num_outputs)
        if model_config["vf_share_layers"]:
            self.value_func = nn.Linear(merge_fcnet_hiddens[-1], 1)
        else:
            self.value_convnet = self._build_convnet(model_config)
            self.value_fcnet = self._build_fcnet(model_config)
            self.value_merge_fcnet = self._build_fcnet(model_config, filters=merge_fcnet_hiddens)
            self.value_func = nn.Linear(merge_fcnet_hiddens[-1], 1)

    def forward(self, input_dict, state, seq_lens):
        vis_obs = input_dict["obs"][0].float().permute(0, 3, 1, 2)
        vis_feat = self.convnet(vis_obs)
        vis_feat = vis_feat.flatten(start_dim=1)
        vis_feat = self.fcnet(vis_feat)

        vec_obs = input_dict["obs"][1]
        
        feat = torch.cat([vis_feat, vec_obs], dim=1)
        feat = self.merge_fcnet(feat)

        self.value_inp = feat if self.model_config["vf_share_layers"] else [vis_obs, vec_obs]
        logits = self.action_fc(feat)

        return logits, state

    def value_function(self):
        if not self.model_config["vf_share_layers"]:
            vis_obs, vec_obs = self.value_inp
            vis_feat = self.value_convnet(vis_obs)
            vis_feat = vis_feat.flatten(start_dim=1)
            vis_feat = self.value_fcnet(vis_feat)
            feat = torch.cat([vis_feat, vec_obs], dim=1)
            feat = self.value_merge_fcnet(feat)
            self.value_inp = feat
        return self.value_func(self.value_inp).squeeze(1)

    def _build_convnet(self, model_config, to_nn_seq=True):
        activation = self.model_config.get("conv_activation")
        activation = get_activation_fn(activation, "torch")
        filters = self.model_config["conv_filters"]
        modules = nn.ModuleList()
        for filt in filters:
            modules.append(nn.Conv2d(*filt, padding=0))
            modules.append(activation())
        if to_nn_seq:
            modules = nn.Sequential(*modules)
        return modules

    def _build_fcnet(self, model_config, to_nn_seq=True, filters=None):
        activation = self.model_config.get("fcnet_activation")
        activation = get_activation_fn(activation, "torch")
        if filters is None:
            filters = self.model_config["fcnet_hiddens"]
        modules = nn.ModuleList()
        for i in range(len(filters)-1):
            modules.append(nn.Linear(filters[i], filters[i+1]))
            modules.append(activation())
        if to_nn_seq:
            modules = nn.Sequential(*modules)
        return modules