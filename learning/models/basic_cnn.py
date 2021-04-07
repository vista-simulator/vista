import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import get_activation_fn


class BasicCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.convnet = self._build_convnet(model_config)
        self.fcnet = self._build_fcnet(model_config)
        self.action_fc = nn.Linear(model_config["fcnet_hiddens"][-1], num_outputs)
        if model_config["vf_share_layers"]:
            self.value_func = nn.Linear(model_config["fcnet_hiddens"][-1], 1)
        else:
            self.value_func = self._build_convnet(model_config, to_nn_seq=False) + \
                              self._build_fcnet(model_config, to_nn_seq=False) + \
                              [nn.Linear(model_config["fcnet_hiddens"][-1], 1)]
            self.value_func = nn.Sequential(*self.value_func)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float().permute(0, 3, 1, 2)
        feat = self.convnet(obs)
        feat = feat.flatten(start_dim=1)
        feat = self.fcnet(feat)
        self.value_inp = feat if self.model_config["vf_share_layers"] else obs
        logits = self.action_fc(feat)
        return logits, state

    def value_function(self):
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

    def _build_fcnet(self, model_config, to_nn_seq=True):
        activation = self.model_config.get("fcnet_activation")
        activation = get_activation_fn(activation, "torch")
        filters = self.model_config["fcnet_hiddens"]
        modules = nn.ModuleList()
        for i in range(len(filters)-1):
            modules.append(nn.Linear(filters[i], filters[i+1]))
            modules.append(activation())
        if to_nn_seq:
            modules = nn.Sequential(*modules)
        return modules