import math
import torch
import torch.nn as nn
import torchvision.models as models
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import get_activation_fn


class ResNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, resnet_layer=18):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        # get resnet as feature extracter. NOTE: need to download checkpoint before being in sc computation node
        assert resnet_layer in [18, 34, 50]
        resnet = getattr(models, 'resnet{}'.format(resnet_layer))(pretrained=True)
        in_channels = obs_space.shape[-1]
        if in_channels == 3:
            conv1 = resnet.conv1
            bn1 = resnet.bn1
        else:
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bn1 = nn.BatchNorm2d(64)
            conv1.apply(weights_init)
            bn1.apply(weights_init)
        conv_final = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        conv_final.apply(weights_init)
        self.convnet = nn.Sequential(conv1, bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, conv_final)
        del resnet

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
        self.convnet.train(mode=input_dict.get("is_training", False)) # NOTE: important for bn
        obs = input_dict["obs"].float().permute(0, 3, 1, 2)
        feat = self.convnet(obs)
        feat = feat.flatten(start_dim=1)
        feat = self.fcnet(feat)
        self.value_inp = feat if self.model_config["vf_share_layers"] else obs
        logits = self.action_fc(feat)
        return logits, state

    def value_function(self):
        return self.value_func(self.value_inp).squeeze(1)

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


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
