import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.framework import get_activation_fn
from ray.rllib.utils.framework import try_import_torch

import kerasncp as kncp
from kerasncp.torch import LTCCell

torch, nn = try_import_torch()


class NCP(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, 
                 value_fcnet_hiddens,
                 value_fcnet_activation,
                 inter_neurons, # Number of inter neurons
                 command_neurons, # Number of command neurons
                 sensory_fanout, # How many outgoing synapses has each sensory neuron
                 inter_fanout, # How many outgoing synapses has each inter neuron
                 recurrent_command_synapses, # Now many recurrent synapses are in the command neuron layer
                 motor_fanin, # How many incoming syanpses has each motor neuron
                 **kwargs):
        nn.Module.__init__(self)
        super(NCP, self).__init__(obs_space, action_space, num_outputs,
                                  model_config, name)
        assert model_config['vf_share_layers'] == False

        extracter_hidden = [obs_space.shape[0]] + model_config['fcnet_hiddens']
        extracter_activation = model_config['fcnet_activation']
        self.extracter = self._build_fcnet(extracter_hidden, extracter_activation)
        # TODO: support CNN extractor

        self.wiring = kncp.wirings.NCP(
            inter_neurons=inter_neurons,
            command_neurons=command_neurons,
            motor_neurons=self.num_outputs,
            sensory_fanout=sensory_fanout,
            inter_fanout=inter_fanout,
            recurrent_command_synapses=recurrent_command_synapses,
            motor_fanin=motor_fanin,
        )
        self.cell_size = num_outputs + inter_neurons + command_neurons
        self.ltc_cell = LTCCell(self.wiring, extracter_hidden[-1])

        assert value_fcnet_hiddens[-1] == 1
        if not model_config['vf_share_layers']:
            value_fcnet_hiddens = [obs_space.shape[0]] + value_fcnet_hiddens
            self.value_func = self._build_fcnet(value_fcnet_hiddens, value_fcnet_activation, no_last_act=True)

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        bs, ts, c = inputs.shape
        obs = inputs.reshape(-1, c)
        feats = self.extracter(obs).reshape(bs, ts, -1)
        outs, new_states = [], []
        for t in range(ts):
            res = self.ltc_cell(feats[:,t,:], state[0])
            outs.append(res[0])
            new_states.append(res[1])
        outs = torch.stack(outs, dim=1)
        new_states = torch.stack(new_states, dim=0).squeeze(0)
        self.value_inp = obs # TODO: probably can do computation here

        return outs, [new_states]

    @override(ModelV2)
    def value_function(self):
        return torch.reshape(self.value_func(self.value_inp), [-1])

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros((self.cell_size), np.float32),
        ]

    def _build_convnet(self, filters, activation, to_nn_seq=True):
        activation = get_activation_fn(activation, 'torch')
        modules = nn.ModuleList()
        for filt in filters:
            modules.append(nn.Conv2d(*filt, padding=0))
            modules.append(activation())
        if to_nn_seq:
            modules = nn.Sequential(*modules)
        return modules

    def _build_fcnet(self, filters, activation, to_nn_seq=True, no_last_act=False):
        activation = get_activation_fn(activation, 'torch')
        modules = nn.ModuleList()
        for i in range(len(filters)-1):
            modules.append(nn.Linear(filters[i], filters[i+1]))
            if not no_last_act:
                modules.append(activation())
        if to_nn_seq:
            modules = nn.Sequential(*modules)
        return modules
