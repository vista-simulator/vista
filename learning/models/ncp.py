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


class NCP(Base):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, 
                 inter_neurons, # Number of inter neurons
                 command_neurons, # Number of command neurons
                 sensory_fanout, # How many outgoing synapses has each sensory neuron
                 inter_fanout, # How many outgoing synapses has each inter neuron
                 recurrent_command_synapses, # Now many recurrent synapses are in the command neuron layer
                 motor_fanin, # How many incoming syanpses has each motor neuron
                 value_fcnet_hiddens,
                 value_fcnet_activation,
                 vec_obs_dim=0,
                 vec_branch_hiddens=None,
                 vec_branch_activation=None,
                 use_cnn=False,
                 use_recurrent=True,
                 with_bn=False,
                 **kwargs):
        assert use_recurrent, 'NCP is a recurrent model'
        nn.Module.__init__(self)
        super(NCP, self).__init__(obs_space, action_space, num_outputs,
                                  model_config, name, value_fcnet_hiddens,
                                  value_fcnet_activation, vec_obs_dim, vec_branch_hiddens, 
                                  vec_branch_activation, use_cnn, use_recurrent, with_bn, **kwargs)

        # define policy
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
        self.ltc_cell = LTCCell(self.wiring, self.feat_channel)

    def policy_inference(self, inputs, state, seq_lens):
        bs, ts, c = inputs.shape
        outs, new_states = [], []
        for t in range(ts):
            res = self.ltc_cell(inputs[:,t,:], state[0])
            outs.append(res[0])
            new_states.append(res[1])
        outs = torch.stack(outs, dim=1)
        new_states = torch.stack(new_states, dim=0).squeeze(0)

        return outs, [new_states]

    @override(ModelV2)
    def get_initial_state(self):
        device = next(self.parameters()).device
        return [
            torch.zeros((self.cell_size), dtype=torch.float32).to(device)
        ]
