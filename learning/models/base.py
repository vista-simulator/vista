import os
import sys
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
                 value_fcnet_dropout=0.,
                 vec_obs_dim=0,
                 vec_branch_hiddens=None,
                 vec_branch_activation=None,
                 use_cnn=False,
                 use_recurrent=False,
                 use_seg_pretrained=False,
                 seg_model_name=None,
                 seg_out_channel=16,
                 feat_roi_crop=None,
                 with_bn=False,
                 obs_idcs=[0, 1],
                 **kwargs):
        nn.Module.__init__(self)
        super(Base, self).__init__(obs_space, action_space, num_outputs,
                                   model_config, name)
        self.use_recurrent = use_recurrent
        self.feat_roi_crop = feat_roi_crop
        self.obs_idcs = obs_idcs
        
        # define feature extractor
        self.use_cnn = use_cnn
        self.use_seg_pretrained = use_seg_pretrained
        if self.use_cnn:
            if self.use_seg_pretrained:
                sys.path.insert(0, os.environ.get('SEG_PRETRAINED_ROOT'))
                try:
                    from mit_semseg.models import ModelBuilder
                    from mit_semseg.config import cfg

                    config_arg = os.path.join(
                        os.environ.get('SEG_PRETRAINED_ROOT'),
                        'config/{}.yaml'.format(seg_model_name))
                    cfg.merge_from_file(config_arg)
                    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
                    cfg.MODEL.weights_encoder = os.path.join(
                        os.environ.get('SEG_PRETRAINED_ROOT'),
                        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
                    assert os.path.exists(cfg.MODEL.weights_encoder), 'checkpoint does not exist!'

                    self.extractor = ModelBuilder.build_encoder(
                        arch=cfg.MODEL.arch_encoder,
                        fc_dim=cfg.MODEL.fc_dim,
                        weights=cfg.MODEL.weights_encoder)
                    for param in self.extractor.parameters():
                        param.requires_grad = False
                    fake_inp = torch.zeros([1] + list(obs_space.shape)).permute(0, 3, 1, 2)
                    fake_out = self.extractor(fake_inp)
                    assert len(fake_out) == 1
                    raw_feat_channel = fake_out[0].shape[1]

                    if self.feat_roi_crop:
                        ori_h, ori_w = fake_inp.shape[2:]
                        feat_h, feat_w = fake_out[0].shape[2:]
                        crop_i1 = int(np.floor(self.feat_roi_crop[0] / ori_h * feat_h))
                        crop_i2 = int(np.ceil(self.feat_roi_crop[2] / ori_h * feat_h))
                        crop_j1 = int(np.floor(self.feat_roi_crop[1] / ori_w * feat_w))
                        crop_j2 = int(np.ceil(self.feat_roi_crop[3] / ori_w * feat_w))
                        self.feat_roi_crop = (crop_i1, crop_j1, crop_i2, crop_j2)
                        fake_out[0] = fake_out[0][:, :, crop_i1:crop_i2, crop_j1:crop_j2]

                    if seg_out_channel is None: # set to default
                        seg_out_channel = raw_feat_channel // 64
                    self.extractor_post = nn.Sequential(
                        nn.Conv2d(raw_feat_channel, seg_out_channel, 1, 1, 0),
                        nn.Dropout2d(p=0.5),
                        nn.Flatten())
                    feat_channel = self.extractor_post(fake_out[0]).shape[1]
                except ImportError:
                    raise ImportError('Fail to import segmentation repo. Do you forget to set SEG_PRETRAINED_ROOT ?')
            else:
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
            assert False, 'Cannot set dropout here'
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
                assert False, 'Cannot set dropout here'
                self.vf_extractor = self._build_fcnet(extractor_filters, 
                    extractor_activation, with_bn=with_bn)
        assert value_fcnet_hiddens[-1] == 1, 'Last channel should be 1 in value function'
        self.vf_fcs = self._build_fcnet(value_fcnet_hiddens, value_fcnet_activation, dropout=value_fcnet_dropout,
            with_bn=with_bn, no_last_act=True) # NOTE: last layer is linear; not using NCP for value function

    def policy_inference(self):
        raise NotImplementedError

    @override(ModelV2)
    def get_initial_state(self):
        device = next(self.parameters()).device
        return [
            torch.zeros((1), dtype=torch.float32).to(device)
        ]

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # set train/eval mode for BN
        self.train(mode=input_dict.get('is_training', False))

        # feature extraction
        if self.use_cnn:
            if isinstance(input_dict['obs'], list):
                img_obs = input_dict['obs'][self.obs_idcs[0]].permute(0, 3, 1, 2).float()
                vec_obs = input_dict['obs'][self.obs_idcs[1]]
                if self.use_seg_pretrained:
                    self.extractor.eval() # NOTE: fix pretrained network weights
                    with torch.no_grad():
                        feat = self.extractor(img_obs)[0]
                    if self.feat_roi_crop:
                        crop_i1, crop_j1, crop_i2, crop_j2 = self.feat_roi_crop
                        feat = feat[:, :, crop_i1:crop_i2, crop_j1:crop_j2]
                    feat = self.extractor_post(feat)
                else:
                    feat = self.extractor(img_obs)
                if hasattr(self, 'vec_extractor'):
                    vec_feat = self.vec_extractor(vec_obs)
                else:
                    vec_feat = vec_obs
                feat = torch.cat([feat, vec_feat], -1)
                obs = [img_obs, vec_obs]
            else:
                obs = input_dict['obs'].permute(0, 3, 1, 2).float()
                if self.use_seg_pretrained:
                    self.extractor.eval() # NOTE: fix pretrained network weights
                    with torch.no_grad():
                        feat = self.extractor(obs)[0]
                    if self.feat_roi_crop:
                        crop_i1, crop_j1, crop_i2, crop_j2 = self.feat_roi_crop
                        feat = feat[:, :, crop_i1:crop_i2, crop_j1:crop_j2]
                    feat = self.extractor_post(feat)
                else:
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
            output, new_state = self.policy_inference(feat, state, seq_lens)

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
