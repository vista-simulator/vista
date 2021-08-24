"""SuperSlowMo
Code is edited from https://github.com/avinashpaliwal/Super-SloMo

"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import model


class SlowMoWarp:
    """SlowMoWarp
    Wrapper around Slowmo Code
    Allows to interpolate based on-demand

    Args:
        height: image height
        width: image width
        checkpoint: network checkpoint
        lambda flow: modules max flow to produce N images
        cuda: use cuda
    """
    def __init__(self, height, width, checkpoint, lambda_flow=0.5, cuda=True):

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and cuda else "cpu"
        )
        self.flowComp = model.UNet(6, 4)
        self.flowComp.to(self.device)
        self.flowComp.eval()
        self.ArbTimeFlowIntrp = model.UNet(20, 5)
        self.ArbTimeFlowIntrp.to(self.device)
        self.ArbTimeFlowIntrp.eval()

        self.flowBackWarp = model.backWarp(width, height, self.device)
        self.flowBackWarp = self.flowBackWarp.to(self.device)
        self.flowBackWarp.eval()

        dict1 = torch.load(checkpoint, map_location="cpu")
        self.ArbTimeFlowIntrp.load_state_dict(dict1["state_dictAT"])
        self.flowComp.load_state_dict(dict1["state_dictFC"])

        mean = [0.429, 0.431, 0.397]
        # mean = [1,1,1]
        std = [1, 1, 1]
        self.normalize = transforms.Normalize(mean=mean, std=std)

        self.transform = transforms.Compose([transforms.ToTensor(), self.normalize])

        negmean = [x * -1 for x in mean]
        self.revNormalize = transforms.Normalize(mean=negmean, std=std)

        move_back_channel = lambda x: x.permute(1, 2, 0).contiguous()
        to_numpy = lambda x: (x * 255).cpu().data.numpy().astype(np.uint8)
        self.rev_transform = transforms.Compose(
            [self.revNormalize, move_back_channel, to_numpy]
        )
        self.lambda_flow = lambda_flow

    def interpolate_time(self, I0, I1, F_0_1, F_1_0, t):
        temp = -t * (1 - t)
        fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

        g_I0_F_t_0 = self.flowBackWarp(I0, F_t_0)
        g_I1_F_t_1 = self.flowBackWarp(I1, F_t_1)

        intrpOut = self.ArbTimeFlowIntrp(
            torch.cat(
                (I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1
            )
        )

        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        V_t_0 = torch.sigmoid(
            intrpOut[:, 4:5, :, :]
        )  # this could be critical for the simulator!!!
        V_t_1 = 1 - V_t_0

        g_I0_F_t_0_f = self.flowBackWarp(I0, F_t_0_f)
        g_I1_F_t_1_f = self.flowBackWarp(I1, F_t_1_f)

        wCoeff = [1 - t, t]

        Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
            wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1
        )

        return Ft_p

    def interpolate_sync_batch(self, I0, I1, F_0_1, F_1_0, sf=2):
        if sf > 1:
            ts = [float(_i) / sf for _i in range(1, sf)]

            I0_b = I0.repeat(sf-1, 1, 1, 1)
            I1_b = I1.repeat(sf-1, 1, 1, 1)
            F_0_1_b = F_0_1.repeat(sf-1, 1, 1, 1)
            F_1_0_b = F_1_0.repeat(sf-1, 1, 1, 1)

            fCoeff_fn = lambda _t: [-_t * (1 - _t), _t * _t, (1 - _t) * (1 - _t), -_t * (1 - _t)]
            fCoeffs = torch.Tensor([fCoeff_fn(t) for t in ts])[...,None,None].to(self.device)

            wCoeff_fn = lambda _t: [1 - _t, _t]
            wCoeffs = torch.Tensor([wCoeff_fn(t) for t in ts])[...,None,None].to(self.device)

            F_t_0_b = fCoeffs[:,0:1] * F_0_1_b + fCoeffs[:,1:2] * F_1_0_b
            F_t_1_b = fCoeffs[:,2:3] * F_0_1_b + fCoeffs[:,3:4] * F_1_0_b

            g_I0_F_t_0_b = self.flowBackWarp(I0_b, F_t_0_b)
            g_I1_F_t_1_b = self.flowBackWarp(I1_b, F_t_1_b)

            intrpOut_b = self.ArbTimeFlowIntrp(
                torch.cat(
                    (I0_b, I1_b, F_0_1_b, F_1_0_b, F_t_1_b, F_t_0_b, g_I1_F_t_1_b, g_I0_F_t_0_b), dim=1
                )
            )

            F_t_0_f_b = intrpOut_b[:, :2, :, :] + F_t_0_b
            F_t_1_f_b = intrpOut_b[:, 2:4, :, :] + F_t_1_b
            V_t_0_b = torch.sigmoid(
                intrpOut_b[:, 4:5, :, :]
            )  # this could be critical for the simulator!!!
            V_t_1_b = 1 - V_t_0_b

            g_I0_F_t_0_f_b = self.flowBackWarp(I0_b, F_t_0_f_b)
            g_I1_F_t_1_f_b = self.flowBackWarp(I1_b, F_t_1_f_b)

            Ft_p_b = (wCoeffs[:,0:1] * V_t_0_b * g_I0_F_t_0_f_b + \
                    wCoeffs[:,1:2] * V_t_1_b * g_I1_F_t_1_f_b) / (
                    wCoeffs[:,0:1] * V_t_0_b + wCoeffs[:,1:2] * V_t_1_b)

            interpolated = [self.rev_transform(Ft_p) for Ft_p in Ft_p_b]
        else:
            interpolated = []

        return interpolated

    def interpolate_sync(self, I0, I1, F_0_1, F_1_0, sf=2):
        interpolated = []
        for intermediateIndex in range(1, sf):
            t = float(intermediateIndex) / sf
            Ft_p = self.interpolate_time(I0, I1, F_0_1, F_1_0, t)
            out = self.rev_transform(Ft_p[0])
            interpolated.append(out)
        return interpolated

    def interpolate_max_flow(self, I0, I1, F_0_1, F_1_0, max_sf=10):
        fwd_mag = F_0_1.norm(dim=1)
        bwd_mag = F_1_0.norm(dim=1)

        fwd_mag_max = fwd_mag.max().item()
        bwd_mag_max = bwd_mag.max().item()

        sf = int(round(max(fwd_mag_max, bwd_mag_max) * self.lambda_flow))
        sf = min(sf, max_sf)
        return self.interpolate_sync_batch(I0, I1, F_0_1, F_1_0, sf), sf

    def forward(self, I0, I1):
        if not isinstance(I0, torch.Tensor):
            I0 = self.transform(I0).to(self.device)[None]
        if not isinstance(I1, torch.Tensor):
            I1 = self.transform(I1).to(self.device)[None]

        P01 = torch.cat((I0, I1), dim=1)
        flowOut = self.flowComp(P01)

        return flowOut        

    def forward_warp(self, frame0, frame1, max_sf=-1, use_max_flow=True):
        I0 = self.transform(frame0).to(self.device)[None]
        I1 = self.transform(frame1).to(self.device)[None]

        flowOut = self.forward(I0, I1)
        F_0_1 = flowOut[:, :2, :, :]
        F_1_0 = flowOut[:, 2:, :, :]

        if max_sf == -1:
            max_sf = 10
            use_max_flow = True

        if use_max_flow:
            inter_frame, sf = self.interpolate_max_flow(I0, I1, F_0_1, F_1_0, max_sf)
        else:
            inter_frame = self.interpolate_sync(I0, I1, F_0_1, F_1_0, max_sf)
            sf = max_sf

        return {"flow": (F_0_1, F_1_0), "interpolated": inter_frame, "sf": sf}
