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

    def interpolate_sync(self, I0, I1, F_0_1, F_1_0, sf=2):
        interpolated = []
        for intermediateIndex in range(1, sf):
            t = float(intermediateIndex) / sf
            Ft_p = self.interpolate_time(I0, I1, F_0_1, F_1_0, t)
            out = self.rev_transform(Ft_p[0])
            interpolated.append(out)
        return interpolated

    def interpolate_max_flow(self, I0, I1, F_0_1, F_1_0, max_sf=-1):
        fwd_mag = F_0_1.norm(dim=1)
        bwd_mag = F_1_0.norm(dim=1)

        fwd_mag_max = fwd_mag.max().item()
        bwd_mag_max = bwd_mag.max().item()

        sf = int(round(max(fwd_mag_max, bwd_mag_max) * self.lambda_flow))
        sf = sf if max_sf == -1 else min(sf, max_sf)
        return self.interpolate_sync(I0, I1, F_0_1, F_1_0, sf), sf

    def forward(self, frame0, frame1):
        I0 = self.transform(frame0).to(self.device)[None]
        I1 = self.transform(frame1).to(self.device)[None]

        P01 = torch.cat((I0, I1), dim=1)
        flowOut = self.flowComp(P01)

        return flowOut        

    def forward_warp(self, frame0, frame1, sf=-1, max_sf=-1):
        I0 = self.transform(frame0).to(self.device)[None]
        I1 = self.transform(frame1).to(self.device)[None]

        flowOut = self.forward(frame0, frame1)
        F_0_1 = flowOut[:, :2, :, :]
        F_1_0 = flowOut[:, 2:, :, :]

        if sf == -1:
            inter_frame, sf = self.interpolate_max_flow(I0, I1, F_0_1, F_1_0, max_sf)
        else:
            inter_frame = self.interpolate_sync(I0, I1, F_0_1, F_1_0, sf)

        return {"flow": (F_0_1, F_1_0), "interpolated": inter_frame, "sf": sf}
