"""
script to upsample your videos
"""
from __future__ import absolute_import

import os
import urllib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from skvideo.io import FFmpegWriter
from PIL import Image
import h5py

from torchvision.utils import make_grid
from tqdm import tqdm

from video_stream import VideoStream
from slowmo_warp import SlowMoWarp
try:
    from utils import grab_videos, draw_arrows
except:
    print('Fail to import grab_videos and draw_arrows')
    pass


def show_slowmo(last_frame, frame, flow_fw, flow_bw, interp, fps):
    """SlowMo visualization
    Args:
        last_frame: prev rgb frame (h,w,3)
        current_frame: current rgb frame (h,w,3)
        flow_fw: flow forward (1,h,w,2)
        flow_bw: flow backward (1,h,w,2)
        interp: last_frame + interpolated frames
        fps: current frame-rate
    """

    def viz_flow(frame, flow):
        flow1 = flow.data.cpu().numpy()
        img_h, img_w = flow1.shape[2:]
        step = 10
        assert img_h % step == 0 and img_w % step == 0, \
            "Image size should be completely divided by `step` for drawing optical flow."
        frame0 = draw_arrows(frame, flow1[0], step=step, flow_unit="pixels")
        return frame0

    color = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = interp[0].shape[:2]
    virtual_fps = fps * len(interp)

    viz_flow_fw = viz_flow(last_frame.copy(), flow_fw)
    viz_flow_bw = viz_flow(frame.copy(), flow_bw)
    for j, item in enumerate(interp):
        img = item.copy()

        img = cv2.putText(img, "orig fps: " + str(fps), (10, height - 90), font, 1.0, color, 2)
        img = cv2.putText(img, "virtual fps: " + str(virtual_fps), (10, height - 60), font, 1.0, color, 2, )
        img = cv2.putText(img, "#" + str(j), (10, height - 30), font, 1.0, color, 2)

        vizu = np.concatenate([viz_flow_fw[None], viz_flow_bw[None], img[None]])

        vizu = torch.from_numpy(vizu).permute(0, 3, 1, 2).contiguous()
        vizu = make_grid(vizu, nrow=2).permute(1, 2, 0).contiguous().numpy()

        cv2.imshow("result", vizu)
        key = cv2.waitKey(5)
        if key == 27:
            return 0
    return 1


def main_video(
        video_filename,
        out_name="",
        video_fps=240,
        height=-1,
        width=-1,
        sf=-1,
        seek_frame=0,
        max_frames=-1,
        lambda_flow=0.5,
        cuda=True,
        viz=False,
        checkpoint='SuperSloMo.ckpt',
        crf=1,
        run_dvs=False,
        extract_flow=False,
):
    """SlowMo Interpolates video
    It produces another .mp4 video + .npy file for timestamps.
    Args:
        video_filename: file path or list of ordered frame images
        out_name: out file path
        video_fps: video frame rate
        height: desired height
        width: desired width
        sf: desired frame-rate scale-factor (if -1 it interpolates based on maximum optical flow)
        seek_frame: seek in video before interpolating
        max_frames: maximum number of frames to interpolate
        lambda_flow: when interpolating with maximum flow, we multiply the maximum flow by this
        factor to compute the actual number of frames.
        cuda: use cuda
        viz: visualize the flow and interpolated frames
        checkpoint: if not provided will download it
        run_evs: run event-based camera simulation
        extract_flow: extract optical flow only
    """
    print("Out Video: ", out_name)

    if isinstance(video_filename, list):
        # video_filename is a list of images
        print("First image of the list: ", video_filename[0])
        im = Image.open(video_filename[0])
        width, height = im.size
        stream = video_filename
    else:
        print("Video filename: ", video_filename)
        stream = VideoStream(
            video_filename,
            height,
            width,
            seek_frame=seek_frame,
            max_frames=max_frames,
            random_start=False,
            rgb=True)
        height, width = stream.height, stream.width

    slomo = SlowMoWarp(height, width, checkpoint, lambda_flow=lambda_flow, cuda=cuda)

    fps = video_fps

    delta_t = 1.0 / fps
    delta_t_us = delta_t * 1e6

    print("fps: ", fps)
    print("Total length: ", len(stream))
    timestamps = []

    last_frame = None
    first_write = True

    num_video = 0

    if out_name:
        if extract_flow:
            flow_as_video = True
            if flow_as_video:
                out_name_no_ext = os.path.splitext(out_name)[0]
                out1_name = out_name_no_ext + '_forward.mp4'
                out2_name = out_name_no_ext + '_backward.mp4'

                flow1_writer = FFmpegWriter(out1_name, outputdict={
                    '-vcodec': 'libx264',  # use the h.264 codec
                    #'-preset': 'veryslow'  # the slower the better compression, in princple, try
                })
                flow2_writer = FFmpegWriter(out2_name, outputdict={
                    '-vcodec': 'libx264',  # use the h.264 codec
                    #'-preset': 'veryslow'  # the slower the better compression, in princple, try
                })

                out_meta_name = out_name_no_ext + '_meta.h5'
                if os.path.exists(out_meta_name):
                    os.remove(out_meta_name)
                flow_meta_writer = h5py.File(out_meta_name, 'a')
            else:
                if os.path.exists(out_name):
                    os.remove(out_name)
                flow_writer = h5py.File(out_name, 'a')
        else:
            video_writer = FFmpegWriter(out_name, outputdict={
                '-vcodec': 'libx264',  # use the h.264 codec
                '-crf': str(crf),  # set the constant rate factor to 0, which is lossless
                #'-preset': 'veryslow'  # the slower the better compression, in princple, try
                # other options see https://trac.ffmpeg.org/wiki/Encode/H.264
            })

            if run_dvs:
                dvs_name_list = list(os.path.splitext(out_name))
                dvs_out_name = dvs_name_list[0] + '_dvs' + dvs_name_list[1]
                dvs_video_writer = FFmpegWriter(dvs_out_name, outputdict={
                    '-vcodec': 'libx264',  # use the h.264 codec
                    '-crf': str(crf),  # set the constant rate factor to 0, which is lossless
                    #'-preset': 'veryslow'  # the slower the better compression, in princple, try
                    # other options see https://trac.ffmpeg.org/wiki/Encode/H.264
                })

    last_ts = 0
    for i, frame in enumerate(tqdm(stream)):
        if isinstance(frame, str):
            frame = cv2.imread(frame)[:, :, ::-1]
            assert frame.shape[2] == 3

        ts = i * delta_t

        if last_frame is not None:

            if extract_flow:
                with torch.no_grad():
                    out = slomo.forward(last_frame, frame)
                flow = out.cpu().numpy()
                if flow_as_video:
                    flow = flow[0].transpose(1, 2, 0)
                    flow1_img, flow1_minmax = flow2img(flow[..., :2])
                    flow2_img, flow2_minmax = flow2img(flow[..., 2:])

                    flow1_writer.writeFrame(flow1_img)
                    flow2_writer.writeFrame(flow2_img)

                    flow1_minmax = np.array(flow1_minmax)[None,:]
                    flow2_minmax = np.array(flow2_minmax)[None,:]
                    if 'forward' not in flow_meta_writer.keys():
                        flow_meta_writer.create_dataset('forward', data=flow1_minmax,
                                                        chunks=True, maxshape=[None, 2])
                        flow_meta_writer.create_dataset('backward', data=flow2_minmax,
                                                        chunks=True, maxshape=[None, 2])
                    else:
                        flow_meta_writer['forward'].resize((flow_meta_writer['forward'].shape[0] + 1), 
                                                           axis=0)
                        flow_meta_writer['forward'][-1:] = flow1_minmax
                        
                        flow_meta_writer['backward'].resize((flow_meta_writer['backward'].shape[0] + 1), 
                                                            axis=0)
                        flow_meta_writer['backward'][-1:] = flow2_minmax
                else:
                    if 'forward' not in flow_writer.keys():
                        flow_writer.create_dataset('forward', data=flow[:,:2], chunks=True, 
                                                   maxshape=[None]+list(flow[:,:2].shape[1:]))
                        flow_writer.create_dataset('backward', data=flow[:,2:], chunks=True, 
                                                   maxshape=[None]+list(flow[:,2:].shape[1:]))
                    else:
                        flow_writer['forward'].resize((flow_writer['forward'].shape[0] + 1), axis=0)
                        flow_writer['forward'][-1:] = flow[:,:2]

                        flow_writer['backward'].resize((flow_writer['backward'].shape[0] + 1), axis=0)
                        flow_writer['backward'][-1:] = flow[:,2:]
            else:
                t_start = last_ts
                t_end = ts

                with torch.no_grad():
                    out = slomo.forward_warp(last_frame, frame, sf=sf)

                interp = [last_frame] + out["interpolated"]
                dt = (t_end - t_start) / len(interp)
                interp_ts = np.linspace(t_start, t_end - dt, len(interp))

                if out["sf"] == 0:
                    print("skipping here, flow too small")
                    continue

                if out_name:
                    for item in interp:
                        video_writer.writeFrame(item)

                if run_dvs:
                    positive_C = 0.3
                    negative_C = 0.3

                    dvs_frame = np.zeros(last_frame.shape[:2])
                    if len(interp) >= 2:
                        last_interp_frame = cv2.cvtColor(interp[0], cv2.COLOR_RGB2GRAY)
                        for interp_frame in interp[1:]:
                            interp_frame = cv2.cvtColor(interp_frame, cv2.COLOR_RGB2GRAY)
                            log_d_interp = np.log(interp_frame / 255. + 1e-8) - \
                                        np.log(last_interp_frame / 255. + 1e-8)
                            positive_events = log_d_interp >= positive_C
                            negative_events = log_d_interp <= -negative_C
                            dvs_frame[positive_events] += 1
                            dvs_frame[negative_events] -= 1
                            dvs_frame = np.clip(dvs_frame, -1, 1)
                            last_interp_frame = interp_frame

                    if out_name:
                        dvs_frame_vis = np.zeros_like(last_frame)
                        dvs_frame_vis[dvs_frame == 1, :] = np.array([255, 255, 255])
                        dvs_frame_vis[dvs_frame == -1, :] = np.array([114, 188, 212])
                        dvs_compare = np.concatenate([frame, dvs_frame_vis], axis=1)
                        dvs_video_writer.writeFrame(dvs_compare)

                timestamps.append(interp_ts)

                if viz:
                    key = show_slowmo(last_frame, frame, *out['flow'], interp, fps)
                    if key == 0:
                        break

                last_ts = ts

        last_frame = frame.copy()

    if viz:
        cv2.destroyWindow("result")

    if out_name:
        if extract_flow:
            if flow_as_video:
                flow_meta_writer.close()
                flow1_writer.close()
                flow2_writer.close()
            else:
                flow_writer.close()
        else:
            video_writer.close()
            timestamps_out = np.concatenate(timestamps)
            np.save(os.path.splitext(out_name)[0] + "_ts.npy", timestamps_out)


def flow2img(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))
    nans = np.isnan(mag)
    if np.any(nans):
        nans = np.where(nans)
        mag[nans] = 0.

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    mag_min, mag_max = mag.min(), mag.max()
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = (mag - mag_min) / (mag_max - mag_min) * 255
    hsv[..., 2] = 255

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return img, (mag_min, mag_max)


def main(
        input_path,
        output_path,
        video_fps=240,
        height=-1,
        width=-1,
        sf=-1,
        seek_frame=0,
        max_frames=-1,
        lambda_flow=0.5,
        cuda=True,
        viz=False,
        checkpoint='SuperSloMo.ckpt',
        crf=1,
        run_dvs=False,
        extract_flow=False):
    """Same Documentation, just with additional input directory"""
    main_fun = lambda x, y: main_video(x, y, video_fps, height, width, sf, seek_frame, max_frames, lambda_flow, cuda, viz,
                                       checkpoint, crf, run_dvs, extract_flow)
    wsf = str(sf) if sf > 0 else "asynchronous"
    print('Interpolation frame_rate factor: ', wsf)
    if os.path.isdir(input_path):
        assert os.path.isdir(output_path)
        filenames = grab_videos(input_path)
        for item in filenames:
            otem, _ = os.path.splitext(os.path.basename(item))
            otem = os.path.join(odir, otem + ext)
            if os.path.exists(otem):
                continue
            main_fun(item, otem)
    else:
        main_fun(input_path, output_path)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
