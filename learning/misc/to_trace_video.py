import os
import sys
from pprint import pprint
import rosbag
import numpy as np
import utm
from collections import deque
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from matplotlib.animation import FuncAnimation
from shapely.geometry import box as Box
from shapely import affinity
from shapely.geometry import LineString
from descartes import PolygonPatch
import pandas as pd
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation


ROAD_HALF_WIDTH = 3.5
LEXUS_SIZE = (5, 1.9) #(5, 1.9)
BLUE_PRIUS_SIZE = (4.5, 1.8) #(4.5, 1.8)
MAX_TRAJ_LENGTH = 500
LEXUS_TRAJ_WIDTH = 30
BLUE_PRIUS_TRAJ_WIDTH = 20
USE_SAFE_INTERP = True
SKIP_FRAME = 10
USE_HEADER_TIMESTAMP = True

class MplColorHelper:
  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)[:3]


class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=height + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def read_rosbag(bag_path):
    bag = rosbag.Bag(bag_path)
    topic_info = bag.get_type_and_topic_info()[1]
    data = {k: [] for k in topic_info.keys()}
    for topic, msg, t in bag.read_messages():
        data[topic].append([t.to_sec(), msg])
    
    return data


def fetch_gps(data, topic):
    gps= []
    dt = []
    for t, msg in data[topic]:
        x, y, _, _ = utm.from_latlon(msg.latitude, msg.longitude)
        dt.append(np.abs(t - msg.header.stamp.to_sec()))
        if USE_HEADER_TIMESTAMP:
            t = msg.header.stamp.to_sec()
        gps.append([t, x, y])
    gps = np.array(gps)
    return gps


def fetch_intervention(data, topic='/lexus/ssc/module_states'):
    intervention = []
    for t, msg in data[topic]:
        if msg.info == 'Operator Override':
            intervention.append(t)
    return intervention


def fetch_yaw(data, topic='/lexus/oxts/imu/data'):
    yaws = []
    for t, msg in data[topic]:
        q = msg.orientation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        yaw = rot.as_euler('zyx')[0]
        if USE_HEADER_TIMESTAMP:
            t = msg.header.stamp.to_sec()
        yaws.append([t, yaw])
    yaws = np.array(yaws)
    return yaws


def check_turn_signal(data, check_values=[0, 2], topic='/lexus/pacmod/parsed_tx/turn_rpt'):
    signal = []
    for t, msg in data[topic]:
        if msg.manual_input in check_values:
            signal.append(t)
    has_turn_signal = len(signal) > 0
    return has_turn_signal


def get_poly(pose, car_dim):
    x, y, theta = pose
    car_length, car_width = car_dim
    poly = Box(x-car_width/2., y-car_length/2., x+car_width/2., y+car_length/2.)
    poly = affinity.rotate(poly, np.degrees(theta))
    return poly


def compute_relative_transform(poses, ref_pose):
    theta = -ref_pose[2]
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    xy_in_ref = np.matmul(poses[:,:2] - ref_pose[:2][None,:], R.T)
    theta_in_ref = poses[:,2] - ref_pose[2]
    poses_in_ref = np.concatenate([xy_in_ref, theta_in_ref[:,None]], axis=1)
    return poses_in_ref


def my_interp1d(ts, xs):
    def func(t):
        if t < ts.min() or t > ts.max():
            raise ValueError('Exceed bound')
        closest_two_idcs = np.argsort(np.abs(t - ts))[:2]
        idx1 = np.max(closest_two_idcs)
        idx0 = np.min(closest_two_idcs)
        x = (xs[idx1] - xs[idx0]) * (t - ts[idx0]) / (ts[idx1] - ts[idx0]) + xs[idx0]
        return x
    return func


def subsample_by_time(data, dt=0.5):
    new_data = []
    for i in range(data.shape[0]):
        if len(new_data) != 0 and (data[i,0] - new_data[-1][0] < dt):
            continue
        new_data.append(data[i])
    new_data = np.array(new_data)
    return new_data


def generate_video(data_lexus, data_blue_prius, ref_trace_dir, out_path, add_legend=True, verbose=False, annotate_extrapolate=True):
    # get gps for both vehicle
    gps_blue_prius = fetch_gps(data_blue_prius, '/blue_prius/oxts/gps/fix')
    gps_blue_prius_fx = interp1d(gps_blue_prius[:,0], gps_blue_prius[:,1])
    gps_blue_prius_fy = interp1d(gps_blue_prius[:,0], gps_blue_prius[:,2])

    gps_lexus = fetch_gps(data_lexus, '/lexus/oxts/gps/fix')
    gps_lexus_fx = interp1d(gps_lexus[:,0], gps_lexus[:,1])
    gps_lexus_fy = interp1d(gps_lexus[:,0], gps_lexus[:,2])

    yaw_lexus = fetch_yaw(data_lexus, '/lexus/oxts/imu/data')
    yaw_blue_prius = fetch_yaw(data_blue_prius, '/blue_prius/oxts/imu/data')
    yaw_lexus_f = interp1d(yaw_lexus[:,0], yaw_lexus[:,1], fill_value='extrapolate')
    yaw_blue_prius_f = interp1d(yaw_blue_prius[:,0], yaw_blue_prius[:,1])

    yaw_lexus_sync = np.array([yaw_lexus_f(ts) for ts in gps_lexus[:,0]])
    pose_lexus = np.concatenate([gps_lexus[:,1:], yaw_lexus_sync[:,None]], axis=1)
    if USE_SAFE_INTERP:
        yaw_blue_prius_sync = []
        xy_blue_prius_sync = []
        interp_extrapolate = []
        interp_yaw_dt = []
        interp_xy_dt = []
        for i, ts in enumerate(gps_lexus[:,0]):
            extrapolate = False
            interp_yaw_dt.append(np.abs(yaw_blue_prius[:,0] - ts).min())
            if ts < yaw_blue_prius[0,0] or ts > yaw_blue_prius[-1,0]:
                closest_ts_idx = np.argmin(np.abs(yaw_blue_prius[:,0] - ts))
                yaw = yaw_blue_prius[closest_ts_idx,1]
                extrapolate = True
            else:
                yaw = yaw_blue_prius_f(ts)
            
            interp_xy_dt.append(np.abs(gps_blue_prius[:,0] - ts).min())
            if ts < gps_blue_prius[0,0] or ts > gps_blue_prius[-1,0]:
                closest_ts_idx = np.argmin(np.abs(gps_blue_prius[:,0] - ts))
                xy = [gps_blue_prius[closest_ts_idx,1], gps_blue_prius[closest_ts_idx,2]]
                extrapolate = True
            else:
                xy = np.array([gps_blue_prius_fx(ts), gps_blue_prius_fy(ts)])
            yaw_blue_prius_sync.append(yaw)
            xy_blue_prius_sync.append(xy)
            interp_extrapolate.append(extrapolate)
        yaw_blue_prius_sync = np.array(yaw_blue_prius_sync)
        xy_blue_prius_sync = np.array(xy_blue_prius_sync)
        interp_extrapolate = np.array(interp_extrapolate)
        interp_yaw_dt = np.array(interp_yaw_dt)
        interp_xy_dt = np.array(interp_xy_dt)
    else:
        yaw_blue_prius_sync = np.array([yaw_blue_prius_f(ts) for ts in gps_lexus[:,0]])
        xy_blue_prius_sync = np.array([[gps_blue_prius_fx(ts), gps_blue_prius_fy(ts)] for ts in gps_lexus[:,0]])
        interp_extrapolate = np.ones((xy_blue_prius_sync.shape[0],), dtype=np.bool)
    pose_blue_prius = np.concatenate([xy_blue_prius_sync, yaw_blue_prius_sync[:,None]], axis=1)

    if False:
        ###DEBUG
        fig, axes = plt.subplots(2, 1)
        axes[0].scatter(yaw_lexus[:,0], yaw_lexus[:,1])
        axes[1].scatter(yaw_blue_prius[:,0], yaw_blue_prius[:,1])
        axes[1].scatter(gps_lexus[:,0], yaw_blue_prius_sync, s=2)
        axes[1].set_xlim(*axes[0].get_xlim())
        fig.savefig('test.png')
        import pdb; pdb.set_trace()
        ###DEBUG

    # get intervention
    intervention = fetch_intervention(data_lexus)
    intervention_x = np.array([gps_lexus_fx(t) for t in intervention])
    intervention_y = np.array([gps_lexus_fy(t) for t in intervention])

    # get road
    assert len(sys.argv) >= 3, 'Need to specify reference trace path if ADD_ROAD is True'
    ref_trace_dir = os.path.expanduser(ref_trace_dir)
    gps_ref = np.array(pd.read_csv(os.path.join(ref_trace_dir, 'gps.csv')))
    gps_ref_time = gps_ref[:,0]
    gps_ref = gps_ref[:,1:3]
    gps_ref = np.array([utm.from_latlon(*v)[:2] for v in gps_ref])
    gps_heading_ref = np.array(pd.read_csv(os.path.join(ref_trace_dir, 'gps_heading.csv')))
    gps_heading_ref_f = interp1d(gps_heading_ref[:,0], gps_heading_ref[:,1], fill_value='extrapolate')

    gps_heading_ref_sync = np.array([gps_heading_ref_f(ts) for ts in gps_ref_time])
    pose_road = np.concatenate([gps_ref, gps_heading_ref_sync[:,None]], axis=1)

    # set color
    color_vlim = [0., MAX_TRAJ_LENGTH*1.6 + 1]
    cm_lexus = MplColorHelper('Greens', *color_vlim)
    cm_blue_prius = MplColorHelper('Oranges', *color_vlim)
    color_lexus = [cm_lexus.get_rgb(v) for v in np.arange(MAX_TRAJ_LENGTH + 1)]
    color_blue_prius = [cm_blue_prius.get_rgb(v) for v in np.arange(MAX_TRAJ_LENGTH + 1)]

    # plot
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.expanduser(out_path)
    writer = cv2.VideoWriter(out_path, fourcc, int(100/SKIP_FRAME), (600, 600))

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    xlim = [-20, 20]
    ylim = [-20, 20]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    artists = dict()
    if add_legend:
        ax.legend(handles=[mpatches.Patch(color=color_lexus[-1], label='Ego Car'),
                        mpatches.Patch(color=color_blue_prius[-1], label='Front Car'),
                        mpatches.Circle((0.5,0.5), radius=0.25, color='firebrick', label='Intervention')][:2],
                handler_map={mpatches.Circle: HandlerEllipse()}, 
                fontsize=20, loc='lower left')
    if annotate_extrapolate:
        artists['text_extrapolate'] = fig.text(0.05, 0.9, 'Extrapolate!!', color='red', fontsize=10, fontweight='bold')
        artists['text_xy_dt'] = fig.text(0.05, 0.87, 'Interp xy dt: 0.00', color='red', fontsize=10, fontweight='bold')
        artists['text_yaw_dt'] = fig.text(0.05, 0.84, 'Interp yaw dt: 0.00', color='red', fontsize=10, fontweight='bold')
    for i in tqdm(range(pose_lexus.shape[0])):
        try:
            if (i % SKIP_FRAME) != 0:
                continue

            if False: # DEBUG
                # ref_pose = np.array([pose_lexus[0,0], pose_lexus[0,1], 0])
                ref_pose = np.array([pose_lexus[i,0], pose_lexus[i,1], 0])
                road_in_ref = pose_road - ref_pose

                start = max(0, i - MAX_TRAJ_LENGTH)
                end = i + 1
                blue_prius_in_ref_traj = pose_blue_prius[start:end] - ref_pose
                lexus_in_ref_traj = pose_lexus[start:end] - ref_pose
                blue_prius_in_ref = blue_prius_in_ref_traj[-1]
                lexus_in_ref = lexus_in_ref_traj[-1]
            else: 
                ref_pose = pose_lexus[i]
                road_in_ref = compute_relative_transform(pose_road, ref_pose)

                start = max(0, i - MAX_TRAJ_LENGTH)
                end = i + 1
                blue_prius_in_ref_traj = compute_relative_transform(pose_blue_prius[start:end], ref_pose)
                lexus_in_ref_traj = compute_relative_transform(pose_lexus[start:end], ref_pose)
                blue_prius_in_ref = blue_prius_in_ref_traj[-1]
                lexus_in_ref = lexus_in_ref_traj[-1]

            if verbose:
                print(i, pose_blue_prius[i], blue_prius_in_ref, ref_pose)

            if 'patch_road' in artists.keys():
                artists['patch_road'].remove()
            road_roi = np.logical_and(np.logical_and(road_in_ref[:,0] >= xlim[0], road_in_ref[:,0] <= xlim[1]),
                                    np.logical_and(road_in_ref[:,1] >= ylim[0], road_in_ref[:,1] <= ylim[1]))
            road_in_ref_roi = road_in_ref[road_roi]
            if road_in_ref_roi.shape[0] > 1:
                patch_road = LineString(road_in_ref_roi).buffer(ROAD_HALF_WIDTH)
                patch_road = PolygonPatch(patch_road, fc='none', ec='black', zorder=1)
            ax.add_patch(patch_road)
            artists['patch_road'] = patch_road

            if 'patch_lexus' in artists.keys():
                artists['patch_lexus'].remove()
            poly_lexus = get_poly(lexus_in_ref, LEXUS_SIZE)
            patch_lexus = PolygonPatch(poly_lexus, fc=color_lexus[-1], ec=color_lexus[-1], zorder=2)
            ax.add_patch(patch_lexus)
            artists['patch_lexus'] = patch_lexus

            if 'patch_blue_prius' in artists.keys():
                artists['patch_blue_prius'].remove()
            poly_blue_prius = get_poly(blue_prius_in_ref, BLUE_PRIUS_SIZE)
            patch_blue_prius = PolygonPatch(poly_blue_prius, fc=color_blue_prius[-1], ec=color_blue_prius[-1], zorder=2)
            ax.add_patch(patch_blue_prius)
            artists['patch_blue_prius'] = patch_blue_prius

            if MAX_TRAJ_LENGTH > 0:
                if 'traj_lexus' in artists.keys():
                    artists['traj_lexus'].remove()
                color = color_lexus[-lexus_in_ref_traj.shape[0]:]
                artists['traj_lexus'] = ax.scatter(lexus_in_ref_traj[:,0], 
                    lexus_in_ref_traj[:,1], c=color, s=LEXUS_TRAJ_WIDTH, marker='s', zorder=2)

                if 'traj_blue_prius' in artists.keys():
                    artists['traj_blue_prius'].remove()
                color = color_blue_prius[-blue_prius_in_ref_traj.shape[0]:]
                artists['traj_blue_prius'] = ax.scatter(blue_prius_in_ref_traj[:,0], 
                    blue_prius_in_ref_traj[:,1], c=color, s=BLUE_PRIUS_TRAJ_WIDTH, marker='s')

            if annotate_extrapolate:
                artists['text_extrapolate'].set_visible(interp_extrapolate[i])
                artists['text_xy_dt'].set_text('Interp xy dt: {:.2f}'.format(interp_xy_dt[i]))
                artists['text_yaw_dt'].set_text('Interp yaw dt: {:.2f}'.format(interp_yaw_dt[i]))

            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)[:,:,:3]
            img = img[:,:,::-1] # to bgr
            writer.write(img)
        except KeyboardInterrupt:
            break

    writer.release()


def main():
    # parse arguments
    bag_path_lexus = sys.argv[1]
    ref_trace_dir = sys.argv[2]
    out_path = sys.argv[3]

    # read rosbag
    bag_path_lexus = os.path.expanduser(bag_path_lexus)
    data_lexus = read_rosbag(bag_path_lexus)
    data_blue_prius = data_lexus

    # generate video
    generate_video(data_lexus, data_blue_prius, ref_trace_dir, out_path, False, False, True)


if __name__ == '__main__':
    main()
