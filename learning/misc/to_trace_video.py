import os
import sys
from pprint import pprint
import rosbag
import numpy as np
import utm
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


ROAD_HALF_WIDTH = 3.5
LEXUS_SIZE = (5, 1.9) #(5, 1.9)
BLUE_PRIUS_SIZE = (4.5, 1.8) #(4.5, 1.8)
MAX_TRAJ_LENGTH = 50
USE_SAFE_INTERP = True

max_mul = 1.05
min_mul = 0.4
marker_size = 4
legend_size = 20


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
    for t, msg in data[topic]:
        x, y, _, _ = utm.from_latlon(msg.latitude, msg.longitude)
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
        yaw = np.arctan2(2.*(q.x*q.y + q.z*q.w) , (q.w**2 - q.z**2 - q.y*2 + q.x**2))
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
    c = np.cos(ref_pose[2])
    s = np.sin(ref_pose[2])
    R = np.array([[c, s], [-s, c]])
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


def filter_gps_by_speed(data, speed=100):
    dt = data[1:,0] - data[:-1,0]
    dist = np.linalg.norm(data[1:,1:] - data[:-1,1:], axis=1)
    v = dist / dt
    # drop_seed = v > speed
    win = 100
    mask = np.ones((data.shape[0],), dtype=np.bool)
    # for seed in np.where(dt < 1e-2)[0]:
    for seed in np.where(v > 5000)[0]:
        mask[seed-2*win:seed+win] = False
    # data[np.logical_not(mask),1] = data[:,1].mean()
    # import pdb; pdb.set_trace()
    # return data
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(dt.shape[0]), dt)
    fig.savefig('test.png')
    import pdb; pdb.set_trace()
    return data[mask]
    # mask = np.logical_and(v <= speed, v>0.01)
    # mask = np.concatenate([[True], mask])
    # v = np.concatenate([[0.], v])
    # dt = np.concatenate([[0.], dt])
    # return v, dt


def subsample_by_time(data, dt=0.5):
    new_data = []
    for i in range(data.shape[0]):
        if len(new_data) != 0 and (data[i,0] - new_data[-1][0] < dt):
            continue
        new_data.append(data[i])
    new_data = np.array(new_data)
    return new_data


# read rosbag
bag_path_lexus = os.path.expanduser(sys.argv[1])
data_lexus = read_rosbag(bag_path_lexus)
data_blue_prius = data_lexus

# get gps for both vehicle
gps_blue_prius = fetch_gps(data_blue_prius, '/blue_prius/oxts/gps/fix')
gps_blue_prius = subsample_by_time(gps_blue_prius)
gps_blue_prius_fx = interp1d(gps_blue_prius[:,0], gps_blue_prius[:,1])
gps_blue_prius_fy = interp1d(gps_blue_prius[:,0], gps_blue_prius[:,2])
###DEBUG
from scipy.interpolate import UnivariateSpline
gps_blue_prius_fx = UnivariateSpline(gps_blue_prius[:,0], gps_blue_prius[:,1], k=3)
gps_blue_prius_fy = UnivariateSpline(gps_blue_prius[:,0], gps_blue_prius[:,2], k=3)
# gps_blue_prius_fx.set_smoothing_factor(3)
# gps_blue_prius_fy.set_smoothing_factor(3)
###DEBUG

gps_lexus = fetch_gps(data_lexus, '/lexus/oxts/gps/fix')
gps_lexus_fx = interp1d(gps_lexus[:,0], gps_lexus[:,1])
gps_lexus_fy = interp1d(gps_lexus[:,0], gps_lexus[:,2])

###DEBUG
fig, ax = plt.subplots(2,1)
ax[0].scatter([ts for ts in gps_lexus[:,0] if ts > gps_blue_prius[0,0] and ts < gps_blue_prius[-1,0]], [gps_blue_prius_fx(ts) for ts in gps_lexus[:,0] if ts > gps_blue_prius[0,0] and ts < gps_blue_prius[-1,0]], c='r')
ax[0].scatter(gps_blue_prius[:,0], gps_blue_prius[:,1])
ax[1].scatter([ts for ts in gps_lexus[:,0] if ts > gps_blue_prius[0,0] and ts < gps_blue_prius[-1,0]], [gps_blue_prius_fy(ts) for ts in gps_lexus[:,0] if ts > gps_blue_prius[0,0] and ts < gps_blue_prius[-1,0]], c='r')
ax[1].scatter(gps_blue_prius[:,0], gps_blue_prius[:,2])
ax[0].scatter(gps_blue_prius[1,0], gps_blue_prius[1,1], c='y')
ax[1].scatter(gps_blue_prius[1,0], gps_blue_prius[1,2], c='y')
# ax[0].scatter(gps_blue_prius[:,0], gps_blue_prius[:,1])
# ax[1].scatter(gps_blue_prius[:,0], v)
# ax[2].scatter(gps_blue_prius[:,0], dt)
fig.savefig('test1.png')
import pdb; pdb.set_trace()
###DEBUG

yaw_lexus = fetch_yaw(data_lexus, '/lexus/oxts/imu/data')
yaw_blue_prius = fetch_yaw(data_blue_prius, '/blue_prius/oxts/imu/data')
yaw_blue_prius = subsample_by_time(yaw_blue_prius)
yaw_lexus_f = interp1d(yaw_lexus[:,0], yaw_lexus[:,1], fill_value='extrapolate')
yaw_blue_prius_f = interp1d(yaw_blue_prius[:,0], yaw_blue_prius[:,1])

yaw_lexus_sync = np.array([yaw_lexus_f(ts) for ts in gps_lexus[:,0]])
pose_lexus = np.concatenate([gps_lexus[:,1:], yaw_lexus_sync[:,None]], axis=1)
if USE_SAFE_INTERP:
    yaw_blue_prius_sync = []
    xy_blue_prius_sync = []
    for i, ts in enumerate(gps_lexus[:,0]):
        if ts < yaw_blue_prius[0,0] or ts > yaw_blue_prius[-1,0]:
            closest_ts_idx = np.argmin(np.abs(yaw_blue_prius[:,0] - ts))
            yaw = yaw_blue_prius[closest_ts_idx,1]
        else:
            yaw = yaw_blue_prius_f(ts)
        if ts < gps_blue_prius[0,0] or ts > gps_blue_prius[-1,0]:
            closest_ts_idx = np.argmin(np.abs(gps_blue_prius[:,0] - ts))
            xy = [gps_blue_prius[closest_ts_idx,1], gps_blue_prius[closest_ts_idx,2]]
        else:
            xy = np.array([gps_blue_prius_fx(ts), gps_blue_prius_fy(ts)])

        # try:
        #     yaw = yaw_blue_prius_f(ts)
        #     xy = np.array([gps_blue_prius_fx(ts), gps_blue_prius_fy(ts)])
        # except:
        #     closest_ts_idx = np.argmin(np.abs(yaw_blue_prius[:,0] - ts))
        #     yaw = yaw_blue_prius[closest_ts_idx,1]

        #     closest_ts_idx = np.argmin(np.abs(gps_blue_prius[:,0] - ts))
        #     xy = [gps_blue_prius[closest_ts_idx,1], gps_blue_prius[closest_ts_idx,2]]
        yaw_blue_prius_sync.append(yaw)
        xy_blue_prius_sync.append(xy)
    yaw_blue_prius_sync = np.array(yaw_blue_prius_sync)
    xy_blue_prius_sync = np.array(xy_blue_prius_sync)
else:
    yaw_blue_prius_sync = np.array([yaw_blue_prius_f(ts) for ts in gps_lexus[:,0]])
    xy_blue_prius_sync = np.array([[gps_blue_prius_fx(ts), gps_blue_prius_fy(ts)] for ts in gps_lexus[:,0]])
pose_blue_prius = np.concatenate([xy_blue_prius_sync, yaw_blue_prius_sync[:,None]], axis=1)

# get intervention
intervention = fetch_intervention(data_lexus)
intervention_x = np.array([gps_lexus_fx(t) for t in intervention])
intervention_y = np.array([gps_lexus_fy(t) for t in intervention])

# get road
assert len(sys.argv) >= 3, 'Need to specify reference trace path if ADD_ROAD is True'
ref_trace_dir = os.path.expanduser(sys.argv[2])
gps_ref = np.array(pd.read_csv(os.path.join(ref_trace_dir, 'gps.csv')))
gps_ref_time = gps_ref[:,0]
gps_ref = gps_ref[:,1:3]
gps_ref = np.array([utm.from_latlon(*v)[:2] for v in gps_ref])
gps_heading_ref = np.array(pd.read_csv(os.path.join(ref_trace_dir, 'gps_heading.csv')))
gps_heading_ref_f = interp1d(gps_heading_ref[:,0], gps_heading_ref[:,1], fill_value='extrapolate')

gps_heading_ref_sync = np.array([gps_heading_ref_f(ts) for ts in gps_ref_time])
pose_road = np.concatenate([gps_ref, gps_heading_ref_sync[:,None]], axis=1)

# set color
cm_lexus = MplColorHelper('Greens', 0., MAX_TRAJ_LENGTH)
cm_blue_prius = MplColorHelper('Oranges', 0., MAX_TRAJ_LENGTH)
color_lexus = [cm_lexus.get_rgb(v) for v in gps_lexus[:,0]]
color_blue_prius = [cm_blue_prius.get_rgb(v) for v in gps_blue_prius[:,0]]

# plot
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('/home/tsunw/tmp/test.mp4', fourcc, 100, (300, 300))

fig, ax = plt.subplots(1, 1, figsize=(3,3))
xlim = [-20, 20]
ylim = [-20, 20]
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
artists = dict()
for i in range(pose_lexus.shape[0]):
    if (i % 1) != 0: # DEBUG
        continue

    if False: # DEBUG
        ref_pose = np.array([pose_lexus[i,0], pose_lexus[i,1], 0])
        road_in_ref = pose_road - ref_pose
        blue_prius_in_ref = pose_blue_prius[i] - ref_pose
        lexus_in_ref = pose_lexus[i] - ref_pose
    else: 
        ref_pose = pose_lexus[i]
        road_in_ref = compute_relative_transform(pose_road, ref_pose)
        blue_prius_in_ref = compute_relative_transform(pose_blue_prius[i:i+1], ref_pose)[0]
        lexus_in_ref = compute_relative_transform(pose_lexus[i:i+1], ref_pose)[0]

    #print(i, pose_blue_prius[i], pose_lexus[i])  #DEBUG
    print(i, pose_blue_prius[i], blue_prius_in_ref, ref_pose)

    if 'patch_road' in artists.keys():
        artists['patch_road'].remove()
    road_roi = np.logical_and(np.logical_and(road_in_ref[:,0] >= xlim[0], road_in_ref[:,0] <= xlim[1]),
                              np.logical_and(road_in_ref[:,1] >= ylim[0], road_in_ref[:,1] <= ylim[1]))
    road_in_ref_roi = road_in_ref[road_roi]
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

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:,:,:3]
    img = img[:,:,::-1] # to bgr
    writer.write(img)

writer.release()
