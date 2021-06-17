import os
import sys
from pprint import pprint
import rosbag
import numpy as np
import utm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from shapely.geometry import box as Box
from shapely import affinity
from shapely.geometry import LineString
from descartes import PolygonPatch
import pandas as pd


SYNCED_BAGS = True
DROP_AFTER_INTERVENTION = False
ADD_POLYGON = True
CALIBRATE_VIEW = True
ADD_ROAD = True
ROAD_HALF_WIDTH = 3.5
LEXUS_SIZE = (5, 2.0) #(5, 1.9)
BLUE_PRIUS_SIZE = (4.5, 2.0) #(4.5, 1.8)


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
        gps.append([t, x, y])
    gps = np.array(gps)
    origin = gps[0,1:]
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


if SYNCED_BAGS:
    bag_path_lexus = os.path.expanduser(sys.argv[1])

    data_lexus = read_rosbag(bag_path_lexus)
    data_blue_prius = data_lexus
else:
    bag_path_lexus = os.path.expanduser(sys.argv[1])
    bag_path_blue_prius = os.path.expanduser(sys.argv[2])

    data_lexus = read_rosbag(bag_path_lexus)
    data_blue_prius = read_rosbag(bag_path_blue_prius)

gps_blue_prius = fetch_gps(data_blue_prius, '/blue_prius/oxts/gps/fix')
gps_lexus = fetch_gps(data_lexus, '/lexus/oxts/gps/fix')

yaw_lexus = fetch_yaw(data_lexus, '/lexus/oxts/imu/data')
yaw_blue_prius = fetch_yaw(data_blue_prius, '/blue_prius/oxts/imu/data')
yaw_lexus_f = interp1d(yaw_lexus[:,0], yaw_lexus[:,1], fill_value='extrapolate')
yaw_blue_prius_f = interp1d(yaw_blue_prius[:,0], yaw_blue_prius[:,1], fill_value='extrapolate')

gps_lexus_fx = interp1d(gps_lexus[:,0], gps_lexus[:,1], fill_value='extrapolate')
gps_lexus_fy = interp1d(gps_lexus[:,0], gps_lexus[:,2], fill_value='extrapolate')
intervention = fetch_intervention(data_lexus)
if DROP_AFTER_INTERVENTION and intervention != []:
    gps_lexus = gps_lexus[gps_lexus[:,0]<intervention[0]]
    gps_blue_prius = gps_blue_prius[gps_blue_prius[:,0]<intervention[0]]
    intervention = intervention[0:1]
intervention_x = np.array([gps_lexus_fx(t) for t in intervention])
intervention_y = np.array([gps_lexus_fy(t) for t in intervention])

global_theta = 0.
if ADD_ROAD:
    assert len(sys.argv) >= 3, 'Need to specify reference trace path if ADD_ROAD is True'
    ref_trace_dir = os.path.expanduser(sys.argv[2])
    gps_ref = np.array(pd.read_csv(os.path.join(ref_trace_dir, 'gps.csv')))
    gps_ref_time = gps_ref[:,0]
    gps_ref = gps_ref[:,1:3]
    gps_ref = np.array([utm.from_latlon(*v)[:2] for v in gps_ref])
    gps_heading_ref = np.array(pd.read_csv(os.path.join(ref_trace_dir, 'gps_heading.csv')))
    gps_heading_ref_f = interp1d(gps_heading_ref[:,0], gps_heading_ref[:,1])

    origin = np.concatenate([gps_lexus[:,1:], gps_blue_prius[:,1:]], axis=0).mean(0)
    gps_ref = gps_ref - origin
    if CALIBRATE_VIEW:
        idx = np.linalg.norm(gps_ref, axis=1).argmin()
        ts = gps_ref_time[idx]
        global_theta = -(gps_heading_ref_f(ts) + np.pi)
else:
    origin = gps_lexus[0,1:]
    if CALIBRATE_VIEW:
        global_theta = -yaw_blue_prius_f(gps_blue_prius[0,0])

gps_blue_prius[:,1:] -= origin
gps_lexus[:,1:] -= origin

c, s = np.cos(global_theta), np.sin(global_theta)
rot_mat = np.array([[c, -s], [s, c]])
gps_lexus[:,1:] = np.matmul(gps_lexus[:,1:], rot_mat.T)
gps_blue_prius[:,1:] = np.matmul(gps_blue_prius[:,1:], rot_mat.T)
if ADD_ROAD:
    gps_ref = np.matmul(gps_ref, rot_mat.T)

cm_lexus = MplColorHelper('Greens', gps_lexus[0,0], gps_lexus[-1,0])
cm_blue_prius = MplColorHelper('Blues', gps_blue_prius[0,0], gps_blue_prius[-1,0])
color_lexus = [cm_lexus.get_rgb(v) for v in gps_lexus[:,0]]
color_blue_prius = [cm_blue_prius.get_rgb(v) for v in gps_blue_prius[:,0]]

fig, ax = plt.subplots(1, 1)
if CALIBRATE_VIEW:
    vmin = gps_lexus[:,1:].min() * 1.3
    vmax = gps_lexus[:,1:].max() * 1.4
    v_half_range = (vmax - vmin) / 2.
    ax.set_xlim(-v_half_range, v_half_range)
    ax.set_ylim(vmin, vmax)
else:
    vmin = gps_lexus[:,1:].min(axis=0)
    vmax = gps_lexus[:,1:].max(axis=0)
    lim_min = vmin.min() * 1.3
    lim_max = vmax.max() * 1.3
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
if ADD_POLYGON:
    lexus_pose = np.concatenate([gps_lexus[-1,1:], [yaw_lexus_f(gps_lexus[-1,0])+global_theta]])
    blue_prius_pose = np.concatenate([gps_blue_prius[-1,1:], [yaw_blue_prius_f(gps_blue_prius[-1,0])+global_theta]])
    poly_lexus = get_poly(lexus_pose, LEXUS_SIZE)
    poly_blue_prius = get_poly(blue_prius_pose, BLUE_PRIUS_SIZE)
    patch_lexus = PolygonPatch(poly_lexus, fc=color_lexus[-1], ec=color_lexus[-1], zorder=2)
    patch_blue_prius = PolygonPatch(poly_blue_prius, fc=color_blue_prius[-1], ec=color_blue_prius[-1], zorder=2)
    ax.add_patch(patch_lexus)
    ax.add_patch(patch_blue_prius)
if ADD_ROAD:
    road_patch = LineString(gps_ref).buffer(ROAD_HALF_WIDTH)
    road_patch = PolygonPatch(road_patch, fc='none', ec='black', zorder=1)
    ax.add_patch(road_patch)
# ax.set_title('Turn signal activated = {}'.format(check_turn_signal(data_lexus)))
ax.set_xticks([])
ax.set_yticks([])
ax.scatter(gps_lexus[:,1], gps_lexus[:,2], c=color_lexus, s=1, label='lexus')
ax.scatter(gps_blue_prius[:,1], gps_blue_prius[:,2], c=color_blue_prius, s=1, label='blue prius')
ax.scatter(intervention_x, intervention_y, c='r', label='intervention')
plt.legend(handles=[mpatches.Patch(color=color_lexus[-1], label='Ego-car'),
                    mpatches.Patch(color=color_blue_prius[-1], label='Front Car'),
                    mpatches.Circle((0.5,0.5), radius=0.25, color='r', label='Intervention')],
           handler_map={mpatches.Circle: HandlerEllipse()})
fig.tight_layout()
fig.canvas.draw()
fig.savefig('test.png')

