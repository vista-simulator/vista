import os
import sys
from pprint import pprint
import rosbag
import glob
import numpy as np
import utm
import re
import traceback
import argparse
import pickle5 as pickle
from colorama import Fore, Back, Style
import cv2
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from shapely.geometry import box as Box
from shapely import affinity
import pandas as pd


LEXUS_SIZE = (5, 1.9) #(5, 1.9)
BLUE_PRIUS_SIZE = (4.5, 1.8) #(4.5, 1.8)


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


def fetch_yaw(data, topic='/lexus/oxts/imu/data'):
    yaws = []
    for t, msg in data[topic]:
        q = msg.orientation
        yaw = np.arctan2(2.*(q.x*q.y + q.z*q.w) , (q.w**2 - q.z**2 - q.y*2 + q.x**2))
        yaws.append([t, yaw])
    yaws = np.array(yaws)
    return yaws


def fetch_intervention(data, topic='/lexus/ssc/module_states'):
    intervention = []
    for t, msg in data[topic]:
        if msg.info == 'Operator Override':
            intervention.append(t)
    return intervention


def get_poly(pose, car_dim):
    x, y, theta = pose
    car_length, car_width = car_dim
    poly = Box(x-car_width/2., y-car_length/2., x+car_width/2., y+car_length/2.)
    poly = affinity.rotate(poly, np.degrees(theta))
    return poly


def compute_relative_transform(pose, ref_pose):
    x_state, y_state, theta_state = pose
    ref_x_state, ref_y_state, ref_theta_state = ref_pose

    c = np.cos(ref_theta_state)
    s = np.sin(ref_theta_state)
    R_2 = np.array([[c, -s], [s, c]])
    xy_global_centered = np.array([[x_state - ref_x_state],
                                    [ref_y_state - y_state]])
    [[translation_x], [translation_y]] = np.matmul(R_2, xy_global_centered)
    translation_y *= -1  # negate the longitudinal translation (due to VS setup)

    theta = theta_state - ref_theta_state
    if theta > np.pi/2.:
        theta -= np.pi

    return translation_x, translation_y, theta


def get_statistics(data_lexus, data_blue_prius, ref_trace_dir):
    stats = dict()

    # get gps data for two car
    gps_blue_prius = fetch_gps(data_blue_prius, '/blue_prius/oxts/gps/fix')
    gps_lexus = fetch_gps(data_lexus, '/lexus/oxts/gps/fix')

    # get road gps
    ref_trace_dir = os.path.expanduser(ref_trace_dir)
    gps_ref = np.array(pd.read_csv(os.path.join(ref_trace_dir, 'gps.csv')))
    gps_ref_time = gps_ref[:,0]
    gps_ref = gps_ref[:,1:3]
    gps_ref = np.array([utm.from_latlon(*v)[:2] for v in gps_ref])
    gps_heading_ref = np.array(pd.read_csv(os.path.join(ref_trace_dir, 'gps_heading.csv')))
    gps_heading_ref_f = interp1d(gps_heading_ref[:,0], gps_heading_ref[:,1])

    # align at center
    origin = np.concatenate([gps_lexus[:,1:], gps_blue_prius[:,1:]], axis=0).mean(0)
    gps_ref = gps_ref - origin
    gps_blue_prius[:,1:] -= origin
    gps_lexus[:,1:] -= origin

    # get interp data
    gps_lexus_fx = interp1d(gps_lexus[:,0], gps_lexus[:,1], fill_value='extrapolate')
    gps_lexus_fy = interp1d(gps_lexus[:,0], gps_lexus[:,2], fill_value='extrapolate')
    gps_blue_prius_fx = interp1d(gps_blue_prius[:,0], gps_blue_prius[:,1], fill_value='extrapolate')
    gps_blue_prius_fy = interp1d(gps_blue_prius[:,0], gps_blue_prius[:,2], fill_value='extrapolate')

    yaw_lexus = fetch_yaw(data_lexus, '/lexus/oxts/imu/data')
    yaw_blue_prius = fetch_yaw(data_blue_prius, '/blue_prius/oxts/imu/data')
    yaw_lexus_f = interp1d(yaw_lexus[:,0], yaw_lexus[:,1], fill_value='extrapolate')
    yaw_blue_prius_f = interp1d(yaw_blue_prius[:,0], yaw_blue_prius[:,1], fill_value='extrapolate')

    # get interevention
    intervention = fetch_intervention(data_lexus)
    stats['has_intervention'] = len(intervention) > 0
    if intervention != []:
        gps_lexus = gps_lexus[gps_lexus[:,0]<intervention[0]]
        gps_blue_prius = gps_blue_prius[gps_blue_prius[:,0]<intervention[0]]
        intervention = intervention[0:1]
    intervention_x = np.array([gps_lexus_fx(t) for t in intervention])
    intervention_y = np.array([gps_lexus_fy(t) for t in intervention])

    # compute distance
    stats['poly_dist'] = []
    stats['rel_to_road'] = []
    stats['passed'] = []
    for i in range(gps_lexus.shape[0]):
        # get pose of two cars
        ts = gps_lexus[i,0]
        xy_lexus = gps_lexus[i,1:]
        theta_lexus = yaw_lexus_f(ts)
        pose_lexus = list(xy_lexus)+[theta_lexus]
        xy_blue_prius = np.array([gps_blue_prius_fx(ts), gps_blue_prius_fy(ts)])
        theta_blue_prius = yaw_blue_prius_f(ts)
        pose_blue_prius = list(xy_blue_prius)+[theta_blue_prius]

        # skip if doing extrapolation
        if ts < gps_blue_prius[0,0] or ts > gps_blue_prius[-1,0] or \
           ts < yaw_blue_prius[0,0] or ts > yaw_blue_prius[-1,0]:
           continue
        
        # compute polygon distance
        poly_lexus = get_poly(pose_lexus, LEXUS_SIZE)
        poly_blue_prius = get_poly(pose_blue_prius, BLUE_PRIUS_SIZE)
        poly_dist = poly_lexus.distance(poly_blue_prius)
        stats['poly_dist'].append(poly_dist)

        # get closest pose at road
        idx = np.linalg.norm(gps_ref - xy_lexus, axis=1).argmin()
        closest_ts = gps_ref_time[idx]
        xy_road = gps_ref[idx]
        theta_road = gps_heading_ref_f(closest_ts)
        pose_road = list(xy_road)+[theta_road]
        dx, dy, dtheta = compute_relative_transform(pose_lexus, pose_road)
        stats['rel_to_road'].append([dx, dy, dtheta])

        # check if pass
        _, trans_long, _ = compute_relative_transform(pose_lexus, pose_blue_prius)
        passed = trans_long > 0
        stats['passed'].append(passed)

    stats['poly_dist'] = np.array(stats['poly_dist'])
    stats['rel_to_road'] = np.array(stats['rel_to_road'])
    stats['passed'] = np.array(stats['passed'])

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bag-root-dir',
        type=str,
        required=True,
        help='Root directory to all bags.')
    parser.add_argument(
        '--ref-trace-dir',
        type=str,
        required=True,
        help='Directory to reference trace.')
    parser.add_argument(
        '--include',
        type=str,
        nargs='+',
        default=None,
        help='Include bag with subnames. Default None to include all')
    parser.add_argument(
        '--out-path',
        type=str,
        required=True,
        help='Output path.')
    args = parser.parse_args()

    total_stats = {
        'has_intervention': [],
        'min_dist': [],
        'max_rot': [],
        'max_dev': [],
        'passed': [],
        'poly_dist': [],
        'rel_to_road': [],
    }

    bag_path_root_dir = os.path.expanduser(args.bag_root_dir)
    for bag_path_dir in os.listdir(bag_path_root_dir):
        bag_path_dir = os.path.join(bag_path_root_dir, bag_path_dir)
        if os.path.isdir(bag_path_dir):
            trial_name = bag_path_dir.split('/')[-1]

            skip = False
            if args.include is not None:
                for include in args.include:
                    if include not in trial_name:
                        skip = True
                        break
            if skip:
                print(Fore.WHITE + 'Not included. Skip {}'.format(trial_name))
                continue

            ## HACKY
            if trial_name in ['20210615-185738_left_position1_static_trial2',
                '20210615-203533_right_position3_dynamic_trial4']:
                print(Fore.WHITE + 'Invalid trial. Skip {}'.format(trial_name)) 
                continue

            # process sensor bag
            bag_path = glob.glob(os.path.join(bag_path_dir, 'sensor*.bag'))
            if len(bag_path) == 1:
                bag_path = bag_path[0]
                try:
                    data_lexus = read_rosbag(bag_path)
                    data_blue_prius = data_lexus
                    fail_to_read_bag = False
                except KeyboardInterrupt:
                    return
                except:
                    print(Fore.RED + 'Fail to read bag from {}'.format(bag_path))
                    fail_to_read_bag = True
                if not fail_to_read_bag:
                    try:
                        stats = get_statistics(data_lexus, data_blue_prius, args.ref_trace_dir)
                        has_intervention = stats['has_intervention']
                        min_dist = stats['poly_dist'].min()
                        max_rot = np.abs(stats['rel_to_road'][:,2]).max()
                        passed = np.any(stats['passed'])
                        if passed:
                            max_dev = np.abs(stats['rel_to_road'][:,0][stats['passed']]).max()
                            total_stats['max_dev'].append(max_dev)

                        ### HACKY: forgot to stop recording
                        if trial_name == '20210615-202314_right_position3_dynamic_trial3':
                            has_intervention = False 
                        
                        total_stats['has_intervention'].append(has_intervention)
                        total_stats['min_dist'].append(min_dist)
                        total_stats['max_rot'].append(max_rot)

                        total_stats['poly_dist'].append(stats['poly_dist'])
                        total_stats['rel_to_road'].append(stats['rel_to_road'])
                        total_stats['passed'].append(stats['passed'])
                    except KeyError:
                        m = re.search("'([^']*)'", traceback.format_exc())
                        key = m.group(1)
                        print(Fore.YELLOW + 'Missing key {} at {}'.format(key, trial_name))
                    except KeyboardInterrupt:
                        return
                    except:
                        print(Fore.RED + 'Unexpected error at {}'.format(bag_path))
                        print(Fore.RED + traceback.format_exc())
            else:
                print(Fore.RED + 'Fail to properly get bags at {}'.format(bag_path_dir))

    with open(args.out_path, 'wb') as f:
        pickle.dump(total_stats, f)
    print('=================================')
    print('Intervention: {} / {} ({:.3f})'.format(np.sum(total_stats['has_intervention']), 
        len(total_stats['has_intervention']), np.mean(total_stats['has_intervention'])))
    print('Min Dist: {} ({})'.format(np.mean(total_stats['min_dist']), np.std(total_stats['min_dist'])))
    print('Max Dev: {} ({})'.format(np.mean(total_stats['max_dev']), np.std(total_stats['max_dev'])))
    print('Max Rot: {} ({})'.format(np.mean(total_stats['max_rot']), np.std(total_stats['max_rot'])))


if __name__ == '__main__':
    main()
