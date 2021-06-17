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
from colorama import Fore, Back, Style
import cv2
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch


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


def check_turn_signal(data, check_values=[0, 2], topic='/lexus/pacmod/parsed_tx/turn_rpt'):
    signal = []
    for t, msg in data[topic]:
        if msg.manual_input in check_values:
            signal.append(t)
    has_turn_signal = len(signal) > 0
    return has_turn_signal


def generate_plot(data_lexus, data_blue_prius):
    gps_blue_prius = fetch_gps(data_blue_prius, '/blue_prius/oxts/gps/fix')
    gps_lexus = fetch_gps(data_lexus, '/lexus/oxts/gps/fix')
    origin = gps_lexus[0,1:]
    gps_blue_prius[:,1:] -= origin
    gps_lexus[:,1:] -= origin

    gps_lexus_fx = interp1d(gps_lexus[:,0], gps_lexus[:,1], fill_value='extrapolate')
    gps_lexus_fy = interp1d(gps_lexus[:,0], gps_lexus[:,2], fill_value='extrapolate')
    intervention = fetch_intervention(data_lexus)
    intervention_x = np.array([gps_lexus_fx(t) for t in intervention])
    intervention_y = np.array([gps_lexus_fy(t) for t in intervention])

    cm_lexus = MplColorHelper('Oranges', gps_lexus[0,0], gps_lexus[-1,0])
    cm_blue_prius = MplColorHelper('Blues', gps_blue_prius[0,0], gps_blue_prius[-1,0])
    color_lexus = [cm_lexus.get_rgb(v) for v in gps_lexus[:,0]]
    color_blue_prius = [cm_blue_prius.get_rgb(v) for v in gps_blue_prius[:,0]]

    fig, ax = plt.subplots(1, 1)
    ax.set_title('Turn signal activated = {}'.format(check_turn_signal(data_lexus)))
    ax.scatter(gps_lexus[:,1], gps_lexus[:,2], c=color_lexus, s=1, label='lexus')
    ax.scatter(gps_blue_prius[:,1], gps_blue_prius[:,2], c=color_blue_prius, s=1, label='blue prius')
    ax.scatter(intervention_x, intervention_y, c='r', label='intervention')
    plt.legend(handles=[mpatches.Patch(color=color_lexus[-1], label='lexus'),
                        mpatches.Patch(color=color_blue_prius[-1], label='blue prius'),
                        mpatches.Circle((0.5,0.5), radius=0.25, color='r', label='intervention')],
            handler_map={mpatches.Circle: HandlerEllipse()})
    fig.canvas.draw()

    return fig, ax


def ros_image_to_np(data):
    return np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)


def dump_video(data, video_path):
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (960, 600))
    for t, v in data['/lexus/camera_array/front_center/image_raw']:
        img = ros_image_to_np(v)
        out.write(img)
    out.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bag-root-dir',
        type=str,
        required=True,
        help='Root directory to all bags.')
    parser.add_argument(
        '--out-dir',
        type=str,
        required=True,
        help='Output directory.')
    parser.add_argument(
        '--dump-video',
        action='store_true',
        default=False,
        help='Dump video')
    args = parser.parse_args()
    
    out_dir = os.path.expanduser(args.out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    bag_path_root_dir = os.path.expanduser(args.bag_root_dir)
    for bag_path_dir in os.listdir(bag_path_root_dir):
        bag_path_dir = os.path.join(bag_path_root_dir, bag_path_dir)
        if os.path.isdir(bag_path_dir):
            trial_name = bag_path_dir.split('/')[-1]

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
                        fig, ax = generate_plot(data_lexus, data_blue_prius)
                        out_path = os.path.join(out_dir, trial_name+'.jpg')
                        fig.savefig(out_path)
                        plt.close('all')
                        del fig, ax
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

            # process camera bag
            if args.dump_video:
                camera_bag_path = glob.glob(os.path.join(bag_path_dir, 'camera*.bag'))
                if len(camera_bag_path) == 1:
                    try:
                        camera_bag_path = camera_bag_path[0]
                        data_camera = read_rosbag(camera_bag_path)
                        fail_to_read_bag = False
                    except KeyboardInterrupt:
                        return
                    except:
                        print(Fore.RED + 'Fail to read camera bag from {}'.format(camera_bag_path))
                        fail_to_read_bag = True
                    if not fail_to_read_bag:
                        try:
                            video_path = os.path.join(out_dir, trial_name+'.mp4')
                            dump_video(data_camera, video_path)
                        except KeyError:
                            m = re.search("'([^']*)'", traceback.format_exc())
                            key = m.group(1)
                            print(Fore.YELLOW + '[Video] Missing key {} at {}'.format(key, trial_name))
                        except KeyboardInterrupt:
                            return
                        except:
                            print(Fore.RED + 'Unexpected error at {}'.format(camera_bag_path))
                            print(Fore.RED + traceback.format_exc())
                else:
                    print(Fore.RED + 'Fail to properly get camera bags at {}'.format(bag_path_dir))


if __name__ == '__main__':
    main()
