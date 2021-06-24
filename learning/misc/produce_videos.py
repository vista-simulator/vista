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
import pandas as pd
from tqdm import tqdm

from to_trace_video import read_rosbag, generate_video


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
        '--ref-trace-dir',
        type=str,
        required=True,
        help='Directory to reference trace.')
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

            # create output subdir
            out_subdir = os.path.join(out_dir, trial_name)
            if not os.path.isdir(out_subdir):
                os.makedirs(out_subdir)

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
                        out_path = os.path.join(out_subdir, 'bev.mp4')
                        print('Parse from {}'.format(bag_path))
                        generate_video(data_lexus, data_blue_prius, args.ref_trace_dir, out_path, False, False, False)
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
                            video_path = os.path.join(out_subdir, 'front_camera.mp4')
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
        print(Fore.WHITE + '', end='')


if __name__ == '__main__':
    main()
