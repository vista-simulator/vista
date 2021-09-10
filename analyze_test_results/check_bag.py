import os
import argparse
from pprint import pprint
import matplotlib.pyplot as plt

from utils import *


def main():
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bag-path',
        type=str,
        required=True,
        help='Path to rosbag.')
    args = parser.parse_args()

    args.bag_path = validate_path(args.bag_path)

    # Extract data from bag
    data = read_rosbag(args.bag_path)
    print('Topics')
    pprint(list(data.keys()))
    gps = fetch_gps(data)
    intervention = fetch_intervention(data)
    curvature_command = fetch_curvature(data, '/lexus/deepknight/goal_steer')
    curvature_feedback = fetch_curvature(data, '/lexus/ssc/curvature_feedback')
    speed = fetch_speed(data)

    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(8,18))
    ts_origin = curvature_command[0,0]
    visualize_gps(gps, intervention, figax=[fig, axes[0]])
    # visualize_curvature(curvature_command, curvature_feedback, 
    #                     figax=[fig, axes[1]], ts_origin=ts_origin)
    for cc_seg, cf_seg in zip(split_to_segments(curvature_command), 
                              split_to_segments(curvature_feedback)):
        ax = axes[1]
        visualize_curvature(cc_seg, cf_seg, figax=[fig, ax], ts_origin=ts_origin)
        ax.axvline(cc_seg[0,0] - ts_origin, linestyle='--', c='grey')
    for speed_seg in split_to_segments(speed):
        ax = axes[2]
        visualize_speed(speed_seg, figax=[fig, ax], ts_origin=ts_origin)
        ax.axvline(speed_seg[0,0] - ts_origin, linestyle='--', c='grey')
    
    fig.tight_layout()

    # Save results
    out_path = os.path.join(os.path.dirname(args.bag_path), 'sensor_bag.png')
    fig.savefig(out_path)


if __name__ == '__main__':
    main()