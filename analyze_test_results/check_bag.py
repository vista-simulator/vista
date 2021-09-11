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
    parser.add_argument(
        '--devens-road-path',
        type=str,
        default=None,
        help='Path to the pickle file that stores Devens road')
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
    if args.devens_road_path is not None:
        args.devens_road_path = validate_path(args.devens_road_path)
        devens_road = load_devens_road(args.devens_road_path)
        plot_devens_road(devens_road, [fig, axes[0]])
    visualize_gps(gps, intervention, figax=[fig, axes[0]])

    use_speed_filter = True
    if use_speed_filter: # filter out low-speed segment
        min_speed = 3.
        keep_ts = speed[speed[:,1] > min_speed,0]
        keep_ts = [v[:,0] for v in split_to_segments(keep_ts[:,None])]

    # visualize_curvature(curvature_command, curvature_feedback, 
    #                     figax=[fig, axes[1]], ts_origin=ts_origin)
    plot_legend = True
    for seg_i, (cc_seg, cf_seg) in enumerate(zip(split_to_segments(curvature_command), 
                                               split_to_segments(curvature_feedback))):
        ax = axes[1]
        ax.axvline(cc_seg[-1,0] - ts_origin, linestyle='--', c='grey')
        ax.text(cc_seg[-1,0] - ts_origin, ax.get_ylim()[1] if seg_i !=0 else cc_seg[:,1].max(), f'{seg_i+1}', rotation=0)
        if use_speed_filter:
            cc_seg = filter_with_ts(cc_seg, keep_ts)
            cf_seg = filter_with_ts(cf_seg, keep_ts)
        visualize_curvature(cc_seg, cf_seg, figax=[fig, ax], ts_origin=ts_origin)
        if plot_legend:
            ax.legend()
            plot_legend = False

    for seg_i, speed_seg in enumerate(split_to_segments(speed)):
        ax = axes[2]
        ax.axvline(speed_seg[-1,0] - ts_origin, linestyle='--', c='grey')
        ax.text(speed_seg[-1,0] - ts_origin, ax.get_ylim()[1] if seg_i !=0 else speed_seg[:,1].max(), f'{seg_i+1}', rotation=0)
        if use_speed_filter:
            speed_seg = filter_with_ts(speed_seg, keep_ts)
        visualize_speed(speed_seg, figax=[fig, ax], ts_origin=ts_origin)
    
    fig.tight_layout()

    # Save results
    out_path = os.path.join(os.path.dirname(args.bag_path), 'sensor_bag.png')
    fig.savefig(out_path)
    print(f'save to {out_path}')


def filter_with_ts(data, keep_ts):
    ts = data[0,0]
    keep_ts_idx = np.argmin([np.abs(v[0] - ts) for v in keep_ts])
    mask = np.logical_and(data[:,0] > keep_ts[keep_ts_idx][0], data[:,0] < keep_ts[keep_ts_idx][-1])
    return data[mask]


def is_segment(full, segment):
    return np.all([s in full for s in segment])


if __name__ == '__main__':
    main()