import os
import argparse
import numpy as np
import pickle5 as pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
import seaborn as sns
sns.set(style='white')

from simple_metrics import append_poly_info, overwrite_with_new_overlap_threshold, compute_collision_rate


def main():
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'rollout_path',
        type=str,
        help='Path to rollout.')
    parser.add_argument(
        '--agent-id',
        type=str,
        default='agent_0',
        help='ID of the agent to be evaluated.')
    parser.add_argument(
        '--exclude',
        type=str,
        nargs='+',
        default=['trace_done'],
        help='Exclude episodes with certain terminal condition.')
    parser.add_argument(
        '--dilate',
        type=float,
        nargs='+',
        default=[0., 0.],
        help='Dilation of the ego-car.')
    args = parser.parse_args()

    if True: # DEBUG
        # load data
        print('')
        print('Load from {}'.format(args.rollout_path))
        with open(args.rollout_path, 'rb') as f:
            data = pickle.load(f)

        new_data = []
        for ep_data in data:
            last_step = ep_data[-1]
            info = last_step[-1]
            if not np.any([info[args.agent_id][v] for v in args.exclude]):
                new_data.append(ep_data)
        data = new_data

        # compute stats for different overlap threshold
        threshold_list = np.linspace(0., 0.05, 11)
        overwritten_data_list = []
        for threshold in threshold_list:
            overwritten_data = copy.deepcopy(data)
            append_poly_info(overwritten_data, args.dilate)
            overwrite_with_new_overlap_threshold(overwritten_data, threshold)
            overwritten_data_list.append(overwritten_data)

        # compute stats
        collision_rate_list = []
        for overwritten_data in overwritten_data_list:
            collision_rate = compute_collision_rate(overwritten_data, args.agent_id, args.exclude)
            collision_rate_list.append(collision_rate)
    else:
        threshold_list = np.linspace(0., 0.05, 6)
        collision_rate_list = [0.16752312435765673, 0.14, 0.10, 0.08, 0.03, 0.0020554984583761563]
    
    # plot
    label_size = 22
    number_ticklabel_size = 16

    fig, ax = plt.subplots(1, 1, figsize=(9,6))
    ax.set_xlabel('Polygon Overlap Threshold', fontsize=label_size)
    ax.set_ylabel('Intervention Rate (%)', fontsize=label_size)
    ax.xaxis.set_tick_params(labelsize=number_ticklabel_size)
    ax.yaxis.set_tick_params(labelsize=number_ticklabel_size)
    ax.grid(color='grey', alpha=0.5, linestyle='dashed', linewidth=0.5)
    ax.plot(threshold_list, np.array(collision_rate_list)*100, linewidth=3, marker='D', markersize=10)
    fig.tight_layout()
    fig.savefig('test.png')
    # fig.savefig('crash_at_overlap.pdf')
    import pdb; pdb.set_trace()



if __name__ == '__main__':
    main()
