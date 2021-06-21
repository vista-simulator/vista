import os
import yaml
import argparse
import numpy as np
import pickle5 as pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
sns.set(style='white')

from simple_metrics import collect_all_info


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
        default=[],
        help='Exclude episodes with certain terminal condition.')
    args = parser.parse_args()

    # load data and only keep episodes that end with crash
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

    n_episodes = len(data)
    episode_len = [len(v) for v in data]
    print('n_episodes: {}'.format(n_episodes))
    print('epsiode_len: {}'.format(episode_len))

    # get config
    config_fpath = os.path.join(os.path.dirname(args.rollout_path), 'rollout_eval_config.yaml')
    with open(config_fpath, 'r') as f:
        config = yaml.load(f, Loader=yaml.BaseLoader) # drop correctly loading python object
    lane_change_freq = int(config['env_config']['lane_change_freq'])
    max_horizon = config['env_config'].get('max_horizon', 500)

    # compute stats
    n_bins = 15
    ego_translation = collect_all_info(data, 'translation', args.agent_id, args.exclude)
    other_translation = collect_all_info(data, 'other_translation', args.agent_id, args.exclude)
    lat_shift = [vv1 - vv2 for v1, v2 in zip(ego_translation, other_translation) for vv1, vv2 in zip(v1, v2)]
    hist, bin_edge = np.histogram(lat_shift, 15)
    
    # plot
    label_size = 22
    number_ticklabel_size = 16
    legend_size = 18
    annot_color = 'tab:orange'

    fig, ax = plt.subplots(1, 1, figsize=(9,6))
    bin_interval = bin_edge[-1] - bin_edge[-2]
    bar_x = [(bin_edge[i] + bin_edge[i+1])/2. for i in range(bin_edge.shape[0]-1)]
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,3))
    ax.bar(bar_x, hist, width=bin_interval)
    ylim = ax.get_ylim()
    vline = ax.vlines(0, *ylim, linewidth=3, linestyle='dashed', colors=annot_color)
    ax.set_ylim(*ylim)
    ax.set_xticks(bin_edge)
    ax.set_xticklabels(['{:.2f}'.format(v) for v in bin_edge])
    ax.xaxis.set_tick_params(rotation=60)
    ax.set_xlabel('Signed Tracking Error (m)', fontsize=label_size)
    ax.set_ylabel('Number Of Steps', fontsize=label_size)
    ax.xaxis.set_tick_params(labelsize=number_ticklabel_size)
    ax.yaxis.set_tick_params(labelsize=number_ticklabel_size)
    ax.grid(color='grey', alpha=0.5, linestyle='dashed', linewidth=0.5)
    ax.legend([vline], ['Zero'], fontsize=legend_size)
    fig.text(0.42, 0.75, 'Left', color=annot_color, fontsize=label_size, fontweight='bold')
    fig.text(0.7, 0.75, 'Right', color=annot_color, fontsize=label_size, fontweight='bold')
    fig.tight_layout()
    fig.savefig('test.png')
    # fig.savefig('track_rel_pose.pdf')
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
