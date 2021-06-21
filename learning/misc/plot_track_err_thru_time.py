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
    track_err = collect_all_info(data, 'following_lat_shift', args.agent_id, args.exclude)
    per_step_track_err = [[] for _ in range(max_horizon)]
    for ep_err in track_err:
        for s_i, step_err in enumerate(ep_err):
            per_step_track_err[s_i].append(step_err)
    avg_track_err = np.array([np.mean(v) for v in per_step_track_err])
    std_track_err = np.array([np.std(v) for v in per_step_track_err])

    # plot
    label_size = 22
    number_ticklabel_size = 20
    legend_size = 20
    use_time_s = True

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    if use_time_s:
        ax.set_xlabel('Time (sec)', fontsize=label_size)
    else:
        ax.set_xlabel('Timesteps', fontsize=label_size)
    ax.set_ylabel('Mean Tracking Error (m)', fontsize=label_size)
    ax.xaxis.set_tick_params(labelsize=number_ticklabel_size)
    ax.yaxis.set_tick_params(labelsize=number_ticklabel_size)
    x_vals = np.arange(max_horizon) + 1
    if use_time_s:
        x_vals = x_vals / 30.
    ax.plot(x_vals, avg_track_err, linewidth=3)
    # ax.fill_between(x_vals, (avg_track_err - 2*std_track_err), (avg_track_err + 2*std_track_err), color='b', alpha=.1)
    ylim = ax.get_ylim()
    for vert_vals in range(0, max_horizon, lane_change_freq)[1:]:
        if use_time_s:
            vert_vals = vert_vals / 30.
        vline = ax.vlines(vert_vals, ymin=ylim[0], ymax=ylim[1], linestyle='dashed', linewidth=3, color='tab:orange')
    ax.set_ylim(*ylim)
    ax.legend([vline], ['Front Car\nLane Change'], fontsize=legend_size, framealpha=1.0)
    ax.grid(color='grey', alpha=0.5, linestyle='dashed', linewidth=0.5)
    fig.tight_layout()
    fig.savefig('test.png')
    # fig.savefig('track_err_thru_time.pdf')
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
