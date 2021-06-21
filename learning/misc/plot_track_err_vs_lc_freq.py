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
        'rollout_paths',
        type=str,
        nargs='+',
        default=[],
        help='Paths to rollout.')
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
    all_data = []
    all_configs = []
    for rollout_path in args.rollout_paths:
        all_data.append(load_data(rollout_path, args.agent_id, args.exclude))
        all_configs.append(get_config(rollout_path))

    # compute stats
    all_mean_track_err = []
    all_lane_change_freq = []
    for data, config in zip(all_data, all_configs):
        track_err = collect_all_info(data, 'following_lat_shift', args.agent_id, args.exclude)
        all_mean_track_err.append(np.mean([np.mean(v) for v in track_err]))
        all_lane_change_freq.append(int(config['env_config']['lane_change_freq']))

    # plot
    fig, ax = plt.subplots(1, 1)
    ax.plot(all_lane_change_freq, all_mean_track_err)
    fig.savefig('test.png')
    import pdb; pdb.set_trace() # DEBUG


def load_data(rollout_path, agent_id, exclude):
    print('')
    print('Load from {}'.format(rollout_path))
    with open(rollout_path, 'rb') as f:
        data = pickle.load(f)

    new_data = []
    for ep_data in data:
        last_step = ep_data[-1]
        info = last_step[-1]
        if not np.any([info[agent_id][v] for v in exclude]):
            new_data.append(ep_data)
    data = new_data

    n_episodes = len(data)
    episode_len = [len(v) for v in data]
    print('n_episodes: {}'.format(n_episodes))
    print('epsiode_len: {}'.format(episode_len))

    return data


def get_config(rollout_path):
    config_fpath = os.path.join(os.path.dirname(rollout_path), 'rollout_eval_config.yaml')
    with open(config_fpath, 'r') as f:
        config = yaml.load(f, Loader=yaml.BaseLoader) # drop correctly loading python object

    return config


if __name__ == '__main__':
    main()
