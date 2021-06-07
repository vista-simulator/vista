import argparse
import numpy as np
import pickle5 as pickle
import matplotlib.pyplot as plt


def collect_last_step_info(data, key, agent_id, exclude):
    results = []
    for ep_data in data:
        last_step = ep_data[-1]
        info = last_step[-1]
        ep_result = info[agent_id][key]
        if not np.any([info[agent_id][v] for v in exclude]):
            results.append(ep_result)
    return results


def collect_all_info(data, key, agent_id, exclude):
    results = []
    for ep_data in data:
        last_step = ep_data[-1]
        info = last_step[-1]
        if not np.any([info[agent_id][v] for v in exclude]):
            ep_result = []
            for step in ep_data:
                info = step[-1]
                ep_result.append(info[agent_id][key])
            results.append(ep_result)
    return results


def compute_success_rate(data, agent_id='agent_0', exclude=['trace_done']):
    results = collect_last_step_info(data, 'success', agent_id, exclude)
    return np.mean(results)


def compute_offlane_rate(data, agent_id='agent_0', exclude=['trace_done']):
    results = collect_last_step_info(data, 'off_lane', agent_id, exclude)
    return np.mean(results)


def compute_collision_rate(data, agent_id='agent_0', exclude=['trace_done']):
    results = collect_last_step_info(data, 'has_collided', agent_id, exclude)
    return np.mean(results)


def compute_maxrot_rate(data, agent_id='agent_0', exclude=['trace_done']):
    results = collect_last_step_info(data, 'max_rot', agent_id, exclude)
    return np.mean(results)


def compute_avg_maxlatshift(data, agent_id='agent_0', exclude=['trace_done']):
    results = collect_all_info(data, 'translation', agent_id, exclude)
    return np.mean([np.max(v) for v in results])


def plot_mean_metric_at_key(data, metric, key, bins, xlabel, agent_id='agent_0', 
                            exclude=['trace_done'], mode=2):
    results = collect_last_step_info(data, metric, agent_id, exclude)
    values = collect_all_info(data, key, agent_id, exclude)
    mean_values = np.array([np.mean(np.abs(v)) for v in values])
    
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel(xlabel)
    ylabel = ' '.join([v.capitalize() for v in metric.split('_')])
    ax.set_ylabel(ylabel)
    if mode == 1:
        mean_values_true = mean_values[np.array(results)]
        ax.hist(mean_values_true, bins)
    elif mode == 2:
        _, bin_edge = np.histogram(mean_values, bins)
        hist_true, _ = np.histogram(mean_values[np.array(results)], bin_edge)
        hist_false, _ = np.histogram(mean_values[np.logical_not(results)], bin_edge)
        rate = hist_true / (hist_true + hist_false)
        x = (bin_edge[1:] + bin_edge[:-1]) / 2.
        width = (bin_edge[1:] - bin_edge[:-1]) * 0.9
        ax.bar(x, rate, width=width)
    else:
        raise NotImplementedError('Unrecognized mode {}'.format(mode))


ALL_METRICS = {
    'success_rate': compute_success_rate,
    'offlane_rate': compute_offlane_rate,
    'collision_rate': compute_collision_rate,
    'maxrot_rate': compute_maxrot_rate,
    'avg_maxlatshift': compute_avg_maxlatshift,
}


ALL_PLOTS = {
    'mean_metric_at_key': plot_mean_metric_at_key,
}


def main():
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'rollout_path',
        type=str,
        help='Path to rollout.')
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['LaneFollowing', 'Overtaking'],
        help='Task.')
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
    args = parser.parse_args()

    # load results and print meta-info
    print('')
    print('Load from {}'.format(args.rollout_path))
    with open(args.rollout_path, 'rb') as f:
        data = pickle.load(f)
    n_episodes = len(data)
    episode_len = [len(v) for v in data]
    print('n_episodes: {}'.format(n_episodes))
    print('epsiode_len: {}'.format(episode_len))

    # compute and print metrics
    if args.task == 'Overtaking':
        metrics = ['success_rate', 'collision_rate', 'offlane_rate', 'maxrot_rate', 'avg_maxlatshift']
        plots = [
            ('mean_metric_at_key', {'metric': 'success', 'key': 'human_curvature', 'bins': 5, 'xlabel': 'Absolute Road Curvature'}),
            ('mean_metric_at_key', {'metric': 'success', 'key': 'other_velocity', 'bins': 5, 'xlabel': 'Other Vehicle Speed'}),
            ('mean_metric_at_key', {'metric': 'success', 'key': 'other_translation', 'bins': 5, 'xlabel': 'Other Vehicle Lateral Shift'}),
        ]
    else:
        raise NotImplementedError('Unrecognized task {}'.format(args.task))

    for metric in metrics:
        value = ALL_METRICS[metric](data, args.agent_id, args.exclude)
        print('{}: {}'.format(metric, value))
    print('')

    for plot, plot_kwargs in plots:
        ALL_PLOTS[plot](data, agent_id=args.agent_id, exclude=args.exclude, **plot_kwargs)
    plt.show()


if __name__ == '__main__':
    main()