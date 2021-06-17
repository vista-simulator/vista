import argparse
import numpy as np
from numpy.lib.function_base import append
import pickle5 as pickle
import matplotlib.pyplot as plt
from shapely.geometry import box as Box
from shapely import affinity


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


def compute_maxyaw(data, agent_id='agent_0', exclude=['trace_done']):
    results = collect_all_info(data, 'rotation', agent_id, exclude)
    return np.mean([np.max(v) for v in results])


def compute_avg_maxlatshift(data, agent_id='agent_0', exclude=['trace_done']):
    results = collect_all_info(data, 'translation', agent_id, exclude)
    return np.mean([np.max(v) for v in results])


def compute_passed_cars_rate(data, agent_id='agent_0', exclude=['trace_done']):
    results = collect_last_step_info(data, 'passed_cars', agent_id, exclude)
    return np.mean(results)


def compute_poly_dist(data, agent_id='agent_0', exclude=['trace_done']):
    poly_dist = collect_all_info(data, 'poly_dist', agent_id, exclude)
    overlap_ratio = collect_all_info(data, 'overlap_ratio', agent_id, exclude)
    exclude = [np.max(v)>0 for v in overlap_ratio]
    return np.mean([np.min(v) for v, e in zip(poly_dist, exclude) if not e])


def append_poly_info(data, dilate=[0.,0.], agent_id='agent_0'):
    pose_wrt_others = collect_all_info(data, 'pose_wrt_others', agent_id, [])
    other_dim = collect_all_info(data, 'other_dim', agent_id, [])
    ego_dim = np.array([5., 2.]) + np.array(dilate)
    for i, (ep_pose, ep_other_dim) in enumerate(zip(pose_wrt_others, other_dim)):
        other_poly = get_poly([0., 0., 0.], ep_other_dim[0]) # reference
        for j, step_pose in enumerate(ep_pose):
            ego_poly = get_poly(step_pose, ego_dim)
            dist = ego_poly.distance(other_poly)
            overlap_ratio = ego_poly.intersection(other_poly).area / np.prod(ego_dim)
            data[i][j][-1][agent_id]['poly_dist'] = dist
            data[i][j][-1][agent_id]['overlap_ratio'] = overlap_ratio


def get_poly(pose, car_dim):
    x, y, theta = pose
    car_length, car_width = car_dim
    poly = Box(x-car_width/2., y-car_length/2., x+car_width/2., y+car_length/2.)
    poly = affinity.rotate(poly, np.degrees(theta))
    return poly


def overwrite_with_new_overlap_threshold(data, threshold=None, agent_id='agent_0'):
    if threshold is None:
        return
    overlap_ratio = collect_all_info(data, 'overlap_ratio', agent_id, [])
    for i in range(len(data)):
        cum_collide = 0
        for j in range(len(data[i])):
            crash = overlap_ratio[i][j] > threshold
            data[i][j][-1][agent_id]['collide'] = crash
            data[i][j][-1][agent_id]['cum_collide'] = cum_collide + int(crash)
            data[i][j][-1][agent_id]['has_collided'] = data[i][j][-1][agent_id]['cum_collide'] > 0
            cum_collide = data[i][j][-1][agent_id]['cum_collide']
            data[i][j][-1][agent_id]['success'] = data[i][j][-1][agent_id]['success'] and not data[i][j][-1][agent_id]['has_collided']
            data[i][j][-1][agent_id]['passed_cars'] = data[i][j][-1][agent_id]['passed_cars'] and not data[i][j][-1][agent_id]['has_collided']


def plot_mean_metric_at_key(data, metric, key, bins, xlabel, agent_id='agent_0', 
                            exclude=['trace_done'], mode=3):
    results = collect_last_step_info(data, metric, agent_id, exclude)
    values = collect_all_info(data, key, agent_id, exclude)
    mean_values = np.array([np.mean(np.abs(v)) for v in values])
    
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel(xlabel)
    ylabel = ' '.join([v.capitalize() for v in metric.split('_')])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0., 0.6)
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
    'maxyaw': compute_maxyaw,
    'avg_maxlatshift': compute_avg_maxlatshift,
    'passed_cars': compute_passed_cars_rate,
    'poly_dist': compute_poly_dist,
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
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='New overlap threshold for overwriting results.')
    parser.add_argument(
        '--dilate',
        type=float,
        nargs='+',
        default=[0., 0.],
        help='Dilation of the ego-car.')
    args = parser.parse_args()

    if args.dilate != [0., 0.] and args.threshold is None:
        raise ValueError('Should set threshold if using dilation, otherwise the results won\'t be updated')

    # load results and print meta-info
    print('')
    print('Load from {}'.format(args.rollout_path))
    with open(args.rollout_path, 'rb') as f:
        data = pickle.load(f)
    n_episodes = len(data)
    episode_len = [len(v) for v in data]
    print('n_episodes: {}'.format(n_episodes))
    print('epsiode_len: {}'.format(episode_len))

    append_poly_info(data, args.dilate)
    overwrite_with_new_overlap_threshold(data, args.threshold)

    # compute and print metrics
    if args.task == 'Overtaking':
        metrics = ['success_rate', 'collision_rate', 'offlane_rate', 'maxrot_rate', 'maxyaw', 'avg_maxlatshift', 'passed_cars', 'poly_dist']
        plots = [
            ('mean_metric_at_key', {'metric': 'has_collided', 'key': 'human_curvature', 'bins': 7, 'xlabel': 'Absolute Road Curvature'}),
            ('mean_metric_at_key', {'metric': 'has_collided', 'key': 'other_velocity', 'bins': 7, 'xlabel': 'Other Vehicle Speed'}),
            ('mean_metric_at_key', {'metric': 'has_collided', 'key': 'other_translation', 'bins': 5, 'xlabel': 'Other Vehicle Lateral Shift'}),
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