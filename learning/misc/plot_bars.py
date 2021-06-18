import argparse
import numpy as np
from numpy.lib.function_base import append
import pickle5 as pickle
import matplotlib.pyplot as plt
from shapely.geometry import box as Box
from shapely import affinity
import seaborn as sns
sns.set(style='white')

from simple_metrics import collect_last_step_info, collect_all_info, append_poly_info, overwrite_with_new_overlap_threshold


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

    n_bins = 4
    xticklabels = ['Easy', 'Normal', 'Hard', 'Challenging']
    assert len(xticklabels) == n_bins

    # different road curvature (easy -> hard)
    mean_values, bin_edge = get_bin_edge(data, 'human_curvature', args.agent_id, args.exclude, n_bins)
    per_bin_crash_rate_at_curv = compute_per_bin_crash_rate(data, args.agent_id, args.exclude, mean_values, bin_edge)
    per_bin_max_dev_at_curv = compute_per_bin_max_dev(data, args.agent_id, args.exclude, mean_values, bin_edge)

    # different car velocity (easy -> hard)
    mean_values, bin_edge = get_bin_edge(data, 'other_velocity', args.agent_id, args.exclude, n_bins)
    per_bin_crash_rate_at_velo = compute_per_bin_crash_rate(data, args.agent_id, args.exclude, mean_values, bin_edge)
    per_bin_max_dev_at_velo = compute_per_bin_max_dev(data, args.agent_id, args.exclude, mean_values, bin_edge)

    # different initial condition (hard -> easy) --> (easy --> hard)
    mean_values, bin_edge = get_bin_edge(data, 'other_translation', args.agent_id, args.exclude, n_bins)
    per_bin_crash_rate_at_init = compute_per_bin_crash_rate(data, args.agent_id, args.exclude, mean_values, bin_edge)
    per_bin_crash_rate_at_init = per_bin_crash_rate_at_init[::-1]
    per_bin_max_dev_at_init = compute_per_bin_max_dev(data, args.agent_id, args.exclude, mean_values, bin_edge)
    per_bin_max_dev_at_init = per_bin_max_dev_at_init[::-1]

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(20,6))
    fig.subplots_adjust(left=0.06, bottom=0.092, right=0.991, top=0.97, wspace=0.24, hspace=0.2)
    bw = 0.25
    label_size = 22
    ticklabel_size = 20
    number_ticklabel_size = 16
    legend_size = 18
    x = np.arange(n_bins) + 1

    ax = axes[0]
    ax.bar(x-bw, per_bin_crash_rate_at_curv, width=bw, align='center', label='Road Curvature')
    ax.bar(x, per_bin_crash_rate_at_velo, width=bw, align='center', label='Front Car Speed')
    ax.bar(x+bw, per_bin_crash_rate_at_init, width=bw, align='center', label='Initial Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=ticklabel_size)
    ax.set_ylabel('Intervention', fontsize=label_size)
    ax.yaxis.set_tick_params(labelsize=number_ticklabel_size)
    ax.grid(color='grey', alpha=0.5, linestyle='dashed', linewidth=0.5)
    ax.legend(fontsize=legend_size)

    ax = axes[1]
    ax.bar(x-bw, per_bin_max_dev_at_curv, width=bw, align='center', label='Road Curvature')
    ax.bar(x, per_bin_max_dev_at_velo, width=bw, align='center', label='Car Speed')
    ax.bar(x+bw, per_bin_max_dev_at_init, width=bw, align='center', label='Initial Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=ticklabel_size)
    ax.set_ylabel('Maximal Deviation', fontsize=label_size)
    ax.yaxis.set_tick_params(labelsize=number_ticklabel_size)
    ax.grid(color='grey', alpha=0.5, linestyle='dashed', linewidth=0.5)

    # fig.tight_layout()
    fig.savefig('intervention_max_dev_at.pdf')
    fig.savefig('test.png')
    # plt.show()
    import pdb; pdb.set_trace()



def get_bin_edge(data, key, agent_id, exclude, n_bins):
    values = collect_all_info(data, key, agent_id, exclude)
    mean_values = np.array([np.mean(np.abs(v)) for v in values])
    _, bin_edge = np.histogram(mean_values, n_bins)
    return mean_values, bin_edge


def compute_per_bin_crash_rate(data, agent_id, exclude, mean_values, bin_edge):
    crash = np.array(collect_last_step_info(data, 'has_collided', agent_id, exclude))
    hist_true, _ = np.histogram(mean_values[crash], bin_edge)
    hist_false, _ = np.histogram(mean_values[np.logical_not(crash)], bin_edge)
    crash_rate = hist_true / (hist_true + hist_false)
    return crash_rate


def compute_per_bin_max_dev(data, agent_id, exclude, mean_values, bin_edge):
    max_dev = collect_all_info(data, 'translation', agent_id, exclude)
    max_dev = np.array([np.max(v) for v in max_dev])
    per_bin_max_dev = []
    for i in range(bin_edge.shape[0] - 1):
        low = bin_edge[i]
        high = bin_edge[i+1]
        mask = np.logical_and(mean_values >= low, mean_values <= high)
        per_bin_max_dev.append(np.mean(max_dev[mask]))
    per_bin_max_dev = np.array(per_bin_max_dev)
    return per_bin_max_dev


if __name__ == '__main__':
    main()
