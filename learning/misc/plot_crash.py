import argparse
import numpy as np
import pickle5 as pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from shapely.geometry import box as Box
from shapely import affinity
from descartes import PolygonPatch

from simple_metrics import append_poly_info, overwrite_with_new_overlap_threshold


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
        '--max-episodes',
        type=int,
        default=100,
        help='Maximal number of episodes to be plotted.')
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

    # load data and only keep episodes that end with crash
    print('')
    print('Load from {}'.format(args.rollout_path))
    with open(args.rollout_path, 'rb') as f:
        data = pickle.load(f)

    append_poly_info(data, args.dilate)
    overwrite_with_new_overlap_threshold(data, args.threshold)

    new_data = []
    for ep_data in data:
        last_step = ep_data[-1]
        info = last_step[-1]
        if not np.any([info[args.agent_id][v] for v in args.exclude]):
            new_data.append(ep_data)
    data = new_data

    new_data = []
    for ep_data in data:
        last_step = ep_data[-1]
        info = last_step[-1]
        if info[args.agent_id]['has_collided']:
            new_data.append(ep_data)
    data = new_data

    keep_idcs = np.random.permutation(np.arange(len(data)))[:args.max_episodes]
    data = [v for i, v in enumerate(data) if i in keep_idcs]

    n_episodes = len(data)
    episode_len = [len(v) for v in data]
    print('n_episodes: {}'.format(n_episodes))
    print('epsiode_len: {}'.format(episode_len))

    # generate color library
    colors = list(cm.get_cmap('Set2').colors)
    if True:
        colors = [list(c) for c in colors]
    else:
        rgba2rgb = lambda rgba: np.clip((1 - rgba[:3]) * rgba[3] + rgba[:3], 0., 1.)
        colors = [np.array(list(c) + [0.3]) for c in colors]
        colors = list(map(rgba2rgb, colors))

    ref_alpha = 0.04 # 0.03
    traj_alpha = 0.1
    ego_color = colors[0] + [ref_alpha]
    other_color = colors[1] + [max(ref_alpha / n_episodes, 0.02)] # 0.002

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6,10))
    ax.set_title('Relative Trajectories\nWith Intervention', fontsize=40)
    fig.subplots_adjust(left=0.025, bottom=0.0, right=0.975, top=0.876, wspace=0.2, hspace=0.2)
    ax.set_xlim(-6., 6.)
    ax.set_ylim(-16., 8.)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ego_car_dim = (5, 2) # length, width
    for ep_data in data:
        # plot last step
        last_step_idx = np.min([_i for _i, _v in enumerate(ep_data) if _v[-1][args.agent_id]['overlap_ratio'] > args.threshold])
        last_step = ep_data[last_step_idx]
        agent_info = last_step[-1][args.agent_id]
        
        ego_poly = get_poly(agent_info['pose_wrt_others'], ego_car_dim)
        other_poly = get_poly(np.array([0., 0., 0.]), agent_info['other_dim'])

        ego_patch = PolygonPatch(ego_poly, fc=ego_color, ec=ego_color, zorder=2)
        other_patch = PolygonPatch(other_poly, fc=other_color, ec=other_color, zorder=1)

        ax.add_patch(ego_patch)
        ax.add_patch(other_patch)

        # plot traj
        all_pose = np.array([step[-1][args.agent_id]['pose_wrt_others'] for step in ep_data[:last_step_idx]])
        ax.plot(all_pose[:,0], all_pose[:,1], c=ego_color[:3]+[traj_alpha])
    ax.legend(handles=[mpatches.Patch(color=ego_color[:3]+[0.3], label='Ego Car'),
                        mpatches.Patch(color=other_color[:3]+[0.3], label='Front Car')],
              fontsize=34, loc='lower left')
    fig.tight_layout()
    # plt.show()
    fig.savefig('test.png') # DEBUG
    # fig.savefig('crash_traj.pdf')
    import pdb; pdb.set_trace()


def get_poly(pose, car_dim):
    x, y, theta = pose
    car_length, car_width = car_dim
    poly = Box(x-car_width/2., y-car_length/2., x+car_width/2., y+car_length/2.)
    poly = affinity.rotate(poly, np.degrees(theta))
    return poly


if __name__ == '__main__':
    main()
