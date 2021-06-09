import argparse
import numpy as np
import pickle5 as pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from shapely.geometry import box as Box
from shapely import affinity
from descartes import PolygonPatch


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
    colors = list(cm.get_cmap('Set1').colors)
    if True:
        colors = [list(c) for c in colors]
    else:
        rgba2rgb = lambda rgba: np.clip((1 - rgba[:3]) * rgba[3] + rgba[:3], 0., 1.)
        colors = [np.array(list(c) + [0.3]) for c in colors]
        colors = list(map(rgba2rgb, colors))

    ref_alpha = 0.03
    traj_alpha = 0.1
    ego_color = colors[1] + [ref_alpha]
    other_color = colors[0] + [max(ref_alpha / n_episodes, 0.002)]

    # plot
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Trajectories and the last step of ego-car in crashes')
    ax.set_xlim(-6., 6.)
    ax.set_ylim(-8., 8.)
    ego_car_dim = (5, 2) # length, width
    for ep_data in data:
        # plot last step
        last_step = ep_data[-1]
        agent_info = last_step[-1][args.agent_id]
        
        ego_poly = get_poly(agent_info['pose_wrt_others'], ego_car_dim)
        other_poly = get_poly(np.array([0., 0., 0.]), agent_info['other_dim'])

        ego_patch = PolygonPatch(ego_poly, fc=ego_color, ec=ego_color, zorder=2)
        other_patch = PolygonPatch(other_poly, fc=other_color, ec=other_color, zorder=1)

        ax.add_patch(ego_patch)
        ax.add_patch(other_patch)

        # plot traj
        all_pose = np.array([step[-1][args.agent_id]['pose_wrt_others'] for step in ep_data])
        ax.plot(all_pose[:,0], all_pose[:,1], c=ego_color[:3]+[traj_alpha])
    plt.legend(handles=[mpatches.Patch(color=ego_color[:3]+[0.3], label='ego car'),
                        mpatches.Patch(color=other_color[:3]+[0.3], label='front car')])
    # plt.show()
    fig.savefig('test.png') # DEBUG


def get_poly(pose, car_dim):
    x, y, theta = pose
    car_length, car_width = car_dim
    poly = Box(x-car_width/2., y-car_length/2., x+car_width/2., y+car_length/2.)
    poly = affinity.rotate(poly, np.degrees(theta))
    return poly


if __name__ == '__main__':
    main()
