import argparse
import numpy as np
import pickle5 as pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

from vista.core.Trace import Trace
from vista.entities.agents.Dynamics import StateDynamics


def main():
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'rollout_path',
        type=str,
        help='Path to rollout.')
    parser.add_argument(
        '--trace-path',
        type=str,
        required=True,
        help='Path to trace to be plotted.')
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
        '--cmap',
        type=str,
        default='viridis',
        help='Colormap')
    args = parser.parse_args()

    # trace out trajectory
    trace = Trace(args.trace_path)
    ts = trace.masterClock.get_time_from_frame_num(trace.which_camera, 
        trace.syncedLabeledFrames[0][trace.which_camera][0])
    trajs = []
    dynamics = StateDynamics()
    for segment_idx in range(len(trace.syncedLabeledFrames)):
        trajs.append([[dynamics.x_state, dynamics.y_state]])
        for frame_idx in range(len(trace.syncedLabeledFrames[segment_idx][trace.which_camera])):
            next_ts = trace.syncedLabeledTimestamps[segment_idx][frame_idx]
            dynamics.step(curvature=trace.f_curvature(ts), 
                          velocity=trace.f_speed(ts),
                          delta_t=next_ts-ts)
            ts = next_ts
            trajs[-1].append([dynamics.x_state, dynamics.y_state])

    # load results, filter out episodes with excluded terminal condition, and print meta-info
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

    # compute success rate for every frame index
    success_cnt = [[0. for _ in v] for v in trajs]
    failure_cnt = [[0. for _ in v] for v in trajs]
    for ep_data in data:
        success = ep_data[-1][-1][args.agent_id]['success']
        for step_data in ep_data:
            info = step_data[-1]
            segment_idx = info[args.agent_id]['segment_index']
            frame_idx = info[args.agent_id]['frame_index']
            if success:
                success_cnt[segment_idx][frame_idx] += 1
            else:
                failure_cnt[segment_idx][frame_idx] += 1

    success_rate = [[0. for _ in v] for v in trajs]
    for i in range(len(success_rate)):
        for j in range(len(success_rate[i])):
            total_cnt = success_cnt[i][j] + failure_cnt[i][j]
            if total_cnt > 0:
                success_rate[i][j] = success_cnt[i][j] / total_cnt
            else:
                success_rate[i][j] = -1

    # plot
    fig, ax = plt.subplots(1, 1)
    col = MplColorHelper(args.cmap, 0., 1.)
    bg_color = (0.5, 0.5, 0.5)
    for traj, s_rate in zip(trajs, success_rate):
        traj = np.array(traj)
        color = [col.get_rgb(s) if s > 0 else bg_color for s in s_rate]
        ax.scatter(traj[:,0], traj[:,1], c=color)
    plt.colorbar(col.scalarMap, ax=ax)
    fig.canvas.draw()
    fig.savefig('test.png') # DEBUG


class MplColorHelper:
  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)[:3]


if __name__ == '__main__':
    main()
