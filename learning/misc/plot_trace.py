import os
import argparse
import numpy as np
import pickle5 as pickle
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

from vista.core.Trace import Trace
from vista.entities.agents.Dynamics import StateDynamics

from simple_metrics import append_poly_info, overwrite_with_new_overlap_threshold


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
    parser.add_argument(
        '--use-integration',
        action='store_true',
        default=False,
        help='Use integration to trace out trajectory.')
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

    # get trajectory
    trace = Trace(args.trace_path)
    ts = trace.masterClock.get_time_from_frame_num(trace.which_camera, 
        trace.syncedLabeledFrames[0][trace.which_camera][0])
    trajs = []
    if args.use_integration:
        dynamics = StateDynamics()
    else:
        odom = pd.read_csv(os.path.join(args.trace_path, 'odom.csv'))
        f_position_x = interp1d(odom.iloc[:,0], odom.iloc[:,1], fill_value='extrapolate')
        f_position_y = interp1d(odom.iloc[:,0], odom.iloc[:,2], fill_value='extrapolate')
    for segment_idx in range(len(trace.syncedLabeledFrames)):
        if args.use_integration:
            trajs.append([[dynamics.x_state, dynamics.y_state]])
        else:
            trajs.append([])
        for frame_idx in range(len(trace.syncedLabeledFrames[segment_idx][trace.which_camera])):
            if args.use_integration:
                next_ts = trace.syncedLabeledTimestamps[segment_idx][frame_idx]
                dynamics.step(curvature=trace.f_curvature(ts), 
                            velocity=trace.f_speed(ts),
                            delta_t=next_ts-ts)
                ts = next_ts
                trajs[-1].append([dynamics.x_state, dynamics.y_state])
            else:
                ts = trace.syncedLabeledTimestamps[segment_idx][frame_idx]
                trajs[-1].append([f_position_x(ts), f_position_y(ts)])

    args.exclude = []

    # ### HACKY
    if False:
        pre_ts = trace.syncedLabeledTimestamps[0][0]
        pre_trajs = []
        for _ in range(120):
            pre_ts = pre_ts - 0.05
            pre_trajs.append([f_position_x(pre_ts), f_position_y(pre_ts)])
        pre_trajs = pre_trajs[::-1]
        
        trajs[0] = trajs[0] + pre_trajs
        for _ in range(23):
            ts = ts + 0.05
            trajs[-1].append([f_position_x(ts), f_position_y(ts)])

        # trajs[-1].append([(trajs[0][0][0] + trajs[-1][-1][0])/2., (trajs[0][0][1] + trajs[-1][-1][1])/2.])

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

    append_poly_info(data, args.dilate)
    overwrite_with_new_overlap_threshold(data, args.threshold)

    if False: # crash rate
        # compute success rate for every frame index
        success_cnt = [[0. for _ in v] for v in trajs]
        failure_cnt = [[0. for _ in v] for v in trajs]
        for ep_data in data:
            success = ep_data[-1][-1][args.agent_id]['has_collided']
            # success = ep_data[-1][-1][args.agent_id]['passed_cars'] > 0 # DEBUG
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
                    success_rate[i][j] = 0 #-1

        # plot
        fig, ax = plt.subplots(1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Intervention', fontsize=22)
        fig.patch.set_visible(False)
        ax.axis('off')
        col = MplColorHelper(args.cmap, 0., 0.3)
        bg_color = (0.5, 0.5, 0.5)
        for traj, s_rate in zip(trajs, success_rate):
            traj = np.array(traj)
            color = [col.get_rgb(s) if s >= 0 else bg_color for s in s_rate]
            ax.scatter(traj[:,0], traj[:,1], c=color)
        cbar = plt.colorbar(col.scalarMap, ax=ax)
        cbar.ax.tick_params(labelsize=16) 
        fig.canvas.draw()
        fig.tight_layout()
        # fig.savefig('test.png') # DEBUG
        fig.savefig('intervention_loop.pdf') # DEBUG
    elif False: # deviation from lane center
        # compute max. deviation for every frame index
        max_dev_list = [[[] for _ in v] for v in trajs]
        for ep_data in data:
            for step_data in ep_data:
                info = step_data[-1]
                segment_idx = info[args.agent_id]['segment_index']
                frame_idx = info[args.agent_id]['frame_index']
                max_dev_list[segment_idx][frame_idx].append(np.abs(info[args.agent_id]['translation']))

        max_dev_mean = [[0. for _ in v] for v in trajs]
        for i in range(len(max_dev_mean)):
            for j in range(len(max_dev_mean[i])):
                v = max_dev_list[i][j]
                if len(v) > 0:
                    max_dev_mean[i][j] = np.mean(v)
                else:
                    max_dev_mean[i][j] = 0.

        # plot
        fig, ax = plt.subplots(1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Deviation From Center (m)', fontsize=22)
        # fig.patch.set_visible(False)
        ax.axis('off')
        col = MplColorHelper(args.cmap, 0., 0.7) #np.max([vv for v in max_dev_mean for vv in v]))
        bg_color = (0.5, 0.5, 0.5)
        for traj, max_dev in zip(trajs, max_dev_mean):
            traj = np.array(traj)
            color = [col.get_rgb(s) if s >= 0 else bg_color for s in max_dev]
            ax.scatter(traj[:,0], traj[:,1], c=color)
        cbar = plt.colorbar(col.scalarMap, ax=ax)
        cbar.ax.tick_params(labelsize=16) 
        fig.canvas.draw()
        fig.tight_layout()
        # fig.savefig('test.png') # DEBUG
        fig.savefig('max_dev_loop.pdf')
    else: # cara following
        # compute max. deviation for every frame index
        max_dev_list = [[[] for _ in v] for v in trajs]
        for ep_data in data:
            for step_data in ep_data:
                info = step_data[-1]
                segment_idx = info[args.agent_id]['segment_index']
                frame_idx = info[args.agent_id]['frame_index']
                max_dev_list[segment_idx][frame_idx].append(np.abs(info[args.agent_id]['following_lat_shift']))

        max_dev_mean = [[0. for _ in v] for v in trajs]
        for i in range(len(max_dev_mean)):
            for j in range(len(max_dev_mean[i])):
                v = max_dev_list[i][j]
                if len(v) > 0:
                    max_dev_mean[i][j] = np.mean(v)
                else:
                    max_dev_mean[i][j] = 0.

        # plot
        fig, ax = plt.subplots(1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Tracking Error (m)', fontsize=22)
        # fig.patch.set_visible(False)
        ax.axis('off')
        col = MplColorHelper(args.cmap, 0., np.max([vv for v in max_dev_mean for vv in v]))
        bg_color = (0.5, 0.5, 0.5)
        for traj, max_dev in zip(trajs, max_dev_mean):
            traj = np.array(traj)
            color = [col.get_rgb(s) if s >= 0 else bg_color for s in max_dev]
            ax.scatter(traj[:,0], traj[:,1], c=color)
        cbar = plt.colorbar(col.scalarMap, ax=ax)
        cbar.ax.tick_params(labelsize=16) 
        fig.canvas.draw()
        fig.tight_layout()
        # fig.savefig('test.png') # DEBUG
        fig.savefig('dev_track_loop.pdf')


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
