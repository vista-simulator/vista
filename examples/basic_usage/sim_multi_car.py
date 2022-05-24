import argparse
import numpy as np
import os
import copy
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import box as Box
from shapely import affinity

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging
from vista.tasks import MultiAgentBase
from vista.utils import transform


def main(args):
    # Initialize the simulator
    trace_config = dict(
        road_width=4,
        reset_mode='default',
        master_sensor='camera_front',
    )
    car_config = dict(
        length=5.,
        width=2.,
        wheel_base=2.78,
        steering_ratio=14.7,
        lookahead_road=True,
    )
    examples_path = os.path.dirname(os.path.realpath(__file__))
    sensors_config = [
        dict(
            type='camera',
            # camera params
            name='camera_front',
            size=(200, 320),
            # rendering params
            depth_mode=DepthModes.FIXED_PLANE,
            use_lighting=False,
        )
    ]
    task_config = dict(n_agents=2,
                       mesh_dir=args.mesh_dir,
                       init_dist_range=[6., 10.],
                       init_lat_noise_range=[-1., 1.])
    display_config = dict(road_buffer_size=1000, )

    ego_car_config = copy.deepcopy(car_config)
    ego_car_config['lookahead_road'] = True
    env = MultiAgentBase(trace_paths=args.trace_paths,
                         trace_config=trace_config,
                         car_configs=[car_config] * task_config['n_agents'],
                         sensors_configs=[sensors_config] + [[]] *
                         (task_config['n_agents'] - 1),
                         task_config=task_config,
                         logging_level='DEBUG')

    # Run
    env.reset()
    if args.use_display:
        display = vista.Display(env.world, display_config=display_config)
        display.reset()  # reset should be called after env reset
    if args.visualize_privileged_info:
        fig, axes = plt.subplots(1, task_config['n_agents'])
        for ai, agent in enumerate(env.world.agents):
            axes[ai].set_title(f'Agent ({agent.id})')
        artists = dict()
        fig.tight_layout()
        fig.show()
    done = False
    while not done:
        # follow nominal trajectories for all agents
        actions = generate_human_actions(env.world)

        # step environment
        observations, rewards, dones, infos = env.step(actions)
        done = np.any(list(dones.values()))

        # fetch priviliged information (road, all cars' states)
        privileged_info = dict()
        for agent in env.world.agents:
            privileged_info[agent.id] = fetch_privileged_info(env.world, agent)

        if args.visualize_privileged_info:
            for ai, (aid, pinfo) in enumerate(privileged_info.items()):
                agent = [_a for _a in env.world.agents if _a.id == aid][0]

                update_road_vis(pinfo[0], axes[ai], artists, f'{aid}:road')

                other_car_dims = [(_a.width, _a.length)
                                  for _a in env.world.agents
                                  if _a.id != agent.id]
                ego_car_dim = (agent.width, agent.length)
                update_car_vis(pinfo[1], other_car_dims, ego_car_dim, axes[ai],
                               artists, f'{aid}:ado_car')

                print(aid, pinfo[1])

            fig.canvas.draw()
            if not args.use_display:
                plt.pause(0.03)

        # vista visualization
        if args.use_display:
            img = display.render()
            cv2.imshow("test", img[:, :, ::-1])
            key = cv2.waitKey(20)
            if key == ord('q'):
                break


def state2poly(state, car_dim):
    """ Convert vehicle state to polygon """
    poly = Box(state[0] - car_dim[0] / 2., state[1] - car_dim[1] / 2.,
               state[0] + car_dim[0] / 2., state[1] + car_dim[1] / 2.)
    poly = affinity.rotate(poly, np.degrees(state[2]))
    return poly


def update_car_vis(other_states, other_car_dims, ego_car_dim, ax, artists,
                   name_prefix):
    # clear car visualization at previous timestamp
    for existing_name in artists.keys():
        if name_prefix in existing_name:
            artists[existing_name].remove()

    # initialize some helper object
    colors = list(cm.get_cmap('Set1').colors)
    poly_i = 0

    # plot ego car (reference pose; always at the center)
    ego_poly = state2poly([0., 0., 0.], ego_car_dim)
    artists[f'{name_prefix}_{poly_i:0d}'], = ax.plot(
        ego_poly.exterior.coords.xy[0],
        ego_poly.exterior.coords.xy[1],
        c=colors[poly_i],
    )
    poly_i += 1

    # plot ado cars
    for other_state, other_car_dim in zip(other_states, other_car_dims):
        other_poly = state2poly(other_state, other_car_dim)
        artists[f'{name_prefix}_{poly_i:0d}'], = ax.plot(
            other_poly.exterior.coords.xy[0],
            other_poly.exterior.coords.xy[1],
            c=colors[poly_i],
        )
        poly_i += 1


def update_road_vis(road, ax, artists, name):
    if name in artists.keys():
        artists[name].remove()
    artists[name], = ax.plot(road[:, 0],
                             road[:, 1],
                             c='k',
                             linewidth=2,
                             linestyle='dashed')
    ax.set_xlim(-10., 10.)
    ax.set_ylim(-20., 20.)


def fetch_privileged_info(world, agent):
    # Get ado cars state w.r.t. agent
    other_agents = [_a for _a in world.agents if _a.id != agent.id]
    other_states = []
    for other_agent in other_agents:
        other_latlongyaw = transform.compute_relative_latlongyaw(
            other_agent.ego_dynamics.numpy()[:3],
            agent.ego_dynamics.numpy()[:3])
        other_states.append(other_latlongyaw)

    # Get road w.r.t. the agent
    road = np.array(agent.road)[:, :3].copy()
    ref_pose = agent.ego_dynamics.numpy()[:3]
    road_in_agent = np.array(
        [transform.compute_relative_latlongyaw(_v, ref_pose) for _v in road])

    return road_in_agent, other_states


def generate_human_actions(world):
    actions = dict()
    for agent in world.agents:
        actions[agent.id] = np.array([
            agent.trace.f_curvature(agent.timestamp),
            agent.trace.f_speed(agent.timestamp)
        ])
    return actions


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Run VISTA with multiple cars')
    parser.add_argument('--trace-paths',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Path to the traces to use for simulation')
    parser.add_argument('--mesh-dir',
                        type=str,
                        default=None,
                        help='Directory of meshes for virtual agents')
    parser.add_argument('--use-display',
                        action='store_true',
                        default=False,
                        help='Use VISTA default display')
    parser.add_argument('--visualize-privileged-info',
                        action='store_true',
                        default=False,
                        help='Visualize privileged information')
    args = parser.parse_args()

    main(args)
