import argparse
import numpy as np
import matplotlib.pyplot as plt

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging
from vista.core.Display import fig2img

logging.setLevel(logging.ERROR)


def main(args):
    # Initialize the simulator
    trace_config = dict(
        road_width=4,
        reset_mode='default',
        master_sensor='front_center',
    )
    car_config = dict(
        length=5.,
        width=2.,
        wheel_base=2.78,
        steering_ratio=14.7,
        lookahead_road=True,
    )
    camera_config1 = dict(
        # camera params
        name='front_center',
        rig_path='~/data/traces/20200424-133758_blue_prius_cambridge_rain/RIG.xml',
        size=(200, 320), #(250, 400),
        # rendering params
        depth_mode=DepthModes.FIXED_PLANE,
        use_lighting=False,
    )
    display_config = dict(
        road_buffer_size=1000,
    )
    world = vista.World(args.trace_path, trace_config)
    agent = world.spawn_agent(car_config)
    camera1 = agent.spawn_camera(camera_config1)
    display = vista.Display(world, display_config=display_config)

    fig, ax = plt.subplots(1, 1)
    im = None

    # Main running loop
    while True:
        world.reset()
        display.reset()

        while not agent.done:
            action = np.array([agent.trace.f_curvature(agent.timestamp),
                               agent.trace.f_speed(agent.timestamp)])
            agent.step_dynamics(action)
            agent.step_sensors()

            print(agent.road.shape, agent.road[-1])
            road = agent.road
            if im is None:
                im, = ax.plot(road[:, 0], road[:, 1])
                sc = ax.scatter(*agent.ego_dynamics.numpy()[:2], c='r')
            else:
                im.set_data(road[:, 0], road[:, 1])
                sc.set_offsets(agent.ego_dynamics.numpy()[:2])
                ax.set_xlim(agent.road[:,0].min(), agent.road[:,0].max())
                ax.set_ylim(agent.road[:,1].min(), agent.road[:,1].max())
            fig.canvas.draw()
            road_img = fig2img(fig)

            img = display.render()
            ### DEBUG
            logging.warning('Dump image for debugging and set pdb')
            import cv2; 
            cv2.imwrite('test1.png', road_img[:,:,::-1])
            cv2.imwrite('test2.png', img[:,:,::-1])
            import pdb; pdb.set_trace()
            ### DEBUG


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Run the simulator with random actions')
    parser.add_argument(
        '--trace-path',
        type=str,
        nargs='+',
        help='Path to the traces to use for simulation')
    args = parser.parse_args()

    main(args)
