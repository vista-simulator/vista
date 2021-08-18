import os
import argparse
import numpy as np

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging

logging.setLevel(logging.DEBUG)


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
        wheel_base=2.8,
        steering_ratio=17.6,
    )
    event_cam_config = dict(
        name='event_camera_front',
        rig_path='~/data/traces/20200424-133758_blue_prius_cambridge_rain/RIG.xml',
        base_camera_name='front_center',
        depth_mode=DepthModes.FIXED_PLANE,
        use_lighting=False,
        size=(480, 640), #(200, 320),
        optical_flow_root='../data_prep/Super-SloMo',
        checkpoint='../data_prep/Super-SloMo/ckpt/SuperSloMo.ckpt',
        lambda_flow=0.5,
        max_sf=-1,
        use_gpu=True,
        positive_threshold=0.1,
        sigma_positive_threshold=0.02,
        negative_threshold=-0.1,
        sigma_negative_threshold=0.02,
    )
    display_config = dict(
        road_buffer_size=1000,
    )
    world = vista.World(args.trace_path, trace_config)
    agent = world.spawn_agent(car_config)
    event_cam = agent.spawn_event_camera(event_cam_config)
    display = vista.Display(world, display_config=display_config)

    if args.video_path:
        from skvideo.io import FFmpegWriter
        args.video_path = os.path.abspath(os.path.expanduser(args.video_path))
        video_writer = FFmpegWriter(args.video_path)

    # Main running loop
    while True:
        world.reset()
        display.reset()

        step = 0
        while not agent.done:
            dev = 0 # 0.01 * np.sin(step / 10.)
            action = np.array([agent.trace.f_curvature(agent.timestamp) + dev,
                               agent.trace.f_speed(agent.timestamp)])
            agent.step_dynamics(action)
            agent.step_sensors()

            img = display.render()
            if args.video_path:
                video_writer.writeFrame(img)

            step += 1
        import pdb ; pdb.set_trace()


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Run the simulator with random actions')
    parser.add_argument(
        '--trace-path',
        type=str,
        nargs='+',
        help='Path to the traces to use for simulation')
    parser.add_argument(
        '--video-path',
        type=str,
        default=None,
        help='Path to recorded video; Default as None for not saving')
    args = parser.parse_args()

    main(args)
