import os
import sys
import argparse
import numpy as np
import cv2

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging
from vista.core.Display import events2frame

logging.setLevel(logging.DEBUG)


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
        wheel_base=2.8,
        steering_ratio=17.6,
        lookahead_road=True,
        road_buffer_size=50,
    )
    event_cam_config = dict(
        name='event_camera_front',
        rig_path='../examples/RIG.xml',
        base_camera_name='camera_front',
        base_size=(600, 960),
        depth_mode=DepthModes.FIXED_PLANE,
        use_lighting=False,
        size=(300, 480),
        optical_flow_root='../data_prep/Super-SloMo',
        checkpoint='../data_prep/Super-SloMo/ckpt/SuperSloMo.ckpt',
        lambda_flow=0.5,
        max_sf=16,
        use_gpu=True,
        positive_threshold=0.1,
        sigma_positive_threshold=0.02,
        negative_threshold=-0.1,
        sigma_negative_threshold=0.02,
        reproject_pixel=False,
    )
    # set mode
    mode = args.mode
    if mode == 'rgb':
        event_cam_config['name'] = 'camera_front'
        event_cam_config['base_camera_name'] = 'camera_front'
        event_cam_config['size'] = (600, 960)
    elif mode in ['rgb_in_event', 'flow', 'event', 'event_no_random']:
        event_cam_config['name'] = 'event_camera_front'
        event_cam_config['base_camera_name'] = 'camera_front'
        if mode == 'rgb_in_event':
            event_cam_config['size'] = (480, 640)
        else:
            event_cam_config['size'] = (240, 320)
        if mode == 'event_no_random':
            event_cam_config['sigma_positive_threshold'] = 0.0
            event_cam_config['sigma_negative_threshold'] = 0.0

        if mode == 'flow':
            sys.path.append(os.path.abspath(event_cam_config['optical_flow_root']))
            from async_slomo import flow2img
    else:
        raise NotImplementedError
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
        video_dir = os.path.dirname(args.video_path)
        if not os.path.isdir(video_dir):
            os.makedirs(video_dir)
        fps = 30
        video_writer = FFmpegWriter(args.video_path, inputdict={'-r': str(fps)},
                                                     outputdict={'-r': str(fps),
                                                                 '-c:v': 'libx264',
                                                                 '-pix_fmt': 'yuv420p'})

    sys.path.append(os.path.abspath('../gpl/datasets'))
    from privileged_controller import get_controller
    controller = get_controller(
        {
            'type': 'PurePursuit',
            'lookahead_dist': 5,
            'Kp': 0.5,

            # 'type': 'PID',
            # 'lateral': {
            #     'lookahead_dist': 5,
            #     'Kp': 0.005,
            #     'Ki': 0.00001,
            #     'Kd': 0.0001,
            # },
            # 'heading_err_weight': 1.,
        }
    )
    use_controller_switch = False
    controller_switch = 0
    controller_step = 0
    controller_step_sign = 1

    # Main running loop
    while True:
        world.reset()
        display.reset()

        step = 0
        while not agent.done:
            try:
                if use_controller_switch:
                    if controller_switch == 1:
                        curvature, _ = controller(agent)
                        if abs(agent.relative_state.x) < 0.05:
                            controller_switch = 0
                            controller_step = 0
                            controller_step_sign *= -1
                    else:
                        dev = 0.003 * controller_step_sign * np.sin(controller_step / 10.)
                        curvature = agent.trace.f_curvature(agent.timestamp) + dev
                        controller_step += 1
                        if abs(agent.relative_state.x) > 0.5:
                            controller_switch = 1
                else:
                    curvature = agent.human_curvature
                action = np.array([curvature,
                                agent.trace.f_speed(agent.timestamp)])
                agent.step_dynamics(action)
                agent.step_sensors()

                if mode in ['rgb', 'rgb_in_event']:
                    img = event_cam.prev_frame
                elif mode == 'flow':
                    assert hasattr(event_cam, 'flow')
                    flow_imgs = list(map(lambda _x: flow2img(_x)[0], event_cam.flow))
                    img = flow_imgs[0] # only forward flow
                elif mode in ['event', 'event_no_random']:
                    events = agent.observations['event_camera_front']
                    event_cam_param = event_cam.camera_param
                    img = events2frame(events, event_cam_param.get_height(), 
                                    event_cam_param.get_width())[:,:,::-1]
                else:
                    raise NotImplementedError
                if args.video_path:
                    video_writer.writeFrame(img)

                # ### DEBUG
                # print(step)
                # if step >= 3600: #5400:
                #     import pdb; pdb.set_trace()
                # ### DEBUG

                step += 1
            except:
                video_writer.close()
                print('stop for some reason')
                import pdb; pdb.set_trace()


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
    parser.add_argument(
        '--mode',
        type=str,
        default='event',
        choices=['rgb', 'rgb_in_event', 'flow', 'event', 'event_no_random'],
        help='Determine what to be extracted to the video')
    args = parser.parse_args()

    main(args)
