.. _getting_started-basic-usage:

Basic Usage
===========

Simple Controller With Simulator State
--------------------------------------

We first start with simulating car motion only without synthesizing
sensory measurements, as an example of accessing vehicle state for control.
We start with defining a world in VISTA, adding a car, and attaching display
to the world for visualization. ::

    import vista

    world = vista.World(trace_path,
                        trace_config={'road_width': 4})
    car = world.spawn_agent(config={'length': 5.,
                                    'width': 2.,
                                    'wheel_base': 2.78,
                                    'steering_ratio': 14.7,
                                    'lookahead_road': True})
    display = vista.Display(world)

There are more parameters that can be set in ``trace_config`` and ``car_config``,
which can be seen in the :doc:`API documentation <../api_documentation/core>`. Note
that ``lookahead_road = True`` make ``car`` to keep a cache of the position of road
within a certain distance centered at the car.

Then we reset the environment and control the car to move in the simulator. ::

    world.reset()
    display.reset()

    while not car.done:
        action = state_space_controller(car)
        car.step_dynamics(action)

        vis_img = display.render()

The ``state_space_controller`` function defines a controller that takes simulator
states as input and produces an action for the agent. In the following, we show several
examples of how we can control the car to do lane stable control with the access to Simulator
states. The simplest controller is simply following how humans drive in the dataset. ::

    def follow_human_trajectory(agent):
        action = np.array([
            agent.trace.f_curvature(agent.timestamp),
            agent.trace.f_speed(agent.timestamp)
        ])
        return action

We can also implement a simple pure pursuit controller for steering command (curvature). ::

    from vista.utils import transform
    from vista.entities.agents.Dynamics import tireangle2curvature

    def pure_pursuit_controller(agent):
        # hyperparameters
        lookahead_dist = 5.
        Kp = 3.
        dt = 1 / 30.

        # get road in ego-car coordinates
        ego_pose = agent.ego_dynamics.numpy()[:3]
        road_in_ego = np.array([
            transform.compute_relative_latlongyaw(_v[:3], ego_pose)
            for _v in agent.road
        ])

        # find (lookahead) target
        dist = np.linalg.norm(road_in_ego[:,:2], axis=1)
        dist[road_in_ego[:,1] < 0] = 9999. # drop road in the back
        tgt_idx = np.argmin(np.abs(dist - lookahead_dist))
        dx, dy, dyaw = road_in_ego[tgt_idx]

        # simply follow human trajectory for speed
        speed = agent.human_speed

        # compute curvature
        arc_len = speed * dt
        curvature = (Kp * np.arctan2(-dx, dy) * dt) / arc_len
        curvature_bound = [
            tireangle2curvature(_v, agent.wheel_base)
            for _v in agent.ego_dynamics.steering_bound]
        curvature = np.clip(curvature, *curvature_bound)

        return np.array([curvature, speed])

Note that we should replace ``state_space_controller`` with either of the above
controller (or other custom controller) and you will see a car doing lane following
on the road in the dataset.

Synthesizing Sensory Measurements
---------------------------------

Then, we show how to attach sensors to a car and synthesize sensory
measurements at every timestmap. Currently, RGB camera, LiDAR, and event
camera are supported. Following the setup above, we simply spawn sensors
on the ``car`` object.

Spawning a RGB camera. ::

    camera_config = {'name': 'camera_front',
                     'rig_path': './RIG.xml',
                     'size': (200, 320)}
    camera = car.spawn_camera(camera_config)

Spawning a LiDAR. ::

    lidar_config = {'name': 'lidar_3d',
                    'yaw_res': 0.1,
                    'pitch_res': 0.1,
                    'yaw_fov': (-180., 180.)}
    lidar = car.spawn_lidar(lidar_config)

Spawning an event-based camera. ::

    event_camera_config = {'name': 'event_camera_front',
                           'rig_path': './RIG.xml',
                           'original_size': (480, 640),
                           'size': (240, 320),
                           'optical_flow_root': '../data_prep/Super-SloMo',
                           'checkpoint': '../data_prep/Super-SloMo/ckpt/SuperSloMo.ckpt',
                           'positive_threshold': 0.1,
                           'sigma_positive_threshold': 0.02,
                           'negative_threshold': -0.1,
                           'sigma_negative_threshold': 0.02,
                           'base_camera_name': 'camera_front',
                           'base_size': (600, 960)}
    event_camera = car.spawn_event_camera(event_camera_config)

Remember to check if all paths are valid, e.g., ``rig_path``, ``optical_flow_root``, and
``checkpoint``. We can then start the simulation. ::

    world.reset()
    display.reset()

    while not car.done:
        action = state_space_controller(car)
        car.step_dynamics(action)
        car.step_sensors()

        sensor_data = car.observations

        vis_img = display.render()

The ``sensor_data`` is a dictionary with keys as names of sensors and values as the synthesized
sensor data.

Adding Virtual Cars
-------------------

We can also add more cars (or potentially other objects) to the simulation, where
each car can have different sets of sensors. Note that currently only RGB camera
supports rendering for virtual objects in the scene. ::

    world = vista.World(trace_path)
    car1 = world.spawn_agent()
    car1.spawn_camera()

    car2 = world.spawn_agent()
    car2.spawn_lidar()

It spawns two cars, one with RGB camera and the other with LiDAR. Note that, for now, since only
RGB camera supports rendering of virtual objects, ``car2`` cannot see ``car1``
with its LiDAR measurement. Note that there are still two major functions to be implemented
to make it a reasonable simulation, including initialization of virtual objects and collision
check/dynamics across objects. For more details, check ``vista/task/multi_agent_base.py``.

..
    Configurations
    --------------

    Here we list a set of configurations that might be useful.
