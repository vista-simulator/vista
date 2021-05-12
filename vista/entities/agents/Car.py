import time

# vista classes
from .Dynamics import *
from ..Entity import *

# vista sub-modules
from .. import sensors
from ... import core


class Car(Entity):
    def __init__(self, world):
        super(Car, self).__init__()

        # Pointer to the parent vista.World object where the agent lives
        self.world = world

        # A list of sensors attached to this agent (List[vista.Sensor]).
        self.sensors = []

        # State dynamics for tracking the virtual and human agent
        self.relative_state = State(translation_x=0.0,
                                    translation_y=0.0,
                                    theta=0.0)
        self.human_dynamics = StateDynamics()
        self.ego_dynamics = StateDynamics()

        # Additional attributes of the vehicle
        self.car_width = 2  # meters
        self.car_length = 4
        self.wheel_base = 2.8
        self.steering_ratio = 17.6

        # Reset the car in the world, this will reset several properties:
        # 1. The trace in the world where the car is located
        # 2. The video stream of this trace
        self.reset()

    def spawn_camera(self, rendering_config=None):
        camera = sensors.Camera(attach_to=self, rendering_config=rendering_config)
        self.sensors.append(camera)
        return camera

    def step(self, action, delta_t=1 / 30.):
        step_reward, done, info, next_valid_timestamp = self.step_dynamics(action, delta_t)
        observations = self.step_sensors(next_valid_timestamp)

        return observations, step_reward, done, info

    def step_dynamics(self, action, delta_t=1 / 30.):
        # Force action to be column vector
        action = np.array(action).reshape(-1)
        curvature = action[0]
        velocity = action[1] if action.shape[0] > 1 else None

        current_timestamp = self.get_current_timestamp()
        self.model_curvature = curvature + 1e-11  # Avoid dividing by 0
        self.model_velocity = self.trace.f_speed(current_timestamp) \
            if velocity is None else velocity

        # Step the ego-dynamics forward for some time
        self.ego_dynamics.step(self.model_curvature, self.model_velocity,
                               delta_t)

        # Find the closest point in the human trajectory and compute relative
        # transformation from the ego agent
        self.timestamp, self.human_dynamics = self.get_next_valid_timestamp(
            human=self.human_dynamics, desired_ego=self.ego_dynamics)

        translation_x, translation_y, theta = self.compute_relative_transform()
        self.relative_state.update(translation_x, translation_y, theta)
        self.distance = self.trace.f_distance(self.timestamp) - \
                        self.trace.f_distance(self.first_time)

        # info is used in some envs
        info = dict()
        info["timestamp"] = self.timestamp
        info["first_time"] = self.first_time
        info["human_curvature"] = self.trace.f_curvature(self.timestamp)
        info["human_velocity"] = self.trace.f_speed(self.timestamp)
        info["model_curvature"] = self.model_curvature
        info["model_velocity"] = self.model_velocity
        info["model_angle"] = self.curvature_to_steering(self.model_curvature)
        info["rotation"] = self.relative_state.theta
        info["translation"] = self.relative_state.translation_x
        info["distance"] = self.trace.f_distance(self.timestamp) - \
                           self.trace.f_distance(self.first_time)
        info["done"] = self.isCrashed

        done = self.isCrashed or self.trace_done

        step_reward = 1.0 if not done else 0.0  # simple reward function +1 if not crashed

        return step_reward, done, info, self.timestamp

    def step_sensors(self, next_valid_timestamp, other_agents=[]):
        observations = {}
        for sensor in self.sensors:
            observations[sensor.id] = sensor.capture(next_valid_timestamp, other_agents=other_agents)

        # NOTE: this implementation will cause issue if agent has memory. Also it accumulates reward
        # accross episode. Need a way to connect trace ending and starting point.
        if self.trace_done:
            pass # otherwise will cause incorrect computation of scene state | translated_frame = self.reset()
        return observations

    def get_timestamp(self, index):

        if index >= len(self.trace.syncedLabeledTimestamps[
                self.current_segment_index]) - 1:
            # print("END OF TRACE")
            self.trace_done = True  # Done var will be set to True in deepknight env step

            # Return last timestamp
            return self.trace.syncedLabeledTimestamps[
                self.current_segment_index][-1]

        return self.trace.syncedLabeledTimestamps[
            self.current_segment_index][index]

    def get_current_timestamp(self):
        return self.get_timestamp(self.current_frame_index)

    def get_next_valid_timestamp(self, human, desired_ego):
        first_time = self.first_time
        current_time = self.get_current_timestamp()
        index = self.current_frame_index
        time = self.get_timestamp(index)
        human = human.copy()  # dont edit the original
        closest_dist = float('inf')
        while True:
            next_time = self.get_timestamp(index)

            last_human = human.copy()
            human.step(curvature=self.trace.f_curvature(time),
                       velocity=self.trace.f_speed(time),
                       delta_t=next_time - time)

            dist = np.linalg.norm(human.numpy()[:2] - desired_ego.numpy()[:2])
            time = next_time
            if dist < closest_dist:
                closest_dist = dist
                index += 1
            else:
                break

        self.current_frame_index = index - 1
        closest_time = self.get_current_timestamp()
        return closest_time, last_human

    def compute_relative_transform(self):
        """ TODO
        """

        ego_x_state, ego_y_state, ego_theta_state = \
            self.ego_dynamics.numpy()
        human_x_state, human_y_state, human_theta_state = \
            self.human_dynamics.numpy()

        c = np.cos(human_theta_state)
        s = np.sin(human_theta_state)
        R_2 = np.array([[c, -s], [s, c]])
        xy_global_centered = np.array([[ego_x_state - human_x_state],
                                       [human_y_state - ego_y_state]])
        [[translation_x], [translation_y]] = np.matmul(R_2, xy_global_centered)
        translation_y *= -1  # negate the longitudinal translation (due to VS setup)

        # Adjust based on what the human did
        theta = ego_theta_state - human_theta_state

        # Check if crashed
        free_width = self.trace.road_width - self.car_width
        max_translation = abs(translation_x) > free_width / 2.
        max_rotation = abs(theta) > np.pi / 10.
        if max_translation or max_rotation:
            self.isCrashed = True

        return translation_x, translation_y, theta

    def curvature_to_steering(self, curvature):
        tire_angle = np.arctan(self.wheel_base * curvature)
        angle = tire_angle * self.steering_ratio * 180. / np.pi
        return angle

    def reset(self):
        self.relative_state.reset()
        self.ego_dynamics.reset()
        self.human_dynamics.reset()

        (self.current_trace_index, self.current_segment_index, \
            self.current_frame_index) = self.world.sample_new_location()

        # First timestamp of trace
        self.trace = self.world.traces[self.current_trace_index]
        self.first_time = self.trace.masterClock.get_time_from_frame_num(
            self.trace.which_camera,
            self.trace.syncedLabeledFrames[self.current_trace_index][
                self.trace.which_camera][self.current_frame_index])  # MODIFIED

        self.trace_done = False
        self.isCrashed = False

        for sensor in self.sensors:
            sensor.reset()

        observations = {}
        for sensor in self.sensors:
            observations[sensor.id] = sensor.capture(self.first_time)
        return observations

    def __repr__(self, indent=2):
        tab = " " * indent

        repr = ["{}(id={}, ".format(self.__class__.__name__, self._id)]
        repr.append(f"{tab}parent={self._parent}, ")

        repr.append(f"{tab}sensors=[")
        for sensor in self.sensors:
            repr.append(f"{2*tab}{sensor}, ")
        repr.append(f"{tab}], ")

        return "\n".join(repr) + "\n)"
