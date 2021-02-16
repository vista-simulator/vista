import copy
import cv2
import time

# vista classes
from .Dynamics import *
from ..Entity import *
from ...util import drawing as draw

# vista sub-modules
from .. import sensors
from ... import core

# Temporary -- move into a dedicated GUI class
WINDOW_SIZE = (500, 330, 3)
DRAWING_COORDS = {
    draw.Module.STEERING: draw.Box((.1, .75), (.3, 1)),
    draw.Module.LANE: draw.Box((.65, .7), (1, 1)),
    draw.Module.INFO: draw.Box((.6, 0), (1, .7)),
    draw.Module.SPEED: draw.Box((.3, .8), (.6, .95)),
    draw.Module.FRAME: draw.Box((0, 0), (.6, .7))
}

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

        # Graphical viewing
        self.viewer = None
        self.last_gui_update = time.time()

        # Additional attributes of the vehicle
        self.car_width = 2  # meters
        self.wheel_base = 2.8
        self.steering_ratio = 17.6
        self.reward = 0.0

        # Reset the car in the world, this will reset several properties:
        # 1. The trace in the world where the car is located
        # 2. The video stream of this trace
        self.reset()

    def spawn_camera(self):
        camera = sensors.Camera(attach_to=self)
        self.sensors.append(camera)
        return camera

    def step(self, action, delta_t=1 / 30.):
        # Force action to be column vector
        action = np.array(action).reshape(-1)

        curvature = action[0]
        velocity = action[1] if action.shape[0] > 1 else None

        # info = self.car.step(curvature, velocity, delta_t=1 / 30.)
        current_timestamp = self.get_current_timestamp()
        model_curvature = curvature + 1e-11  # Avoid dividing by 0
        model_velocity = self.trace.f_speed(current_timestamp) \
            if velocity is None else velocity

        self.ego_dynamics.step(model_curvature, model_velocity, delta_t)

        next_valid_timestamp, self.human_dynamics = \
            self.get_next_valid_timestamp(
                human=self.human_dynamics, desired_ego=self.ego_dynamics)


        translation_x, translation_y, theta = self.compute_relative_transform()
        self.relative_state.update(translation_x, translation_y, theta)

        # TODO: move out (?)...
        info = dict()
        info["timestamp"] = next_valid_timestamp
        info["first_time"] = self.first_time
        info["human_curvature"] = self.trace.f_curvature(next_valid_timestamp)
        info["human_velocity"] = self.trace.f_speed(next_valid_timestamp)
        info["model_curvature"] = model_curvature
        info["model_velocity"] = model_velocity
        info["model_angle"] = self.curvature_to_steering(model_curvature)
        info["rotation"] = self.relative_state.theta
        info["translation"] = self.relative_state.translation_x
        info["distance"] = self.trace.f_distance(next_valid_timestamp) - \
                           self.trace.f_distance(self.first_time)
        info["done"] = self.isCrashed

        observations = {}
        for sensor in self.sensors:
            observations[sensor.id] = sensor.capture(next_valid_timestamp)

        self.render(observations[self.sensors[0].id], info)

        done = self.isCrashed

        step_reward = 1.0 if not done else 0.0  # simple reward function +1 if not crashed
        self.reward += step_reward

        if self.trace_done:
            translated_frame = self.reset()

        return observations, step_reward, done, info


    def get_timestamp(self, index):

        if index >= len(self.trace.syncedLabeledTimestamps[
                self.current_segment_index]) - 1:
            print("END OF TRACE")
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

    def render(self, frame, info, mode='gui', FPS=30):
        if self.viewer is None:  # viewer is not yet created, so make the window
            cv2.namedWindow("VISTA", cv2.WINDOW_NORMAL)
            cv2.moveWindow("VISTA", 250, 0)
            cv2.resizeWindow("VISTA", WINDOW_SIZE[0], WINDOW_SIZE[1])
            self.viewer = True

        window = self.draw_gui(frame, info)
        cv2.imshow("VISTA", window)
        gui_delta_t = time.time() - self.last_gui_update
        timeout_ms = 1000 * max(1 / 1000., 1.0 / FPS - gui_delta_t)
        # print "timeout: {}".format(timeout_ms)
        if cv2.waitKey(int(timeout_ms)) == ord(' '):
            print("pause")
            cv2.waitKey()
        self.last_gui_update = time.time()

    def draw_gui(self, frame, info):
        """ Creates the GUI window for the simulator.

        This creates multiple modules that can be placed anywhere in the overall frame. Within each module, items can be
        placed in the reference frame of the specific module. These function are held in drawing.py

        Args:
            frame (np array): The altered frame from ViewSynthesis
            info (dict): the dictionary holding the current timestep's kinematic data

        Returns:
            (np array): the fuse of all the modules into the full GUI screen
        """
        frame = copy.copy(frame)

        # Create each module, feed them into frame
        coordinates_list = [
            DRAWING_COORDS[module] for module in [
                draw.Module.STEERING, draw.Module.LANE, draw.Module.INFO,
                draw.Module.SPEED, draw.Module.FRAME
            ]
        ]

        steering_module = draw.draw_steering_wheel(
            WINDOW_SIZE, DRAWING_COORDS[draw.Module.STEERING],
            info['model_angle'])

        mini_car_on_road_module = draw.draw_mini_car_on_road(
            WINDOW_SIZE, DRAWING_COORDS[draw.Module.LANE], info['translation'],
            info['rotation'], self.trace.road_width)

        info_module = draw.draw_info_sidebar(
            WINDOW_SIZE, DRAWING_COORDS[draw.Module.INFO], info)

        speed_module = draw.draw_speedometer(
            WINDOW_SIZE,
            DRAWING_COORDS[draw.Module.SPEED],
            model=2.23 * info['human_velocity'],
            human=2.23 * info['model_velocity'])

        frame_module = draw.draw_frame(
            WINDOW_SIZE, DRAWING_COORDS[draw.Module.FRAME], frame,
            self.sensors[0].camera, info['model_curvature'],
            info['human_curvature'])

        return draw.fuse(
            WINDOW_SIZE, [
                steering_module, mini_car_on_road_module, info_module,
                speed_module, frame_module
            ],
            coordinates_list,
            draw_outline=False)


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

    def __repr__(self, indent=2):
        tab = " " * indent

        repr = ["{}(id={}, ".format(self.__class__.__name__, self._id)]
        repr.append(f"{tab}parent={self._parent}, ")

        repr.append(f"{tab}sensors=[")
        for sensor in self.sensors:
            repr.append(f"{2*tab}{sensor}, ")
        repr.append(f"{tab}], ")

        return "\n".join(repr) + "\n)"
