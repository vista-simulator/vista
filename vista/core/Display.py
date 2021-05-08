import copy
import cv2
import time

from ..util import drawing as draw

HEADER = "VISTA"
WINDOW_SIZE = (500, 330, 3)
DRAWING_COORDS = {
    draw.Module.STEERING: draw.Box((.1, .75), (.3, 1)),
    draw.Module.LANE: draw.Box((.65, .7), (1, 1)),
    draw.Module.INFO: draw.Box((.6, 0), (1, .7)),
    draw.Module.SPEED: draw.Box((.3, .8), (.6, .95)),
    draw.Module.FRAME: draw.Box((0, 0), (.6, .7))
}


class Display:
    def __init__(self, world, fps=30):

        # Display arguments
        self.world = world
        self.fps = fps

        # Graphical viewing
        self.viewer = None
        self.last_gui_update = time.time()

    def render(self, headless=False):
        if not headless and self.viewer is None:  # viewer is not yet created, so make the window
            for agent in self.world.agents:
                gui_name = f"{HEADER} {agent.id}"
                cv2.namedWindow(gui_name, cv2.WINDOW_NORMAL)
                cv2.moveWindow(gui_name, 250, 0)
                cv2.resizeWindow(gui_name, WINDOW_SIZE[0], WINDOW_SIZE[1])
            self.viewer = True

        for agent in self.world.agents:
            gui = self.__draw_gui(agent)
            if not headless:
                cv2.imshow(f"{HEADER} {agent.id}", gui)

        if not headless:
            gui_delta_t = time.time() - self.last_gui_update
            timeout_ms = 1000 * max(1 / 1000., 1.0 / self.fps - gui_delta_t)
            if cv2.waitKey(int(timeout_ms)) == ord(' '):  # pause if space clicked
                cv2.waitKey()

        self.last_gui_update = time.time()

        return gui

    def __draw_gui(self, agent):
        """ Creates the GUI window for the simulator.

        This creates multiple modules that can be placed anywhere in the overall frame. Within each module, items can be
        placed in the reference frame of the specific module. These function are held in drawing.py

        Args:
            frame (np array): The altered frame from ViewSynthesis
            info (dict): the dictionary holding the current timestep's kinematic data

        Returns:
            (np array): the fuse of all the modules into the full GUI screen
        """

        sensor = agent.sensors[0]
        frame = copy.copy(agent.observations[sensor.id])
        model_curvature = agent.model_curvature
        model_velocity = agent.model_velocity
        model_angle = agent.curvature_to_steering(model_curvature)
        translation = agent.relative_state.translation_x
        rotation = agent.relative_state.theta
        timestamp = agent.timestamp
        distance = agent.distance
        human_velocity = agent.trace.f_speed(timestamp)
        human_curvature = agent.trace.f_curvature(timestamp)

        # Create each module, feed them into frame
        coordinates_list = [
            DRAWING_COORDS[module] for module in [
                draw.Module.STEERING, draw.Module.LANE, draw.Module.INFO,
                draw.Module.SPEED, draw.Module.FRAME
            ]
        ]

        steering_module = draw.draw_steering_wheel(
            WINDOW_SIZE, DRAWING_COORDS[draw.Module.STEERING], model_angle)

        mini_car_on_road_module = draw.draw_mini_car_on_road(
            WINDOW_SIZE, DRAWING_COORDS[draw.Module.LANE], translation,
            rotation, agent.trace.road_width)

        info_module = draw.draw_info_sidebar(
            WINDOW_SIZE, DRAWING_COORDS[draw.Module.INFO], timestamp,
            agent.first_time, model_angle, human_curvature, model_curvature,
            distance, model_velocity, translation, rotation)

        speed_module = draw.draw_speedometer(WINDOW_SIZE,
                                             DRAWING_COORDS[draw.Module.SPEED],
                                             model=2.23 * model_velocity,
                                             human=2.23 * human_velocity)

        frame_module = draw.draw_frame(WINDOW_SIZE,
                                       DRAWING_COORDS[draw.Module.FRAME],
                                       frame, sensor.camera, model_curvature,
                                       human_curvature)

        modules = [
            steering_module, mini_car_on_road_module, info_module,
            speed_module, frame_module
        ]

        return draw.fuse(WINDOW_SIZE,
                         modules,
                         coordinates_list,
                         draw_outline=False)
