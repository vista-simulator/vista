from typing import Dict, Any
from collections import deque
import numpy as np
import simple_pid

from vista.entities.agents.Car import Car
from vista.utils import transform
from vista.entities.agents.Dynamics import tireangle2curvature


def get_controller(config):
    return globals()[config['type']](config)


class BaseController:
    def __init__(self, config: Dict[str, Any], **kwargs):
        self._config = config

    def __call__(self, agent: Car):
        raise NotImplementedError

    @property
    def config(self) -> Dict[str, Any]:
        return self._config


class PurePursuit(BaseController):
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)
        self._ts = deque(maxlen=2)

    def __call__(self, agent: Car):
        lookahead_dist = self.config['lookahead_dist']
        Kp = self.config['Kp']

        if len(self._ts) < 2:
            dt = 1 / 30.
        else:
            dt = self._ts[1] - self._ts[0]
            self._ts.append(agent.timestamp)

        speed = agent.human_speed

        road = agent.road
        ego_pose = agent.ego_dynamics.numpy()[:3]
        road_in_ego = np.array(
            [  # TODO: vectorize this: slow if road buffer size too large
                transform.compute_relative_latlongyaw(_v, ego_pose)
                for _v in road
            ])

        dist = np.linalg.norm(road_in_ego[:, :2], axis=1)
        dist[road_in_ego[:, 1] < 0] = 9999.  # drop road in the back
        tgt_idx = np.argmin(np.abs(dist - lookahead_dist))
        dx, dy, dyaw = road_in_ego[tgt_idx]

        arc_len = speed * dt
        curvature = (Kp * np.arctan2(-dx, dy) * dt) / arc_len
        curvature_bound = [
            tireangle2curvature(_v, agent.wheel_base)
            for _v in agent.ego_dynamics.steering_bound
        ]
        curvature = np.clip(curvature, *curvature_bound)

        return curvature, speed


class PID(BaseController):
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)
        self._ts = deque(maxlen=2)

        self._lat_pid = simple_pid.PID(config['lateral']['Kp'],
                                       config['lateral']['Ki'],
                                       config['lateral']['Kd'],
                                       setpoint=0.)
        if 'longitudinal' in config.keys():
            self._long_pid = simple_pid.PID(config['longitudinal']['Kp'],
                                            config['longitudinal']['Ki'],
                                            config['longitudinal']['Kd'],
                                            setpoint=0.)

    def __call__(self, agent: Car):
        # Lateral PID (with lookahead)
        lookahead_dist = self.config['lateral']['lookahead_dist']

        if len(self._ts) < 2:
            dt = 1 / 30.
        else:
            dt = self._ts[1] - self._ts[0]
            self._ts.append(agent.timestamp)

        road = agent.road
        ego_pose = agent.ego_dynamics.numpy()[:3]
        road_in_ego = np.array(
            [  # TODO: vectorize this: slow if road buffer size too large
                transform.compute_relative_latlongyaw(_v, ego_pose)
                for _v in road
            ])

        dist = np.linalg.norm(road_in_ego[:, :2], axis=1)
        dist[road_in_ego[:, 1] < 0] = 9999.  # drop road in the back
        tgt_idx = np.argmin(np.abs(dist - lookahead_dist))
        lat_dx, lat_dy, lat_dyaw = road_in_ego[tgt_idx]

        heading_err_weight = self.config['lateral'].get(
            'heading_err_weight', 0.0)
        heading_err_tol = self.config['lateral'].get('heading_err_tol', 0.0)
        heading_err = 0. if abs(agent.relative_state.yaw) <= heading_err_tol \
            else agent.relative_state.yaw
        error = lat_dx + heading_err_weight * heading_err
        curvature_bound = [
            tireangle2curvature(_v, agent.wheel_base)
            for _v in agent.ego_dynamics.steering_bound
        ]
        self._lat_pid.output_limits = tuple(curvature_bound)
        self._lat_pid.sample_time = dt
        curvature = self._lat_pid(error)

        # Longtitudinal PID
        if 'longitudinal' in self.config.keys():
            self._long_pid.output_limits = agent.ego_dynamics.speed_bound
            long_dy = road_in_ego[1, 1]
            speed = self._long_pid(-long_dy)
        else:
            speed = agent.human_speed

        return curvature, speed
