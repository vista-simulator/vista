#!/usr/bin/env python

import copy
from threading import Lock

import argparse
import numpy as np
import rospy
import vista
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
# from vista.entities.agents.Dynamics import tireangle2curvature


class VistaSim(object):
    def __init__(self, args):
        self.lock = Lock()

        trace_config = {'road_width': 4,
                        'vis_full_frame': True,
                        }
        self.world = vista.World(args.trace_path, trace_config)
        self.wheel_base = 2.78
        self.car = self.world.spawn_agent(
            config={
                'length': 5.,
                'width': 2.,
                'wheel_base': self.wheel_base,
                'steering_ratio': 14.7,
                'lookahead_road': True
            })

        self.car.spawn_camera(config={
            'size': (300, 400),
        })
        self.display = vista.Display(self.world)

        self.world.reset()
        self.display.reset()

        self.cv_bridge = CvBridge()
        self.viz_pub = rospy.Publisher("viz", Image, queue_size=3)
        self.camera_pub = rospy.Publisher("image", Image, queue_size=3)

        self.speed = 0.0
        # self.tire_angle = 0.0
        self.curvature = 0.0

        self.twist_sub = rospy.Subscriber("cmd_vel", Twist, self.twist_callback, queue_size=5)

        update_rate = 30.0
        dt = rospy.Duration(1.0 / update_rate)
        # timer doesn't work because it will be in different thread than above vista objects
        # self.timer = rospy.Timer(dt, self.update)

        rate = rospy.Rate(update_rate)
        while not rospy.is_shutdown():
            stamp = rospy.Time.now()
            event = rospy.timer.TimerEvent(stamp, stamp, stamp, stamp, dt)
            self.update(event)
            rate.sleep()

    def twist_callback(self, msg):
        with self.lock:
            self.speed = msg.linear.x
            # TODO(lucasw) is this right?
            self.curvature = msg.angular.z

    def update(self, event):
        with self.lock:
            speed = copy.deepcopy(self.speed)
            curvature = copy.deepcopy(self.curvature)

        stamp = event.current_expected

        # curvature = tireangle2curvature(self.tire_angle, self.wheel_base)
        # rospy.loginfo_throttle(2.0, f"{self.tire_angle:0.3f} -> {curvature:0.3f}")

        # action = follow_human_trajectory(self.car)
        action = np.array([curvature, speed])
        self.car.step_dynamics(action)
        # TODO(lucasw) what is the time unit of this?
        self.car.step_sensors()

        sensor_data = self.car.observations
        image_msg = self.cv_bridge.cv2_to_imgmsg(sensor_data['camera_front'], encoding='bgr8')
        image_msg.header.frame_id = 'camera_front'
        image_msg.header.stamp = stamp
        self.camera_pub.publish(image_msg)

        viz_img = self.display.render()
        viz_msg = self.cv_bridge.cv2_to_imgmsg(viz_img, encoding='rgb8')
        viz_msg.header.frame_id = 'viz'
        viz_msg.header.stamp = stamp
        self.viz_pub.publish(viz_msg)


def follow_human_trajectory(agent):
    action = np.array([
        agent.trace.f_curvature(agent.timestamp),
        agent.trace.f_speed(agent.timestamp)
    ])
    return action


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Run the simulator with random actions')
    parser.add_argument('--trace-path',
                        type=str,
                        nargs='+',
                        help='Path to the traces to use for simulation')
    args = parser.parse_args()

    rospy.init_node('vista_sim')
    node = VistaSim(args)
    rospy.spin()
