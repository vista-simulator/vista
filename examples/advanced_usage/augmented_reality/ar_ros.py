""" A minimal example of using VISTA to achieve augmented reality (AR)
    by subscribing to ROS topics. """
import os
import numpy as np
import argparse
import rospy
import utm
import pickle
import cv2
from sensor_msgs.msg import Image, Imu, NavSatFix
from cv_bridge import CvBridge

from vista.entities.sensors.camera_utils import ViewSynthesis, CameraParams
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.entities.sensors.MeshLib import MeshLib
from vista.utils import transform


class VistaARNode(object):
    def __init__(self, node_name, devens_road=None, visualize=False):
        self._node_name = node_name
        self._devens_road = devens_road
        self._visualize = visualize
        self._cvbr = CvBridge()

        # Cached data stream for running Vista
        self._pose = np.zeros((3, ))  # gps_x, gps_y, yaw
        self._static_obstacle_pose = np.zeros((3, ))
        self._gps_time = 0.
        self._imu_time = 0.

        # Instantiate Vista objects required for AR
        camera_name = 'front_center'
        camera_size = (600, 960)
        rig_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'RIG.xml')
        self._camera_param = CameraParams(rig_path, camera_name)
        self._camera_param.resize(*camera_size)

        self._vs_config = dict(depth_mode=DepthModes.FIXED_PLANE, )
        self._vs = None

        n_agents = 2
        mesh_dir = os.path.expanduser('~/data/single_carpack01/')
        self._meshlib = MeshLib(mesh_dir)
        self._meshlib.reset(n_agents)

        # Define subscriber and publisher
        self._sub_image = rospy.Subscriber(
            f'camera_array/{camera_name}/image_raw',
            Image,
            self._image_callback,
            queue_size=1,
            buff_size=695820800)
        self._sub_gps = rospy.Subscriber('/lexus/oxts/gps/fix',
                                         NavSatFix,
                                         self._gps_callback,
                                         queue_size=1)
        self._sub_imu = rospy.Subscriber('/lexus/oxts/imu/data',
                                         Imu,
                                         self._imu_callback,
                                         queue_size=1)
        self._pub = rospy.Publisher('vista/ar_image', Image, queue_size=1)

    def _image_callback(self, msg):
        # NOTE: must initialize here otherwise will cause error in rendering
        if self._vs is None:
            self._vs = ViewSynthesis(self.camera_param, self._vs_config)

        # Get raw image
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, 3))

        # Update static obstacle pose
        if (np.allclose(self._pose, np.zeros((3, )))
                or np.allclose(self._static_obstacle_pose, np.zeros((3, )))):
            self._static_obstacle_pose = self._pose.copy()
            # place static obstacle in the front of ego car
            long_shift = 30.
            theta = -self._pose[2]
            if self._devens_road is None:
                self._static_obstacle_pose[0] += long_shift * np.sin(theta)
                self._static_obstacle_pose[1] += long_shift * np.cos(theta)
            else:
                dist_to_devens_road = np.linalg.norm(self._pose[:2] -
                                                     self._devens_road,
                                                     axis=1)
                idx = np.argmin(dist_to_devens_road)

                small_step_forward = self._pose[:2] + np.array(
                    [np.sin(theta), np.cos(theta)])
                small_step_forward_idx = np.argmin(
                    np.linalg.norm(small_step_forward - self._devens_road,
                                   axis=1))

                if small_step_forward_idx >= idx:
                    dist_to_devens_road[:idx] += np.inf
                else:
                    dist_to_devens_road[idx:] += np.inf
                tgt_idx = np.argmin(np.abs(dist_to_devens_road - long_shift))
                self._static_obstacle_pose[:2] = self._devens_road[tgt_idx]

        # Update object in Vista
        agent_idx = 1
        latlongyaw = transform.compute_relative_latlongyaw(
            self._static_obstacle_pose, self._pose)
        name = f'agent_{agent_idx}'
        trans, rotvec = transform.latlongyaw2vec(latlongyaw)
        quat = transform.euler2quat(rotvec)
        self._vs.update_object_node(name,
                                    self._meshlib.agents_meshes[agent_idx],
                                    trans, quat)

        # Run AR with Vista
        trans = np.zeros((3, ))
        rot = np.zeros((3, ))
        img_ar = self._vs.synthesize(trans, rot,
                                     {self.camera_param.name: img})[0]

        if self._visualize:
            cv2.imshow('Vista (AR)', img_ar)
            cv2.waitKey(20)

        # Publish AR image
        msg_ar = self._cvbr.cv2_to_imgmsg(img_ar)
        self._pub.publish(msg_ar)

    def _gps_callback(self, msg):
        x, y, _, _ = utm.from_latlon(msg.latitude, msg.longitude)
        self._pose[:2] = np.array([x, y])
        self._gps_time = msg.header.stamp.to_sec()

    def _imu_callback(self, msg):
        q = msg.orientation
        yaw_y = 2. * (q.x * q.y + q.z * q.w)
        yaw_x = q.w**2 - q.z**2 - q.y * 2 + q.x**2
        self._pose[2] = np.arctan2(yaw_y, yaw_x)
        self._imu_time = msg.header.stamp.to_sec()

    def _loginfo(self, msg):
        rospy.loginfo(f'[{self.node_name}] {msg}')

    @property
    def node_name(self):
        return self._node_name

    @property
    def camera_param(self):
        return self._camera_param


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use Vista for AR')
    parser.add_argument('--devens-road-path',
                        type=str,
                        default=None,
                        help='Path to devens road pickle file')
    parser.add_argument('--visualize',
                        action='store_true',
                        default=False,
                        help='Visualize the frame')
    args = parser.parse_args()
    if args.devens_road_path is not None:
        devens_road = np.genfromtxt(args.devens_road_path,
                                    delimiter=',',
                                    skip_header=1)
        devens_road = np.array(
            [utm.from_latlon(_x[0], _x[1])[:2] for _x in devens_road[:, 1:3]])
    else:
        devens_road = None

    node_name = 'vista_ar_node'
    rospy.init_node(node_name, anonymous=False)
    vista_ar_node = VistaARNode(node_name, devens_road, args.visualize)
    rospy.spin()
