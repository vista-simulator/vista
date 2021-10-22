""" A minimal example of using VISTA to achieve augmented reality (AR)
    by subscribing to ROS topics. """
import os
import numpy as np
import rospy
import utm
from sensor_msgs.msg import Image, Imu, NavSatFix
from cv_bridge import CvBridge

from vista.entities.sensors.camera_utils import ViewSynthesis, CameraParams
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.entities.sensors.MeshLib import MeshLib
from vista.utils import transform

import OpenGL.EGL as egl
from OpenGL import error
from OpenGL.EGL.EXT.device_base import egl_get_devices
from OpenGL.raw.EGL.EXT.platform_device import EGL_PLATFORM_DEVICE_EXT


def create_initialized_headless_egl_display():
    """Creates an initialized EGL display directly on a device."""
    for device in egl_get_devices():
        display = egl.eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, device,
                                               None)

        if display != egl.EGL_NO_DISPLAY and egl.eglGetError(
        ) == egl.EGL_SUCCESS:
            # `eglInitialize` may or may not raise an exception on failure depending
            # on how PyOpenGL is configured. We therefore catch a `GLError` and also
            # manually check the output of `eglGetError()` here.
            try:
                initialized = egl.eglInitialize(display, None, None)
            except error.GLError:
                pass
            else:
                if initialized == egl.EGL_TRUE and egl.eglGetError(
                ) == egl.EGL_SUCCESS:
                    return display
    return egl.EGL_NO_DISPLAY


class VistaARNode(object):
    def __init__(self, node_name):
        self._node_name = node_name
        self._cvbr = CvBridge()

        # Cached data stream for running Vista
        self._pose = np.zeros((3, ))  # gps_x, gps_y, yaw
        self._static_obstacle_pose = np.zeros((3, ))

        # Instantiate Vista objects required for AR
        camera_name = 'front_center'
        camera_size = (600, 960)
        rig_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'RIG.xml')
        self._camera_param = CameraParams(camera_name, rig_path)
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
            long_shift = 10.
            theta = self._pose[2]  # TODO: need to get a global yaw
            self._static_obstacle_pose[0] += long_shift * np.sin(theta)
            self._static_obstacle_pose[1] += long_shift * np.cos(theta)

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

        # import cv2 # DEBUG
        # cv2.imwrite('test.png', img_ar)
        # import pdb
        # pdb.set_trace()

        # Publish AR image
        msg_ar = self._cvbr.cv2_to_imgmsg(img_ar)
        self._pub.publish(msg_ar)

    def _gps_callback(self, msg):
        x, y, _, _ = utm.from_latlon(msg.latitude, msg.longitude)
        self._pose[:2] = np.array([x, y])

    def _imu_callback(self, msg):
        q = msg.orientation
        yaw_y = 2. * (q.x * q.y + q.z * q.w)
        yaw_x = q.w**2 - q.z**2 - q.y * 2 + q.x**2
        self._pose[2] = np.arctan2(yaw_y, yaw_x)

    def _loginfo(self, msg):
        rospy.loginfo(f'[{self.node_name}] {msg}')

    @property
    def node_name(self):
        return self._node_name

    @property
    def camera_param(self):
        return self._camera_param


if __name__ == '__main__':
    node_name = 'vista_ar_node'
    rospy.init_node(node_name, anonymous=False)
    vista_ar_node = VistaARNode(node_name)
    rospy.spin()
