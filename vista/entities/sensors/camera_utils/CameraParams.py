from typing import Optional, List, Tuple
import numpy as np
import os
import xml.etree.ElementTree as ET
import pathlib
from ....utils.parse_params import ParamsFile


class CameraParams(object):
    """ The **CameraParams** object stores information pertaining to a single physical
    camera mounted on the car. It is useful for encapsulating the relevant calibration
    information for easy access in other modules.

    Args:
        rig_path (str): Path to RIG.xml that specifies camera parameters.
        name (str): Name of the camera identifier to initialize. Must be
                    a valid TopicName and present inside the RIG.xml file.
                    Can also specify `None` to auto grab the first named camera
                    in the RIG.xml file.
        params (dict): Dictionary camera parameters to instantiate with. If
                       not provided then the rig_path is used

    Raises:
        ValueError: if `name` is provided but not found in the rig file.

    """
    def __init__(self,
                 rig_path: str = None,
                 name: str = None,
                 params: dict = None):

        assert (rig_path is not None) ^ (params is not None)
        if rig_path is not None:
            pfile = ParamsFile(rig_path)
            params, self.name = pfile.parse_camera(name)

        self._height = int(params['height'])
        self._width = int(params['width'])

        self._fx = params['fx']
        self._fy = params['fy']

        self._cx = params['cx']
        self._cy = params['cy']

        self._distortion = params['distortion']
        self._quaternion = params['quaternion'].reshape(4, 1)
        self._position = params['position'].reshape(3, 1)
        self._yaw = params['yaw'] if 'yaw' in params else None

        self._roi = params['roi'].astype(np.int)
        self._roi_angle = params['roi_angle'] * np.pi / 180.

        self.__compute_other_forms()

    def resize(self, height: int, width: int) -> None:
        """ Scales the camera object and adjusts the internal parameters such
        that it projects images of a certain size.

        Args:
            height (int): New height of the camera images in pixels.
            width (int): New width of the camera images in pixels.

        """

        scale_y = float(height) / self._height
        scale_x = float(width) / self._width

        # Update height and width
        self._width = int(width)
        self._height = int(height)

        # Scale focal length
        self._fx *= scale_x
        self._fy *= scale_y

        # Scale optical center
        self._cx *= scale_x
        self._cy *= scale_y

        # ROI
        roi = [float(x) for x in self._roi]
        roi = [
            roi[0] * scale_y, roi[1] * scale_x, roi[2] * scale_y,
            roi[3] * scale_x
        ]
        self._roi = [int(x) for x in roi]

        self.__compute_other_forms()

    def crop(self, i1: int, j1: int, i2: int, j2: int) -> None:
        """ Crops a camera object to a given region of interest specified by
        the coordinates of the top left (i1,j1) and bottom right (i2,j2) corner.

        Args:
            i1 (int): Top row of ROI.
            j1 (int): Left column of ROI.
            i2 (int): Bottom row of ROI.
            j2 (int): Right column of ROI.

        """

        # Focal length stays the same
        self._fx = self._fx
        self._fy = self._fy

        # Height, width adjusted to ROI
        self._width = int(j2 - j1)
        self._height = int(i2 - i1)

        # Translate optical center to new frame of reference
        # NOTE: optical center (_cx,_cy) is measured from the bottom-right
        #   corner of the image.
        self._cx -= j1
        self._cy -= i2

        # ROI
        self._roi = [
            self._roi[0] - i1, self._roi[1] - j1, self._roi[2] - i2,
            self._roi[3] - j2
        ]
        self._roi = [int(x) for x in self._roi]

        self.__compute_other_forms()

    def get_height(self) -> int:
        """ Get the raw pixel height of images captured by the camera.

        Returns:
            int: Height in pixels.

        """
        return self._height

    def get_width(self) -> int:
        """ Get the raw pixel width of images captured by the camera.

        Returns:
            int: Width in pixels.

        """
        return self._width

    def get_K(self) -> np.ndarray:
        """ Get intrinsic calibration matrix.

        Returns:
            np.array: Intrinsic matrix (3,3).

        """
        return self._K

    def get_K_inv(self) -> np.ndarray:
        """ Get inverse intrinsic calibration matrix.

        Returns:
            np.array: Inverse intrinsic matrix (3,3).

        """
        return self._K_inv

    def get_distortion(self) -> np.ndarray:
        """ Get the distortion coefficients of the camera.

        Returns:
            np.array: Distortion coefficients (-1,).

        """
        return self._distortion

    def get_position(self) -> np.ndarray:
        """ Get the 3D position of camera.

        Returns:
            np.array: 3D position of camera.

        """
        return self._position

    def get_quaternion(self) -> np.ndarray:
        """ Get the rotation in quaternion of camera.

        Returns:
            np.array: Rotation in quaternion of camera.

        """
        return self._quaternion

    def get_yaw(self) -> float:
        """ Get the yaw of the camera relative the frame of reference.

        Returns:
            float: Yaw of the camera [rads].

        """
        if self._yaw is None:
            raise ValueError(
                f'camera {self.name}, does not have a yaw in the rig file')
        return self._yaw

    def get_ground_plane(self) -> List[float]:
        """ Get the equation of the ground plane.

        The equation of the ground plane is given by: ``Ax + By + Cz = D``
        and is computed from the position and orientation of the camera.

        Returns:
            List[float]: Parameterization of the ground plane: [A,B,C,D].

        """
        return self._ground_plane

    def get_roi(self, axis: Optional[str] = 'ij') -> List[int]:
        """ Get the region of interest of the images captured by the camera.

        Args:
            axis (str): Axis order to return the coordinates in (default 'ij',
            can also be 'xy').

        Returns:
            List[int]: Coordinates of the ROI box.

        Raises:
            ValueError: If `axis` is not valid.

        """

        if axis == 'ij':
            return self._roi
        elif axis == 'xy':
            return [self._roi[i] for i in [1, 2, 3, 0]]
        else:
            raise ValueError("invalid axis: " + axis)

    def get_roi_angle(self) -> float:
        """ Get the angle of the region of interest.

        Returns:
            float: The rotation of the ROI box.

        """
        return self._roi_angle

    def get_roi_points(self) -> List:
        """ Get the points of the region of interest.

        Returns:
            List: the list of points surrounding the ROI box

        """
        return self._roi_points

    def get_roi_dims(self) -> Tuple:
        """ Get the dimensions of the region of interest.

        Returns:
            Tuple: Height and width of ROI.

        """
        dims = (int(self._roi_height), int(self._roi_width))
        return dims if self._roi_angle < np.pi / 4. else dims[::-1]

    def __compute_other_forms(self):
        self.__compute_intrinsic_matrix()
        self.__compute_ground_plane()
        self.__compute_roi()

    def __compute_intrinsic_matrix(self):

        self._K = np.array([[self._fx, 0, self._cx], [0, self._fy, self._cy],
                            [0, 0, 1]])
        self._K_inv = np.linalg.inv(self._K)

    def __compute_ground_plane(self):
        q = self._quaternion.flatten().tolist()
        p = self._position

        normal = [
            2. * (q[1] * q[2] + q[3] * q[1]),
            1 - 2. * (q[0] * q[0] + q[2] * q[2]),
            2. * (q[0] * q[1] - q[2] * q[3])
        ]
        intercept = np.dot(p.T, normal)

        self._ground_plane = [normal[0], normal[1], normal[2], intercept[0]]

    def __compute_roi(self):
        t = self._roi_angle
        (i1, j1, i2, j2) = self._roi
        W = j2 - j1
        H = i2 - i1
        self._roi_width = (1. / (np.cos(t)**2 - np.sin(t)**2)) * (
            W * np.cos(t) - H * abs(np.sin(t)))
        self._roi_height = (1. / (np.cos(t)**2 - np.sin(t)**2)) * (
            -W * abs(np.sin(t)) + H * np.cos(t))
        if t < 0:  #reverse persective
            self._roi_points = [
                (abs(self._roi_height * np.cos(t)), 0),
                (H, abs(self._roi_width * np.cos(t))),
                (abs(self._roi_width * np.sin(t)), W),
                (0, abs(self._roi_height * np.sin(t))),
            ]
        else:
            self._roi_points = [
                (abs(self._roi_width * np.sin(t)), 0),
                (H, abs(self._roi_height * np.sin(t))),
                (abs(self._roi_height * np.cos(t)), W),
                (0, abs(self._roi_width * np.cos(t))),
            ]
        self._roi_points = [(i + i1, j + j1) for (i, j) in self._roi_points]
        self._roi_points = [(j, i) for (i, j) in self._roi_points]
        self._roi_points = np.int32([self._roi_points])
