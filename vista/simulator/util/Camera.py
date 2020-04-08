"""
The **Camera** object stores information pertaining to a single physical camera
mounted on the car. It is useful for encapsulating the relevant calibration
information for easy access in other modules.

"""

import numpy as np
import os
import xml.etree.ElementTree as ET
import pathlib

rig_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'RIG.xml')

def ignore_case(tree):
    t = ET.tostring(tree)
    t = t.lower()
    return ET.fromstring(t)

tree = ET.parse(rig_path)
root = ignore_case(tree.getroot())

xml_cameras = root.findall('sensors/camera')
names = [cam.get('name') for cam in xml_cameras]
cameras = dict(zip(names, xml_cameras))


class Camera(object):
    """Object to store camera calibration information."""

    def __init__(self, name):
        """Initialize the camera object
        Args:
            name (str): Name of the camera identifier to initialize. Must be
                a valid TopicName and present inside the RIG.xml file.

        Attributes:
            name (str): Name of the camera.

        Raises:
            ValueError: If `name` is not found in the rig file.
        """
        if name not in cameras.keys():
            raise ValueError("Camera.py> %s is not a valid camera within RIG.xml".format(name))

        self.name = name
        cam = cameras[self.name]
        xml_props = cam.findall('property')

        pname = [p.get('name') for p in xml_props]
        pvalue = [p.get('value') for p in xml_props]
        props = dict(zip(pname, pvalue))

        self._height = int(props['height'])
        self._width = int(props['width'])

        self._fx = float(props['fx'])
        self._fy = float(props['fy'])

        self._cx = float(props['cx'])
        self._cy = float(props['cy'])

        self._distortion = np.array( [float(x) for x in props['distortion'].split(" ")] )

        self._quaternion = np.array( [float(x) for x in props['quaternion'].split(" ")] ).reshape(4,1)
        self._position = np.array( [float(x) for x in props['position'].split(" ")] ).reshape(3,1)
        self._yaw = float(props['yaw']) if 'yaw' in props else None

        self._roi = np.array( [int(x) for x in props['roi'].split(" ")] )
        self._roi_angle = float(props['roi_angle'])*np.pi/180.

        self.__compute_other_forms()


    def resize(self, height, width):
        """ Scales the camera object and adjusts the internal parameters such
        that it projects images of a certain size.

        Args:
            height (int): new height of the camera images in pixels
            width (int): new width of the camera images in pixels

        Returns:
            None
        """

        scale_y = float(height)/self._height
        scale_x = float(width)/self._width

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
        roi = [roi[0]*scale_y, roi[1]*scale_x,
               roi[2]*scale_y, roi[3]*scale_x]
        self._roi = [int(x) for x in roi]

        self.__compute_other_forms()


    def crop(self, i1, j1, i2, j2):
        """ Crops a camera object to a given region of interest specified by
        the coordinates of the top left (i1,j1) and bottom right (i2,j2) corner

        Args:
            i1 (int): top row of ROI
            j1 (int): left column of ROI
            i2 (int): bottom row of ROI
            j2 (int): right column of ROI

        Returns:
            None
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
        self._roi = [self._roi[0]-i1, self._roi[1]-j1,
                     self._roi[2]-i2, self._roi[3]-j2]
        self._roi = [int(x) for x in self._roi]

        self.__compute_other_forms()

    def get_height(self):
        """Get the raw pixel height of images captured by the camera

        Returns:
            int: Height in pixels
        """
        return self._height

    def get_width(self):
        """Get the raw pixel width of i        <PROPERTY Name="roi_angle" Hint="angle of ROI" Value="-19.7"/>
mages captured by the camera

        Returns:
            int: Width in pixels
        """
        return self._width

    def get_K(self):
        """Get intrinsic calibration matrix

        Returns:
            np.array: Intrinsic matrix (3,3)
        """
        return self._K

    def get_K_inv(self):
        """Get inverse intrinsic calibration matrix

        Returns:
            np.array: Inverse intrinsic matrix (3,3)
        """
        return self._K_inv

    def get_distortion(self):
        """Get the distortion coefficients of the camera

        Returns:
            np.array: Distortion coefficients (-1,)
        """
        return self._distortion

    def get_position(self):
        """Get the 3D position of camera

        Returns:
            np.array: 3D position of camera
        """
        return self._position

    def get_yaw(self):
        """Get the yaw of the camera relative the frame of reference

        Returns:
            int: Yaw of the camera [rads]
        """
        if self._yaw is None:
            raise ValueError("camera {}, does not have a yaw in the rig file".format(self.name))
        return self._yaw

    def get_ground_plane(self):
        """Get the equation of the ground plane

        The equation of the ground plane is given by:
            Ax + By + Cz = D
        and is computed from the position and orientation of the camera

        Returns:
            list: Parameterization of the ground plane: [A,B,C,D]
        """
        return self._ground_plane

    def get_roi(self, axis='ij'):
        """ Get the region of interest of the images captured by the camera.

        Args:
            axis (str): axis order to return the coordinates in (default 'ij',
            can also be 'xy')

        Returns:
            list: coordinates of the ROI box

        Raises:
            ValueError: if `axis` is not valid
        """

        if axis == 'ij':
            return self._roi
        elif axis == 'xy':
            return [ self._roi[i] for i in [1,2,3,0] ]
        else:
            raise ValueError("invalid axis: "+axis)

    def get_roi_angle(self):
        """ Get the angle of the region of interest.

        Returns:
            angle: the rotation of the ROI box

        """
        return self._roi_angle

    def get_roi_points(self):
        """ Get the points of the region of interest.

        Returns:
            pts: the list of points surrounding the ROI box

        """
        return self._roi_points

    def get_roi_dims(self):
        """ Get the dimensions of the region of interest.

        Returns:
            dim: height width of ROI

        """
        dims = (int(self._roi_height), int(self._roi_width))
        return dims if self._roi_angle<np.pi/4. else dims[::-1]
    def __compute_other_forms(self):
        self.__compute_intrinsic_matrix()
        self.__compute_ground_plane()
        self.__compute_roi()

    def __compute_intrinsic_matrix(self):

        self._K = np.array([
            [self._fx,   0,          self._cx],
            [0,          self._fy,   self._cy],
            [0,          0,          1      ]
        ])
        self._K_inv = np.linalg.inv(self._K)

    def __compute_ground_plane(self):
        q = self._quaternion.flatten().tolist()
        p = self._position

        normal = [ 2.*(q[1]*q[2] + q[3]*q[1]),  1-2.*(q[0]*q[0] + q[2]*q[2]),  2.*(q[0]*q[1] - q[2]*q[3]) ]
        intercept = np.dot(p.T, normal)

        self._ground_plane = [normal[0], normal[1], normal[2], intercept[0]]

    def __compute_roi(self):
        t = self._roi_angle
        (i1,j1,i2,j2) = self._roi
        W = j2 - j1
        H = i2 - i1
        self._roi_width = (1./(np.cos(t)**2-np.sin(t)**2)) * (  W * np.cos(t) - H * abs(np.sin(t)))
        self._roi_height = (1./(np.cos(t)**2-np.sin(t)**2)) * (- W * abs(np.sin(t)) + H * np.cos(t))
        if t < 0: #reverse persective
            self._roi_points = [
                (abs(self._roi_height*np.cos(t)), 0),
                (H, abs(self._roi_width*np.cos(t))),
                (abs(self._roi_width*np.sin(t)), W),
                (0, abs(self._roi_height*np.sin(t))),
            ]
        else:
            self._roi_points = [
                (abs(self._roi_width*np.sin(t)), 0),
                (H, abs(self._roi_height*np.sin(t))),
                (abs(self._roi_height*np.cos(t)), W),
                (0, abs(self._roi_width*np.cos(t))),
            ]
        self._roi_points = [(i+i1,j+j1) for (i,j) in self._roi_points]
        self._roi_points = [(j, i) for (i,j) in self._roi_points]
        self._roi_points = np.int32([self._roi_points])
