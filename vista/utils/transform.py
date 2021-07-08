""" Helper functions for spatial transformation. Follow right handed 
    (OpenGL) coordinate system. """
from typing import Optional, Union, List, Tuple, Any
import numpy as np
from scipy.spatial.transform import Rotation


Vec = Union[np.ndarray, List[Any], Tuple[Any]]


def vec2mat(trans: Vec, rot: Vec) -> np.ndarray:
    """ Convert translation and rotation vector to transformation matrix. """
    mat = np.eye(4)
    mat[:3,3] = trans
    R = Rotation.from_euler('xyz', rot)
    mat[:3,:3] = R.as_matrix()
    return mat


def euler2quat(euler: Vec, seq: Optional[str] = 'xyz', 
               degrees: Optional[bool] = False) -> Vec:
    """ Convert Euler rotation to quaternion. """
    R = Rotation.from_euler(seq, euler, degrees)
    return R.as_quat()


def quat2euler(quat: Vec, seq: Optional[str] = 'xyz', 
               degrees: Optional[bool] = False) -> Vec:
    """ Convert quaternion to Euler rotation. """
    R = Rotation.from_quat(quat)
    return R.as_euler(seq, degrees)


def compute_relative_xyyaw(xyyaw1, xyyaw2):
    """ Compute relative x, y, yaw. """
    raise NotImplementedError


def mat2vec(tr):
    """ Convert transformation matrix to translational and rotational vectors. """
    raise NotImplementedError
