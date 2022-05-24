""" Helper functions for spatial transformation. Follow right handed
    (OpenGL) coordinate system. """
from typing import Optional, Union, List, Tuple, Any
import numpy as np
from scipy.spatial.transform import Rotation

Vec = Union[np.ndarray, List[Any], Tuple[Any]]


def rot2mat(rot: Vec, seq: Optional[str] = 'xyz') -> np.ndarray:
    """ Convert euler vector (with sequence order) to rotation matrix.

    Args:
        rot (Vec): A 3-dimensional rotation vector in Euler angle.
        seq (str): The order of the rotation vector, e.g., ``xzy``;
                   default to ``xyz``.

    Returns:
        np.ndarray: A rotation matrix equivalent to the given rotation vector.

    """
    R = Rotation.from_euler(seq, rot)
    return R.as_matrix()


def vec2mat(trans: Vec, rot: Vec) -> np.ndarray:
    """ Convert translation and rotation vector to transformation matrix.

    Args:
        trans (Vec): A 3-dimensional translation vector.
        rot (Vec): A 3-dimensional rotation vector.

    Returns:
        np.ndarray: A 4-by-4 transformation matrix in SE(3).

    """
    mat = np.eye(4)
    mat[:3, 3] = trans
    mat[:3, :3] = rot2mat(rot)
    return mat


def euler2quat(euler: Vec,
               seq: Optional[str] = 'xyz',
               degrees: Optional[bool] = False) -> Vec:
    """ Convert Euler rotation to quaternion.

    Args:
        euler (Vec): A 3-dimensional rotation vector in Euler angle.
        seq (str): The order of the rotation vector, e.g., ``xzy``;
                   default to ``xyz``.

    Returns:
        Vec: A 4-dimensional vector that describes a quaternion.

    """
    R = Rotation.from_euler(seq, euler, degrees)
    return R.as_quat()


def quat2euler(quat: Vec,
               seq: Optional[str] = 'xyz',
               degrees: Optional[bool] = False) -> Vec:
    """ Convert quaternion to Euler rotation.

    Args:
        quat (Vec): A 4-dimensional vector that describes a quaternion.
        seq (str): The order of the output Euler rotation vector; default
                   to ``xyz``.
        degrees (bool): Whether to convert the output rotation to degrees
                        instead of radians; default to ``False``.

    Returns:
        Vec: A 3-dimensional rotation vector in Euler angle.

    """
    R = Rotation.from_quat(quat)
    return R.as_euler(seq, degrees)


def latlongyaw2vec(latlongyaw: Vec) -> Tuple[Vec, Vec]:
    """ Convert lateral, longitudinal, yaw compoenents to translational and
    rotational vectors. Note that the longitudinal component will be negated
    to follow right-handed OpenGL coordinate system.

    Args:
        latlongyaw (Vec): A 3-dimensional vector with entries as lateral shift,
                          longitudinal shift, and yaw difference.

    Returns:
        Return a tuple (``Vec_a``, ``Vec_b``), where ``Vec_a`` is a 3-dimensional
        translation vector and ``Vec_b`` is a 3-dimensional rotation vector in
        Euler angle ``xyz``.

    """
    lat, long, yaw = latlongyaw
    trans = np.array([lat, 0., -long])
    rot = np.array([0., yaw, 0.])
    return trans, rot


def vec2latlongyaw(trans: Vec, rot: Vec) -> Vec:
    """ Convert translational and rotational vectors to lateral,
    longitudinal, yaw compoenents. Note that the y translational
    component will be negated to follow right-handed OpenGL
    coordinate system.

    Args:
        trans (Vec): A 3-dimensional translation vector.
        rot (Vec): A 3-dimensional Euler rotation vector.

    Returns:
        Vec: A 3-dimensional vector with entries as lateral shift,
             longitudinal shift, and yaw difference.

    """
    return np.array([trans[0], -trans[2], rot[1]])


def compute_relative_latlongyaw(latlongyaw: Vec, latlongyaw_ref: Vec) -> Vec:
    """ Compute relative lateral, longitudinal, yaw compoenents.

    Args:
        latlongyaw (Vec): A 3-dimensional vector with lateral,
                          longitudinal, and yaw component.
        latlongyaw_ref (Vec): A reference 3-dimensional vector with lateral,
                              longitudinal, and yaw component.

    Returns:
        Vec: A 3-dimensional vector that describe relative lateral,
             longitudinal shift and yaw difference between the given
             two vectors.

    """
    mat = vec2mat(*latlongyaw2vec(latlongyaw))
    mat_ref = vec2mat(*latlongyaw2vec(latlongyaw_ref))
    rel_mat = np.matmul(SE3_inv(mat_ref), mat)
    rel_trans, rel_rot = mat2vec(rel_mat)
    rel_xyyaw = vec2latlongyaw(rel_trans, rel_rot)
    return rel_xyyaw


def SE3_inv(T_in: np.ndarray) -> np.ndarray:
    """ More efficient matrix inversion for SE(3).

    Args:
        T_in (np.ndarray): a 4-by-4 transformation matrix in SE(3).

    Returns:
        np.ndarray: The inverse of ``T_in``.

    """
    R_in = T_in[:3, :3]
    t_in = T_in[:3, [-1]]
    R_out = R_in.T
    t_out = -np.matmul(R_out, t_in)
    return np.vstack((np.hstack((R_out, t_out)), np.array([0, 0, 0, 1])))


def mat2vec(mat: np.ndarray,
            seq: Optional[str] = 'xyz',
            degrees: Optional[bool] = False) -> Tuple[Vec, Vec]:
    """ Convert transformation matrix to translational and rotational vectors.

    Args:
        mat (np.ndarray): A 4-by-4 transformation matrix in SE(3).
        seq (str): The order of the output Euler rotation vector; default
                   to ``xyz``.
        degrees (bool): Whether to convert the output rotation to degrees
                        instead of radians; default to ``False``.

    Returns:
        Return a tuple (``Vec_a``, ``Vec_b``), where ``Vec_a`` is a 3-dimensional
        translation vector and ``Vec_b`` is a 3-dimensional rotation vector in
        Euler angle ``xyz``.

    """
    trans = mat[:3, 3]
    R = Rotation.from_matrix(mat[:3, :3])
    rot = R.as_euler(seq, degrees)
    return trans, rot


def pi2pi(angle: float) -> float:
    """ Make sure angle is within -pi to pi.

    Args:
        angle (float): Input angle in radians.

    Returns:
        float: Output angle that is within range [-pi, pi]

    """
    if angle >= np.pi:
        angle -= 2 * np.pi
    if angle <= -np.pi:
        angle += 2 * np.pi
    return angle
