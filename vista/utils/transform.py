""" Helper functions for spatial transformation. Follow right handed
    (OpenGL) coordinate system. """
from typing import Optional, Union, List, Tuple, Any
import numpy as np
from scipy.spatial.transform import Rotation

Vec = Union[np.ndarray, List[Any], Tuple[Any]]


def vec2mat(trans: Vec, rot: Vec) -> np.ndarray:
    """ Convert translation and rotation vector to transformation matrix. """
    mat = np.eye(4)
    mat[:3, 3] = trans
    R = Rotation.from_euler('xyz', rot)
    mat[:3, :3] = R.as_matrix()
    return mat


def euler2quat(euler: Vec,
               seq: Optional[str] = 'xyz',
               degrees: Optional[bool] = False) -> Vec:
    """ Convert Euler rotation to quaternion. """
    R = Rotation.from_euler(seq, euler, degrees)
    return R.as_quat()


def quat2euler(quat: Vec,
               seq: Optional[str] = 'xyz',
               degrees: Optional[bool] = False) -> Vec:
    """ Convert quaternion to Euler rotation. """
    R = Rotation.from_quat(quat)
    return R.as_euler(seq, degrees)


def latlongyaw2vec(latlongyaw: Vec) -> Tuple[Vec, Vec]:
    """ Convert lateral, longitudinal, yaw compoenents to translational and rotational vectors. """
    lat, long, yaw = latlongyaw
    trans = np.array([lat, 0., -long])
    rot = np.array([0., yaw, 0.])
    return trans, rot


def vec2latlongyaw(trans: Vec, rot: Vec) -> Vec:
    """ Convert translational and rotational vectors to lateral, longitudinal, yaw compoenents. """
    return np.array([trans[0], -trans[2], rot[1]])


def compute_relative_latlongyaw(latlongyaw: Vec, latlongyaw_ref: Vec) -> Vec:
    """ Compute relative lateral, longitudinal, yaw compoenents. """
    mat = vec2mat(*latlongyaw2vec(latlongyaw))
    mat_ref = vec2mat(*latlongyaw2vec(latlongyaw_ref))
    rel_mat = np.matmul(SE3_inv(mat_ref), mat)
    # print(rel_mat) # DEBUG
    # import pdb; pdb.set_trace()
    rel_trans, rel_rot = mat2vec(rel_mat)
    rel_xyyaw = vec2latlongyaw(rel_trans, rel_rot)
    return rel_xyyaw


def SE3_inv(T_in):
    """ More efficient matrix inversion for SE(3) """
    R_in = T_in[:3,:3]
    t_in = T_in[:3,[-1]]
    R_out = R_in.T
    t_out = -np.matmul(R_out,t_in)
    return np.vstack((np.hstack((R_out,t_out)),np.array([0, 0, 0, 1])))


def mat2vec(mat,
            seq: Optional[str] = 'xyz',
            degrees: Optional[bool] = False) -> Tuple[Vec, Vec]:
    """ Convert transformation matrix to translational and rotational vectors. """
    trans = mat[:3, 3]
    R = Rotation.from_matrix(mat[:3, :3])
    rot = R.as_euler(seq, degrees)
    return trans, rot
