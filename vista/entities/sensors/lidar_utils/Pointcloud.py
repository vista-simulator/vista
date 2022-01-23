from typing import Optional, Union
from enum import Enum
import numpy as np
import torch

tensor_or_ndarray = Union[torch.Tensor, np.ndarray]


class Point(Enum):
    """ Point feature, including x, y, z, intensity, depth, and mask. """
    INTENSITY = "intensity"
    DEPTH = "depth"
    MASK = "mask"
    X = "x"
    Y = "y"
    Z = "z"


class Pointcloud:
    """ A helper class that allow handling point cloud more easily with functionality
    like transforming point cloud and extracting features/properties from point cloud.
    Pointcloud can be built from either ``numpy.ndarray`` or ``torch.Tensor`` data.
    Methods will maintain the same data type.

    Args:
        xyz (tensor): x, y, z position of the point cloud. shape of ``(N,3)``
        intensity (tensor): Intensity of the point cloud. shape ``(N,)``

    """
    def __init__(self,
                 xyz: tensor_or_ndarray,
                 intensity: Optional[tensor_or_ndarray] = None):
        super(Pointcloud, self).__init__()

        self.with_torch = isinstance(xyz, torch.Tensor)
        self._xyz = xyz.reshape((-1, 3))
        self._intensity = 0.0 * self._xyz[:, 0]  # default to all zeros
        if intensity is not None:
            self._intensity[:] = intensity.ravel()

        self._dist: type(xyz) = None
        self._yaw: type(xyz) = None
        self._pitch: type(xyz) = None

    def transform(self,
                  R: Optional[tensor_or_ndarray] = None,
                  trans: Optional[tensor_or_ndarray] = None):
        """ Transform the point cloud.

        Args:
            R (tensor): Rotation matrix with shape (3,3).
            trans (tensor): Translation vector with length 3.

        Raises:
            AssertionError: Invalid rotation matrix (3,3) or translation (3,)

        """
        xyz = self.xyz

        if R is not None:
            assert R.shape == (3, 3), \
                   f"Rotation matrix shape is {R.shape} but should be (3, 3)"
            xyz = xyz @ R

        if trans is not None:
            assert trans.size == 3, \
                   f"trans size is {trans.size} but should be 3"
            trans = trans.reshape(1, 3)
            xyz = xyz + trans

        new_pcd = Pointcloud(xyz, self.intensity)
        return new_pcd

    def get(self, feature: Point) -> np.ndarray:
        """ Get feature (x, y, z, intensity, depth, mask) of the point cloud.

        Args:
            feature (Point): Feature to extract from the point cloud.

        Returns:
            np.ndarray: Point feature.

        Raises:
            ValueError: Unrecognized Point feature.

        """
        feature = Point(feature)  # Cast to a Point if not already
        if feature == Point.X:
            return self.x
        if feature == Point.Y:
            return self.y
        if feature == Point.Z:
            return self.z
        if feature == Point.INTENSITY:
            return self.intensity
        if feature == Point.DEPTH:
            return self.dist
        if feature == Point.MASK:
            ones = torch.ones if self.with_torch else np.ones
            return ones((self.num_points, ))

        raise ValueError(f"Unrecognized Point feature {feature} to" +
                         " extract from pointcloud")

    def __getitem__(self, i):
        new_pcd = Pointcloud(self.xyz[i], self.intensity[i])
        if self._dist is not None:
            new_pcd._dist = self._dist[i]
        return new_pcd

    def __len__(self):
        return self.xyz.shape[0]

    def numpy(self):
        """ Returns a copy of the torch pointcloud built using numpy. If the
        pointcloud is already in numpy format, then a copy is returned. """

        xyz = self.xyz
        intensity = self.intensity
        if self.with_torch:
            xyz = xyz.detach().cpu().numpy()
            intensity = intensity.detach().cpu().numpy()
        pcd_np = Pointcloud(xyz.copy(), intensity.copy())
        if self._dist is not None:
            dist = self._dist
            dist = dist.detach().cpu().numpy() if self.with_torch else dist
            pcd_np._dist = dist.copy()
        return pcd_np

    @property
    def num_points(self) -> int:
        """ Number of points. """
        return len(self)

    @property
    def x(self) -> tensor_or_ndarray:
        """ The `x` component of all points. """
        return self.xyz[:, 0]

    @property
    def y(self) -> tensor_or_ndarray:
        """ The `y` component of all points. """
        return self.xyz[:, 1]

    @property
    def z(self) -> tensor_or_ndarray:
        """ The `z` component of all points. """
        return self.xyz[:, 2]

    @property
    def xyz(self) -> tensor_or_ndarray:
        """ `xyz` of all points. """
        return self._xyz

    @property
    def intensity(self) -> tensor_or_ndarray:
        """ The intensity of all points. """
        return self._intensity

    @property
    def dist(self) -> tensor_or_ndarray:
        """ Distance to the origin of all points. """
        if self._dist is None:
            if self.with_torch:
                self._dist = torch.norm(self._xyz, p=2, dim=1)
            else:
                self._dist = np.linalg.norm(self._xyz, ord=2, axis=1)
        return self._dist

    @property
    def yaw(self) -> tensor_or_ndarray:
        """ Yaw angle (radians) of each point in the cloud. """
        if self._yaw is None:
            atan2 = torch.atan2 if self.with_torch else np.arctan2
            self._yaw = atan2(self.y, self.x)
        return self._yaw

    @property
    def pitch(self) -> tensor_or_ndarray:
        """ Pitch angle (radians) of each point in the cloud. """
        if self._pitch is None:
            arcsin = torch.arcsin if self.with_torch else np.arcsin
            self._pitch = arcsin(self.z / self.dist)
        return self._pitch

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} - ' + \
               f'#points: {self.num_points}' + '>'
