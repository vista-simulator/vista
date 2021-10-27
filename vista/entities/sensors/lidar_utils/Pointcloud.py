from typing import Optional
from enum import Enum
import numpy as np


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

    Args:
        xyz (np.ndarray): x, y, z position of the point cloud.
        intensity (np.ndarray): Intensity of the point cloud.

    """
    def __init__(self,
                 xyz: np.ndarray,
                 intensity: Optional[np.ndarray] = None):
        super(Pointcloud, self).__init__()

        self._xyz = np.reshape(xyz, (-1, 3))
        self._intensity = np.zeros((self._xyz.shape[0], ))
        if intensity is not None:
            self._intensity[:] = intensity.ravel()

        self._dist: np.ndarray = None

    def transform(self,
                  R: Optional[np.ndarray] = None,
                  trans: Optional[np.ndarray] = None):
        """ Transform the point cloud.

        Args:
            R (np.ndarray): Rotation matrix with shape (3,3).
            trans (np.ndarray): Translation vector with length 3.

        Raises:
            AssertionError: Invalid rotation matrix (3,3) or translation (3,)

        """
        xyz = self.xyz

        if R is not None:
            assert R.shape == (3, 3), \
                   f"Rotation matrix shape is {R.shape} but should be (3, 3)"
            xyz = np.matmul(xyz, R)

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
            return np.ones((self.num_points(), ))

        raise ValueError(f"Unrecognized Point feature {feature} to" +
                         " extract from pointcloud")

    def __getitem__(self, i):
        new_pcd = Pointcloud(self.xyz[i], self.intensity[i])
        if self._dist is not None:
            new_pcd._dist = self._dist[i]
        return new_pcd

    def __len__(self):
        return self.xyz.shape[0]

    @property
    def num_points(self) -> int:
        """ Number of points. """
        return len(self)

    @property
    def x(self) -> np.ndarray:
        """ The `x` component of all points. """
        return self.xyz[:, 0]

    @property
    def y(self) -> np.ndarray:
        """ The `y` component of all points. """
        return self.xyz[:, 1]

    @property
    def z(self) -> np.ndarray:
        """ The `z` component of all points. """
        return self.xyz[:, 2]

    @property
    def xyz(self) -> np.ndarray:
        """ `xyz` of all points. """
        return self._xyz

    @property
    def intensity(self) -> np.ndarray:
        """ The intensity of all points. """
        return self._intensity

    @property
    def dist(self) -> np.ndarray:
        """ Distance to the origin of all points. """
        if self._dist is None:
            self._dist = np.linalg.norm(self._xyz, ord=2, axis=1)
        return self._dist
